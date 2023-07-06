import argparse
import copy
import os
from functools import partial
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import torch
from matplotlib.patches import Rectangle
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel

import mmcv
from gpt4roi.train.train import preprocess, preprocess_multimodal
from gpt4roi.utils import build_det_model_from_cfg, inf_single_image
from llava.model.utils import KeywordsStoppingCriteria
from llava.utils import disable_torch_init

DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'

multimodal_cfg = {'is_multimodal': True,
                  'sep_image_conv_front': False,
                  'image_token_len': 256,
                  'image_aspect_ratio': 'square',
                  'use_im_start_end': True}


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def get_init_inputs(img_path,
                    processor,
                    tokenizer,
                    round_ids=0,
                    last_round_source=None):
    if round_ids == 0:
        det_model = build_det_model_from_cfg()
        bbox_results = inf_single_image(det_model, img_path, thr=0.3, number=100)
        image = load_image(img_path)
        image = processor.preprocess(image,
                                     do_center_crop=False,
                                     return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(224, 224),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)  # FIXME: 14 is hardcoded patch size

        pred_bboxes = bbox_results
        ori_bboxes = pred_bboxes

        w, h = pred_bboxes[:, 2] - pred_bboxes[:, 0], pred_bboxes[:, 3] - pred_bboxes[:, 1]
        filter_small = (w > 0.02) & (h > 0.02)
        pred_bboxes = pred_bboxes[filter_small]
        if len(pred_bboxes) == 0:
            pred_bboxes = ori_bboxes[:10][:, :4]
        begin_str = 'The <image> describes the entire picture, while <spi_descript> describes specific regions within the image.\n'
        print('please input the question:')
        question_str = input()
        # question_str = "debug"

        init_question = begin_str + question_str

        init_question = init_question.replace('<spi_descript>', '<bbox>' * len(pred_bboxes))
        sources = dict()
        sources['conversations'] = []
        sources['conversations'].append(
            {'from': 'human', 'value': init_question})
        sources = preprocess_multimodal([sources['conversations']],
                                        multimodal_cfg, cur_token_len)
        ori_source = copy.deepcopy(sources)

    else:
        image = last_round_source['image']
        pred_bboxes = torch.Tensor(last_round_source['bboxes'])

        print('please input the question:')
        question_str = input()
        # question_str = "debug"

        sources = last_round_source['sources'][0]
        sources.append(
            {'from': 'human', 'value': question_str})
        sources = [sources]
        ori_source = sources
        init_question = None

    # import pdb; pdb.set_trace()
    data_dict = preprocess(
        sources,
        tokenizer)

    data_dict = dict(input_ids=data_dict['input_ids'][0],
                     labels=data_dict['labels'][0],
                     sources=ori_source,
                     init_question=init_question,
                     )

    data_dict['image'] = image

    data_dict['bboxes'] = torch.Tensor(pred_bboxes)

    data_dict['img_metas'] = dict(filename=img_path)

    return data_dict


def eval_model(model_name, img_path):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    from gpt4roi.models.spi_llava import SPILlavaMPTForCausalLM

    model = SPILlavaMPTForCausalLM.from_pretrained(model_name,
                                                   low_cpu_mem_usage=True,
                                                   torch_dtype=torch.float16,
                                                   use_cache=True).cuda()

    # model = SPILlavaMPTForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True)
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                         special_tokens=True)
    spi_tokens = ['<bbox>', '<point>']
    tokenizer.add_tokens(spi_tokens, special_tokens=True)

    vision_tower = model.get_model().vision_tower[0]

    if vision_tower.device.type == 'meta':
        vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower.config._name_or_path, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).cuda()
        model.get_model().vision_tower[0] = vision_tower
    else:
        vision_tower.to(device='cuda', dtype=torch.float16)

    vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = \
        tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end

    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    # init inputs: img, inputs ids, texts
    last_source = dict()
    round_ids = 0
    while True:
        init_inputs = get_init_inputs(img_path,
                                      image_processor,
                                      tokenizer,
                                      round_ids=round_ids,
                                      last_round_source=last_source,

                                      )
        round_ids += 1
        last_source = init_inputs
        vis_dir = os.path.join(model_name, 'vis_demo')
        mmcv.mkdir_or_exist(vis_dir)
        bboxes = init_inputs['bboxes'].cuda()
        image = init_inputs['image']
        input_ids = init_inputs['input_ids'].cuda()[None]

        stop_str = '###'
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer,
                                                     input_ids)

        model.model.tokenizer = tokenizer

        with torch.inference_mode():

            model.orig_forward = model.forward
            model.forward = partial(model.orig_forward,
                                    img_metas=[None],
                                    bboxes=[bboxes.half()])

            with torch.amp.autocast(device_type='cuda'):
                output_ids = model.generate(
                    input_ids,
                    images=image.unsqueeze(0).half().cuda(),
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
            model.forward = model.orig_forward

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
                input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:],
                                         skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        file_path = init_inputs['img_metas']['filename']
        print(outputs)
        init_inputs['sources'][0].append({'from': 'gpt', 'value': outputs})
        # vis(file_path,
        #     init_question, outputs, bboxes.tolist(),
        #     id=0,
        #     dir=vis_dir
        #     )

        # print(outputs)


def vis(img_path, gt, pred, bboxes=None, region_cap=None, id=0, dir='coco'):
    img = Image.open(img_path)

    fig, ax = plt.subplots()
    width = img.width
    height = img.height
    ax.imshow(img)
    if bboxes is not None:
        for r_id, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            rect = Rectangle((x1 * width, y1 * height), w * width, h * height,
                             linewidth=5, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            if region_cap:
                text = region_cap[r_id]  # 根据需要修改标注的文字
                ax.text(x1, y1, text, fontsize=10, color='blue')

    ax.text(0, -20, f'gt:{gt}', fontsize=6, color='red')
    ax.text(0, -10, f'pred:{pred}', fontsize=6, color='blue')
    plt.savefig(f'{dir}/{img_path.split("/")[-1]}_{id}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        default='/home/shilong/Desktop/gpt4roi/heavy_roi_checkpoints/checkpoint-12000')
    # parser.add_argument("--det", type=str, default="eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
    parser.add_argument('--img', type=str, default='/data/coco/val2017/000000007574.jpg')
    args = parser.parse_args()

    eval_model(args.model_name, args.img)
