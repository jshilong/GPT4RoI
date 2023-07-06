import copy
import random

import numpy as np
import torch

import mmcv
from gpt4roi.train.train import preprocess, preprocess_multimodal
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO

QUESTIONS = [
    '<spi_descript>',
]

REFG_QUESTIONS = [
    'Can you provide me with a detailed description of the region in the picture marked by <spi_descript>?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <spi_descript> in the image?',
    "I'd like to know more about the area in the photo labeled <spi_descript>. Can you give me a detailed description?",
    'Could you describe the region shown as <spi_descript> in the picture in great detail?',
    'What details can you give me about the region outlined by <spi_descript> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <spi_descript> in the image.',
    'Can you give me a detailed account of the region labeled as <spi_descript> in the picture?',
    "I'm interested in learning more about the region represented by <spi_descript> in the photo. Can you describe it in detail?",
    'What is the region outlined by <spi_descript> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <spi_descript>, please?',
    "I'm curious about the region represented by <spi_descript> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <spi_descript> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <spi_descript>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <spi_descript> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <spi_descript> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <spi_descript> in the image, please.',
    'Can you give me a detailed account of the region labeled as <spi_descript> in the picture, please?',
    "I'm interested in learning more about the region represented by <spi_descript> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <spi_descript> in the picture like, please? Could you give me a detailed description?',
]

import os


class RefCOCO(CocoDataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,
                 vis_processor=None,
                 ann_file=None,
                 img_prefix=None,
                 add_eos=True,
                 ignore_instruction=True,
                 filter_small=False,
                 test_mode=False,
                 max_gt_per_img=15,
                 ):

        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.vis_processor = vis_processor
        self.max_gt_per_img = max_gt_per_img
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
        self.filter_small = filter_small
        self.test_mode = test_mode

        img_norm_cfg = dict(
            mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
            std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
            to_rgb=True)

        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            # dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='FilterAnnotationsFlickr', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundleFlickr'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        else:
            ann_file = self.ann_file
            img_prefix = self.img_prefix
        train = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline, )
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)
        self.id_cap_dict = dict()
        self.begin_str = '<image>\n I will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image, as well as its position within ' \
                         'the image and its basic attributes.'

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # TODO: obtain images that contain annotation
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        num_remove_images = 0
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            if len(info['caption'].split(' ')) < 3:
                num_remove_images += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        print(f'Filtered {num_remove_images} from  {self.ann_file} ')
        return data_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        img_path = os.path.join(self.img_prefix, img_info['file_name'].split('_')[-1])
        self.id_cap_dict[img_info['file_name'].split('_')[-1]] = img_info['caption']
        # flickr
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue

            bbox = [x1, y1, x1 + w, y1 + h]

            gt_bboxes.append(bbox)
            gt_labels.append(img_info['caption'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

        #mmcv.imshow_bboxes(img_path, gt_bboxes, win_name=img_info['caption'])

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            caption=img_info['caption'],
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}

        return self.train_process_test(data_item)

    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels']
        ori_bboxes = data_item['gt_bboxes'].data

        sources = {'conversations': []}

        # DETAILS QUESTION


        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]

        for i in range(len(select_labels)):
            question = random.choice(QUESTIONS).strip()
            question = question.replace('<spi_descript>', '<bbox>')
            answer = select_labels[i]  # already string
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        sources['conversations'][0]['value'] = self.begin_str + \
                                               sources['conversations'][0][
                                                   'value']

        # print(sources["conversations"])
        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)
        assert image.shape[1] == image.shape[2]

        # a hard code [] for sources
        sources = preprocess_multimodal(
            copy.deepcopy([sources['conversations']]),
            self.multimodal_cfg,
            cur_token_len)

        data_dict = preprocess(
            sources,
            self.tokenizer)
        # get single

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image

        # double for last detail question
        ori_bboxes = select_bboxes


        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]

        data_dict['bboxes'] = ori_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data
        return data_dict

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)
        max_loops = 10
        i = 0

        while True:
            if i > max_loops:
                raise ValueError('No gt_labels')
            i += 1
            if len(data_item['gt_labels']) == 0:
                idx = random.randint(0, len(self) - 1)
                data_item = super().__getitem__(idx)
            else:
                break
        # print(data_item["img_metas"])
        # img, input_ids, labels
        data_dict = self.process_text(data_item=data_item)

        return data_dict


class RefCOCOP(RefCOCO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.begin_str = '<image>\n I will provide you with only one region ' \
                         'containing only one object, although there may be other ' \
                         'objects present in the image. It is recommended that you ' \
                         "describe the object's relative position with respect to other " \
                         'objects in the image and its basic attibuts, you should not ' \
                         'give its position within the image' \


class RefCOCOG(RefCOCO):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # self.begin_str = "<image>\n I will provide you with only one region " \
        #                  "containing only one object, although there may be other " \
        #                  "objects present in the image. It is recommended that you " \
        #                  "try to detail descript the region."
        self.begin_str = """The <image> provides an overview of the picture.\n"""
    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels']
        ori_bboxes = data_item['gt_bboxes'].data

        sources = {'conversations': []}

        # DETAILS QUESTION


        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]

        for i in range(len(select_labels)):
            question = random.choice(REFG_QUESTIONS).strip()
            question = question.replace('<spi_descript>', f'region{i+1} <bbox>')
            answer = select_labels[i]  # already string
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        sources['conversations'][0]['value'] = self.begin_str + \
                                               sources['conversations'][0][
                                                   'value']

        # print(sources["conversations"])
        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)
        assert image.shape[1] == image.shape[2]

        # a hard code [] for sources
        sources = preprocess_multimodal(
            copy.deepcopy([sources['conversations']]),
            self.multimodal_cfg,
            cur_token_len)

        data_dict = preprocess(
            sources,
            self.tokenizer)
        # get single

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image

        # double for last detail question
        ori_bboxes = select_bboxes


        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]

        data_dict['bboxes'] = ori_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data
        # print(sources)
        # print(len(ori_bboxes))
        return data_dict
