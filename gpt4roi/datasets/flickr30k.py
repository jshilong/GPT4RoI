import copy
import random

import numpy as np
import torch

from gpt4roi.train.train import preprocess, preprocess_multimodal
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCO

REGION_QUESTIONS = [
    'Which part of your overall description corresponds to the specific area of the image <spi_descript> you are referring to?',
    'In your initial description, which part corresponds to the particular area of the image <spi_descript> you are indicating?',
    'Can you specify which aspect of your overall description corresponds to the particular section of the image <spi_descript> you are pointing to?',
    'Which specific details from your overall description correspond to the particular area of the image <spi_descript> you are identifying?',
    'From your initial description, which parts specifically match the area of the image <spi_descript> you are referring to?',
    'Could you indicate which elements from your overall description relate to the particular section of the image <spi_descript> you are highlighting?',
    'Which aspects of your description correspond to the specific area of the image <spi_descript> you are referencing?',
    'Can you point out the specific parts of your description that correspond to the area of the image <spi_descript> you are focusing on?',
    'In your description, which details correspond to the specific portion of the image <spi_descript> you are indicating?',
    'Could you identify the specific parts of your description that match the section of the image <spi_descript> you are referring to?'
]
FINAL_QUESTIONS = [
    'Could you please give me a detailed description of these areas <spi_descript>?',
    'Can you provide a thorough description of the regions <spi_descript> in this image?',
    'Please describe in detail the contents of the boxed areas <spi_descript>.',
    'Could you give a comprehensive explanation of what can be found within <spi_descript> in the picture?',
    'Could you give me an elaborate explanation of the <spi_descript> regions in this picture?',
    'Can you provide a comprehensive description of the areas identified by <spi_descript> in this photo?',
    'Help me understand the specific locations labeled <spi_descript> in this picture in detail, please.',
    'What is the detailed information about the areas marked by <spi_descript> in this image?',
    'Could you provide me with a detailed analysis of the regions designated <spi_descript> in this photo?',
    'What are the specific features of the areas marked <spi_descript> in this picture that you can describe in detail?',
    'Could you elaborate on the regions identified by <spi_descript> in this image?',
    'What can you tell me about the areas labeled <spi_descript> in this picture?',
    'Can you provide a thorough analysis of the specific locations designated <spi_descript> in this photo?',
    'I am interested in learning more about the regions marked <spi_descript> in this image. Can you provide me with more information?',
    'Could you please provide a detailed description of the areas identified by <spi_descript> in this photo?',
    'What is the significance of the regions labeled <spi_descript> in this picture?',
    'I would like to know more about the specific locations designated <spi_descript> in this image. Can you provide me with more information?',
    'Can you provide a detailed breakdown of the regions marked <spi_descript> in this photo?',
    'What specific features can you tell me about the areas identified by <spi_descript> in this picture?',
    'Could you please provide a comprehensive explanation of the locations labeled <spi_descript> in this image?',
    'Can you provide a detailed account of the regions designated <spi_descript> in this photo?',
    'I am curious about the areas marked <spi_descript> in this picture. Can you provide me with a detailed analysis?',
    'What important details can you tell me about the specific locations identified by <spi_descript> in this image?',
    'Could you please provide a detailed description of the regions labeled <spi_descript> in this photo?',
    'What can you tell me about the features of the areas designated <spi_descript> in this picture?',
    'Can you provide a comprehensive overview of the regions marked <spi_descript> in this image?',
    'I would like to know more about the specific locations identified by <spi_descript> in this photo. Can you provide me with more information?',
    'What is the detailed information you have on the areas labeled <spi_descript> in this picture?',
    'Could you provide me with a thorough analysis of the regions designated <spi_descript> in this image?',
    'Can you provide a detailed explanation of the specific locations marked by <spi_descript> in this photo?'
]

class Flickr30k(CocoDataset):
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
                 max_gt_per_img=150,
                 ):

        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.vis_processor = vis_processor
        # remove this
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
                pipeline=pipeline,)
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)
        self.id_cap_dict = dict()
        self.begin_str = """The <image> provides an overview of the picture.\n"""

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
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            # convert data type for flickr
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])

            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
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
        self.id_cap_dict[img_info['file_name']] = img_info['caption']
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
            if ann['category_id']  in self.cat_ids:
                pass
            else:
                raise ValueError('category_id not in self.cat_ids')
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                # flickr label
                gt_list = [img_info['caption'][atp[0]:atp[1]] for atp in ann['tokens_positive']]
                # TODO
                gt_labels.append(gt_list[0])  # TODO: one box might correspond to multiple labels, join with `, `
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)

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
        question = random.choice(FINAL_QUESTIONS).strip()

        s_bbox_string = ''
        num_bboxes = min(len(ori_labels), self.max_gt_per_img)
        for id in range(num_bboxes):
            s_bbox_string = s_bbox_string + f'region{id+1} <bbox>,'
        question = question.replace('<spi_descript>', s_bbox_string)
        sources['conversations'].append(
            {'from': 'human', 'value': question})
        sources['conversations'].append({'from': 'gpt', 'value':
            self.id_cap_dict[data_item['img_metas'].data['filename'].split('/')[-1]]})

        shuffle_ids = torch.randperm(len(ori_labels))
     
        shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        select_bboxes = ori_bboxes[shuffle_ids]
        select_labels = [ori_labels[i] for i in shuffle_ids]


        for i in range(len(select_labels)):
            question = random.choice(REGION_QUESTIONS).strip()
            question = question.replace('<spi_descript>', f'region {i+1}')
            answer = select_labels[i]  # already string
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

        sources['conversations'][0]['value'] = self.begin_str + sources['conversations'][0]['value']

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

 
        select_bboxes = torch.cat([select_bboxes], dim=0)
       
        select_bboxes = copy.deepcopy(select_bboxes) / image.shape[1]

        data_dict['bboxes'] = select_bboxes
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
                idx = random.randint(0, len(self)-1)
                data_item = super().__getitem__(idx)
            else:
                break
        #print(data_item["img_metas"])
        # img, input_ids, labels
        data_dict = self.process_text(data_item=data_item)

        return data_dict
