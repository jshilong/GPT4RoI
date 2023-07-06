
import copy
import os
import random

import torch

from gpt4roi.train.train import preprocess, preprocess_multimodal
from mmdet.datasets import CocoDataset

# TODO: try to simulate the question of user
QUESTIONS = [
    '<spi_descript>'
    # "Could you please help me recognize the <spi_descript> in this picture?",
    # "Can you assist me in identifying the <spi_descript> in this image?",
    # "Can you tell what is at <spi_descript> in this image?",
    # "What is the object located in <spi_descript> in this picture?",
    # "Would you be able to tell me what is at <spi_descript> in this image?",
    # "Could you identify the item in <spi_descript> for me in this photograph?",
    # "Can you help me recognize the object in <spi_descript> in this picture?",
    # "I'm trying to figure out what is located in <spi_descript> in this image, could you help me?",
    # "What object can you see at <spi_descript> in this photograph?",
    # "Would you mind telling me what is located in <spi_descript> in this picture?",
    # "Can you assist me in identifying the item at <spi_descript> in this image?",
    # "What is the thing that can be seen in <spi_descript> in this photograph?",
    # "Could you please help me identify the object in <spi_descript> in this picture?"
]
#PRE_INSTRUCTION = "Please help me answer some questions about the content of a specific region in this image.\n<image>"



class CocoDet(CocoDataset):

    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,
                 vis_processor=None,
                 vis_root=None,
                 add_eos=True,
                 ignore_instruction=True,
                 filter_small=False,
                 test_mode=False,
                 max_gt_per_img=100,
                 ):
        self.multimodal_cfg = multimodal_cfg
        self.tokenizer = tokenizer
        self.vis_root = vis_root
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
            dict(type='RandomShift', shift_ratio=0.5, max_shift_px=32),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(2.0, 2.0)),
            dict(type='RandomFlip', flip_ratio=0.),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=224),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ]
        if test_mode:
            pipeline = test_pipeline
        else:
            pipeline = train_pipeline

        if test_mode:
            ann_file = f'{self.vis_root}/annotations/instances_val2017.json'
            img_prefix = self.vis_root + '/val2017'
        else:
            ann_file = f'{self.vis_root}/annotations/instances_train2017.json'
            img_prefix = self.vis_root + '/train2017'

        train = dict(
            ann_file=ann_file,
            img_prefix=img_prefix,
            test_mode=False,
            pipeline=pipeline)
        super(CocoDataset, self).__init__(**train)
        # TODO filter the small image? < 32 ?
        self.num_classes = len(self.CLASSES)
        begin_str = '<image>\nIn the conversation below, you simply answer the category name based on what you see ' \
                    'in the imagery inside a particular region.I will give you only one region each time. ' \
                    'Categories Containing '
        class_str = ', '.join(self.CLASSES)
        self.begin_str = begin_str + class_str + '.\n'

    def train_process_test(self, data_item):
        image = data_item['img'].data
        ori_labels = data_item['gt_labels'].data
        ori_bboxes = data_item['gt_bboxes'].data

        shuffle_ids = torch.randperm(len(ori_labels))
        if len(shuffle_ids) > self.max_gt_per_img:
            shuffle_ids = shuffle_ids[:self.max_gt_per_img]
        ori_bboxes = ori_bboxes[shuffle_ids]
        ori_labels = ori_labels[shuffle_ids]

        sources = dict()

        sources['conversations'] = []


        for i in range(len(ori_labels)):
            question = random.choice(QUESTIONS).strip()
            question = question.replace('<spi_descript>', '<bbox>')
            if i == 0:
                question = self.begin_str + question
            answer = self.CLASSES[ori_labels[i]]
            sources['conversations'].append(
                {'from': 'human', 'value': question})
            sources['conversations'].append({'from': 'gpt', 'value': answer})

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
        ori_bboxes = copy.deepcopy(ori_bboxes) / image.shape[1]

        data_dict['bboxes'] = ori_bboxes
        data_dict['img_metas'] = data_item['img_metas'].data
        return data_dict


    def process_text(self, data_item):
        if isinstance(data_item['img'], list):
            # test model
            data_item = {k: v[0] for k, v in data_item.items()}


        return self.train_process_test(data_item)


    def tokenize(self, text):
        res = self.tokenizer(
            text['instruction'] + text['answer'],
            return_tensors=None,
            padding='do_not_pad',
            truncation=True,
            max_length=512,
        )

        # manually add eos token
        if res['input_ids'][-1] != self.tokenizer.eos_token_id and len(
                res['input_ids']) < 512 and self.add_eos:
            res['input_ids'].append(self.tokenizer.eos_token_id)
            res['attention_mask'].append(1)
        labels = copy.deepcopy(res['input_ids'])
        # ignore instruction_token
        if self.ignore_instruction:
            bbox_index = labels.index(self.tokenizer.encode('<bbox>')[1])
            labels[:bbox_index] = [-100] * bbox_index
            # instruction_token = self.tokenizer(
            #     text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            # )
            # labels = [-100] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res

    def __getitem__(self, idx):
        data_item = super().__getitem__(idx)

        # img , input_ids, labels
        data_dict = self.process_text(data_item=data_item)

        return data_dict
