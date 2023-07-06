import copy
import json
import os
import random
import re
import secrets
import string
from tkinter import N

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gpt4roi.train.train import preprocess, preprocess_multimodal

WHY_QUESTIONS = [
    'why?',
    'why',
    "What's the rationale for your decision?",
    'What led you to that conclusion?',
    "What's the reasoning behind your opinion?",
    'Why do you believe that to be true?',
    'Can you explain the basis for your thinking?',
    'What factors influenced your perspective?',
    'How did you arrive at that perspective?',
    'What evidence supports your viewpoint?',
    'What makes you think that way?',
    "What's the logic behind your argument?",
    'Can you provide some context for your opinion?',
    "What's the basis for your assertion?",
    'Why do you hold that belief?',
    'What experiences have shaped your perspective?',
    'What assumptions underlie your reasoning?',
    "What's the foundation of your assertion?",
    "What's the source of your reasoning?",
    "What's the motivation behind your decision?",
    "What's the impetus for your belief?",
    "What's the driving force behind your conclusion?",
    'Why do you think that?',
    "What's your reasoning?",
    'What makes you say that?',
    'Why do you feel that way?',
    "What's the story behind that?",
    "What's your thought process?",
    "What's the deal with that?",
    "What's the logic behind it?",
    'Why do you believe that?',
    "What's the real deal here?",
    "What's the reason behind it?",
    "What's the thought process behind your decision?",
    "What's the rationale for your opinion?",
    'Why do you have that impression?',
    "What's the background to that?",
    "What's the evidence that supports your view?",
    "What's the explanation for that?"
]

Ref_WAY = [
    'There are <spi> in the image,',
    'There are <spi>,',
    'Given <spi>,',
    'Given <spi> in the image,',
    '<spi>,',
    '<spi> in the given image,'


]

class VCRDataset(Dataset):
    CLASSES = ('object',)

    def __init__(self,
                 tokenizer,
                 multimodal_cfg=None,

                 ann_file=None,
                 img_prefix=None,

                 ):
        super(VCRDataset, self).__init__()


        self.img_prefix = img_prefix

        self.tokenizer = tokenizer

        self.multimodal_cfg = multimodal_cfg



        self.begin_str = """The <image> provides an overview of the picture.\n"""
        self.data_infos = self.load_annotations(ann_file)
        print('normal_vcr', len(self.data_infos))

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
          ann_list = [json.loads(line) for line in f]
        data_infos = []

        import re

        def replace_numbers_with_tags(s, class_names):
            pattern = r'\b(\d+)\b'
            try:
                result = re.sub(pattern, lambda match: f'{class_names[int(match.group(1))]} at region{match.group(1)}', s)
            except:
                # contain number not for instance
                return None
            return result


        for ann in ann_list:

            metadata_fn_path = ann['metadata_fn']
            img_fn = ann['img_fn']
            img_path = os.path.join(self.img_prefix,img_fn)
            bboxes = np.array(json.load(open(os.path.join(self.img_prefix, metadata_fn_path)))['boxes'])
            class_names = ann['objects']
            num_objects = len(class_names)
            ref_string = ''
            for i in range(num_objects):
                ref_string = ref_string +  f'region{i+1} <bbox>' + ','
            ref_string = ref_string[:-1]
            ref_prefix = random.choice(Ref_WAY)

            begion_string = ref_prefix.replace('<spi>', ref_string)
            qa_s = []

            q = ann['question_orig']
            q = replace_numbers_with_tags(q, class_names)
            a = ann['answer_orig']
            a = replace_numbers_with_tags(a, class_names)
            why = ann['rationale_orig']
            why = replace_numbers_with_tags(why, class_names)
            if (q is None) or (a is None) or (why) is None:
                continue


            qa_s.append({'from': 'human', 'value': begion_string + q})
            qa_s.append({'from': 'gpt', 'value': a})
            qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})
            qa_s.append({'from': 'gpt', 'value': why})

            # print(qa_s)
            # print(len(bboxes))


            data_infos.append(dict(
                img_path = img_path,
                bboxes=bboxes,
                labels= class_names,
                qas = qa_s)
            )


        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, i):
        # i = 1
        # print(f"{i}th item")
        data_info = self.data_infos[i]


        img_path = data_info['img_path']
        bboxes = data_info['bboxes']

        qas = data_info['qas']
        processor = self.multimodal_cfg['image_processor']
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        # TODO ablation this
        image_file = img_path
        pred_bboxes = bboxes
        pred_bboxes = pred_bboxes[:,:4] / np.array([w,h,w,h])[None]

        image = processor.preprocess(image,
                                     do_center_crop=False,
                                     return_tensors='pt')['pixel_values'][0]

        image = torch.nn.functional.interpolate(image.unsqueeze(0),
                                                size=(224, 224),
                                                mode='bilinear',
                                                align_corners=False).squeeze(0)

        cur_token_len = (image.shape[1] // 14) * (image.shape[2] // 14)  # FIXME: 14 is hardcoded patch size
        qas = copy.deepcopy(qas)
        qas[0]['value'] = self.begin_str + qas[0]['value']

        # print(copy_source)
        sources = preprocess_multimodal(
            copy.deepcopy([qas]),
            self.multimodal_cfg, cur_token_len)

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict['input_ids'][0],
                             labels=data_dict['labels'][0])

        data_dict['image'] = image

        data_dict['bboxes'] = torch.Tensor(pred_bboxes)

        data_dict['img_metas'] = dict(filename=image_file)


        return data_dict

class SingleVCRDataset(VCRDataset):
    # format1
    # q & a only single index

    def judge_format(self, ann):
        q = ann['question_orig']
        a = ann['answer_orig']
        why = ann['rationale_orig']

        why_digits = re.findall(r'\d+', why)
        a_digits = re.findall(r'\d+', a)
        q_digits = re.findall(r'\d+', q)
        format_id = -1
        if set(a_digits).issubset(q_digits):
            # only qa
            format_id = 0
        new_digits = set(a_digits).union(set(why_digits))
        if new_digits.issubset(set(q_digits)):
            format_id = 1
        if len(q_digits) <= 1:
            single_region = True
        else:
            single_region = False
        return format_id, single_region, q_digits, a_digits, why_digits


    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
            ann_list = [json.loads(line) for line in f]
        data_infos = []
        num_filter = 0
        for ann in ann_list:

            metadata_fn_path = ann['metadata_fn']
            img_fn = ann['img_fn']
            img_path = os.path.join(self.img_prefix,img_fn)
            bboxes = np.array(json.load(open(os.path.join(self.img_prefix, metadata_fn_path)))['boxes'])
            class_names = ann['objects']
            num_objects = len(class_names)

            format_id, single_region,  q_digits, a_digits, why_digits = self.judge_format(ann)
            if format_id < 0:
                continue
            if len(a_digits) == 0:
                continue

            qa_s = []

            q = ann['question_orig']
            a = ann['answer_orig']

            why = ann['rationale_orig']
            # replace instance id to <bbox>


            q_instance_index = np.array(q_digits,dtype=np.int64) - 1
            if (q_instance_index < 0).any() or (q_instance_index>(len(bboxes)-1)).any():
                # string has other number
                continue
            else:
                bboxes = bboxes[q_instance_index]


            answer_instance_index = np.array(a_digits,dtype=np.int64) - 1
            why_instance_index = np.array(why_digits, dtype=np.int64) - 1

            if single_region:
                # only replace the answer
                q = re.sub(r'\d+', 'region1 <bbox>', q)
                # TODO fix this
                assert len(bboxes) == 1
                if q.count('<bbox>') != len(bboxes):
                    num_filter += 1
                    continue
                if len(q_instance_index) > 0:
                    q_instance_index = q_instance_index[0]

                if len(a_digits) > 0:
                    a = copy.deepcopy(a).replace(str(a_digits[0]), f'{class_names[q_instance_index]} at region1')
                qa_s.append({'from': 'human', 'value': q})
                qa_s.append({'from': 'gpt', 'value': a})
                # with why
                if format_id == 1:
                    qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})

                    if len(why_digits) > 0:
                        why_instance_index = why_instance_index[0]
                        why = copy.deepcopy(why).replace(str(why_digits[0]), f'{class_names[why_instance_index]} at region1')
                    qa_s.append({'from': 'gpt', 'value': why})

                # print(qa_s)
                # print(len(bboxes))

                data_infos.append(dict(
                        img_path=img_path,
                        bboxes=bboxes,
                        qas=qa_s)
                    )


        # print("single_region",len(data_infos))
        # print("single_region filter", num_filter)
        return data_infos


class MultiVCRDataset(SingleVCRDataset):
    # format1
    # q & a only single index
    def __init__(self,*args,**kwargs):
        def map_number_to_unique_string(number):
            alphabet = string.ascii_letters
            return ''.join(secrets.choice(alphabet) for _ in range(6))
        self.unique_string = list(set([map_number_to_unique_string(i) for i in range(100)]))
        super(MultiVCRDataset, self).__init__(*args,**kwargs)

    def load_annotations(self, ann_file):

        with open(ann_file, 'r') as f:
            ann_list = [json.loads(line) for line in f]
        data_infos = []
        num_filter = 0
        for index , ann in enumerate(ann_list):
            # print(f"index {index}")
            # if index == 15363:
            #     print()
            metadata_fn_path = ann['metadata_fn']
            img_fn = ann['img_fn']
            img_path = os.path.join(self.img_prefix,img_fn)
            bboxes = np.array(json.load(open(os.path.join(self.img_prefix, metadata_fn_path)))['boxes'])
            class_names = ann['objects']
            num_objects = len(class_names)

            format_id, single_region,  q_digits, a_digits, why_digits = self.judge_format(ann)
            if format_id < 0:
                continue
            if len(a_digits) == 0:
                continue

            qa_s = []

            q = ann['question_orig']
            a = ann['answer_orig']

            why = ann['rationale_orig']
            # replace instance id to <bbox>


            q_instance_index = np.array(q_digits, dtype=np.int64) - 1
            if (q_instance_index < 0).any() or (q_instance_index>(len(bboxes)-1)).any():
                # string has other number
                continue
            else:
                bboxes = bboxes[q_instance_index]



            why_instance_index = np.array(why_digits, dtype=np.int64) - 1



            if not single_region:
                # only replace the answer
                for id, instance_index in enumerate(q_digits):
                    # avoid process twice
                    instance_index = int(instance_index)
                    ori_ = r'(\b' + str(instance_index) + r'\b)'
                    q = re.sub(ori_,  self.unique_string[id], q)
                    a = re.sub(ori_,  self.unique_string[id], a)
                    #q = q.replace(str(instance_index), self.unique_string[id])
                    #a = a.replace(str(instance_index), self.unique_string[id])
                    if format_id == 1:
                        #why = why.replace(str(instance_index), self.unique_string[id])
                        why = re.sub(ori_,  self.unique_string[id], why)

                for id, instance_index in enumerate(q_digits):
                    instance_index = int(instance_index)
                    q = q.replace(self.unique_string[id], f'region{id+1} <bbox>')
                    a = a.replace(self.unique_string[id], f'{class_names[instance_index-1]} at region{id+1}')
                    if format_id == 1:
                        why = why.replace(self.unique_string[id], f'{class_names[instance_index - 1]} at region{id+1}')
                if q.count('<bbox>') != len(bboxes):
                    num_filter += 1
                    continue
                qa_s.append({'from': 'human', 'value': q})
                qa_s.append({'from': 'gpt', 'value': a})
                # with why
                if format_id == 1:
                    qa_s.append({'from': 'human', 'value': random.choice(WHY_QUESTIONS)})
                    qa_s.append({'from': 'gpt', 'value': why})
                # import mmcv
                # mmcv.imshow_det_bboxes(img_path,
                #                        bboxes=bboxes,
                #                        labels=np.array(list(range(len(bboxes)))),
                #                        show=False,
                #                        out_file="./debug_vcr/1.png"
                #                        )
                #
                # print(qa_s)
                # print(len(bboxes))


                data_infos.append(dict(
                    img_path = img_path,
                    bboxes=bboxes,
                    qas = qa_s)
                )

        # print("multi_region",len(data_infos))
        # print("multi_region filter", num_filter)
        return data_infos
