import pickle
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from torch.utils.data import ConcatDataset

from gpt4roi.datasets.det_llava import DetLLava
from gpt4roi.datasets.refcoco import RefCOCO, RefCOCOG, RefCOCOP
from gpt4roi.datasets.vg import VGDATA
from gpt4roi.models.spi_llava import add_spatial_token
from gpt4roi.train.train import IGNORE_INDEX
from llava.train.train import LazySupervisedDataset
from mmcv import Config

from .coco_det import CocoDet
from .flickr30k import Flickr30k
from .vcr import MultiVCRDataset, SingleVCRDataset, VCRDataset


@dataclass
class DataCollatorForDetDataset(object):

    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances):

        input_ids, labels, img_metas, bboxes = tuple([instance.get(key,None) for instance in instances]
                                  for key in ('input_ids',
                                              'labels',
                                              'img_metas',
                                              'bboxes'))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            img_metas=img_metas,
            bboxes=bboxes
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_multitask_data_module(tokenizer,
                                data_args) :
    """Make dataset and collator for supervised fine-tuning."""

    if data_args.dataset_config is not None:
        dataset_config = Config.fromfile(data_args.dataset_config)

    multimodal_cfg = dict(
        is_multimodal=data_args.is_multimodal,
        sep_image_conv_front=data_args.sep_image_conv_front,
        image_token_len=data_args.image_token_len,
        image_aspect_ratio=data_args.image_aspect_ratio,
        use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
        image_processor=getattr(data_args, 'image_processor', None))

    train_dataset = build_spi_dataset(dataset_config.spi_datasets,
                            tokenizer=tokenizer,
                            multimodal_cfg=multimodal_cfg)

    data_collator = DataCollatorForDetDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def build_spi_dataset(dataset_config,
                  tokenizer=None,
                  multimodal_cfg=None,
                  **kwargs):
    if isinstance(dataset_config, list):
        datasets = []
        for cfg in dataset_config:
            temp_dataset = build_spi_dataset(cfg,
                                      tokenizer=tokenizer,
                  multimodal_cfg=multimodal_cfg,
                                       **kwargs)
            datasets.append(temp_dataset)
        type_string = [type(item) for item in datasets]
        print('#'*20,type_string,'#'*20)
        for dataset in datasets:
            print('#'*20,type(dataset), f'len = {len(dataset)}','#'*20)

        return ConcatDataset(datasets)
    dataset_type = dataset_config.pop('type')
    ratio = dataset_config.pop('ratio', 1)
    if dataset_type == 'coco_det':
        dataset = CocoDet(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )


    elif dataset_type == 'flickr30k':
        dataset = Flickr30k(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'VGDATA':
        dataset = VGDATA(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'det_llava':
        dataset = DetLLava(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'vcr':
        dataset = VCRDataset(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'single_vcr':
        dataset = SingleVCRDataset(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'multi_vcr':
        dataset = MultiVCRDataset(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'RefCOCO':
        dataset = RefCOCO(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'RefCOCOP':
        dataset = RefCOCOP(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )
    elif dataset_type == 'RefCOCOG':
        dataset = RefCOCOG(
            **dataset_config,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            **kwargs,
        )

    else:
        raise NotImplementedError



    if ratio < 1:
        print(f'randomly sample {ratio} of the dataset {dataset_type}: {int(ratio * len(dataset))}' )
        random_indices = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        subsample_dataset.collater = dataset.collater
        return subsample_dataset
    else:
        return dataset



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)

if __name__ == '__main__':
    spi_datasets, tokenizer, multimodal_cfg = pickle.load(open('./test_dataset.pkl', 'rb'))
    spi_datasets[0]['vis_root'] = '/data/coco'

    tokenizer, _ = add_spatial_token(tokenizer)

    coco_det = build_spi_dataset(spi_datasets,
                      tokenizer=tokenizer,
                      multimodal_cfg=multimodal_cfg)
    coco_det[0]
