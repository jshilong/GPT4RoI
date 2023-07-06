import os
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from gpt4roi.models.layers import MLVLROIQueryModule
from llava.model.llava import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_PATCH_TOKEN,
                               LlavaLlamaForCausalLM, LlavaLlamaModel)


class SPILlavaLlamaModel(LlavaLlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_level_spi_features = 4

        self.spi_module = MLVLROIQueryModule(embed_dims=1024, out_dims=4096,
                                             num_levels=4)

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            img_metas=None,
            bboxes=None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if os.environ.get('DEBUG', None):
            self.cache['img_metas'] = img_metas
            self.cache['bboxes'] = bboxes

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[
                                             1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0),
                                                         output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config,
                                                            'mm_vision_select_layer',
                                                            -1)
                        select_hidden_state = image_forward_out.hidden_states[
                            select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images,
                                                      output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config,
                                                        'mm_vision_select_layer',
                                                        -1)
                    select_hidden_state = image_forward_outs.hidden_states[
                        select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]

                    # TODO: check 3 whether it is resonable
                    mlvl_spi_features = image_forward_outs.hidden_states[
                                        select_hidden_state_layer::-3]
                    mlvl_spi_features = mlvl_spi_features[::-1]
                    mlvl_spi_features = mlvl_spi_features[
                                        -self.num_level_spi_features:]
                    mlvl_spi_features = [item[:, 1:] for item in
                                         mlvl_spi_features]
            if bboxes is not None and (len(bboxes) > 0):
                mlvl_spi_features = self.spi_module(mlvl_spi_features,
                                                    bboxes)
            else:
                mlvl_spi_features = [None for _ in range(len(input_ids))]

            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for
                                  image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(256, 1024,
                                               device=inputs_embeds.device,
                                               dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds, spi_feat in zip(input_ids,
                                                                 inputs_embeds,
                                                                 mlvl_spi_features):
                if (
                        cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (
                            0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue
                if vision_tower.config.use_im_start_end:

                    if (
                            cur_input_ids == vision_tower.config.im_start_token).sum() != (
                            cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError(
                            'The number of image start tokens and image end tokens should be the same.')
                    image_start_tokens = torch.where(
                        cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[
                            image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError(
                                'The image end token should follow the image start token.')
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[
                                                              :image_start_token_pos].detach(),
                                                              cur_input_embeds[
                                                              image_start_token_pos:image_start_token_pos + 1],
                                                              cur_image_features,
                                                              cur_input_embeds[
                                                              image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],
                                                              cur_input_embeds[
                                                              image_start_token_pos + num_patches + 2:].detach()),
                                                             dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[
                                                              :image_start_token_pos + 1],
                                                              cur_image_features,
                                                              cur_input_embeds[
                                                              image_start_token_pos + num_patches + 1:]),
                                                             dim=0)
                        # fill spi features. avoid the inplace fill
                        if spi_feat is not None:
                            spi_embeds = torch.zeros_like(cur_new_input_embeds)
                            spi_mask = (cur_input_ids ==
                                        self.tokenizer.convert_tokens_to_ids(
                                            ['<bbox>'])[0])

                            spi_embeds[spi_mask] = spi_feat.to(spi_embeds.dtype)
                            cur_new_input_embeds = cur_new_input_embeds * (
                                                                              ~spi_mask).to(
                                cur_input_embeds.dtype)[:, None] + spi_embeds
                        else:
                            assert (cur_input_ids ==
                                        self.tokenizer.convert_tokens_to_ids(
                                            ['<bbox>'])[0]).sum() == 0
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (
                            cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError(
                            'The number of image patch tokens should be the same as the number of image patches.')
                    masked_indices = torch.where(
                        cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start,
                                                       mask_index_start + num_patches,
                                                       device=masked_indices.device,
                                                       dtype=masked_indices.dtype)).any():
                        raise ValueError(
                            'The image patch tokens should be consecutive.')
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[
                                                          :mask_index_start].detach(),
                                                          cur_image_features,
                                                          cur_input_embeds[
                                                          mask_index_start + num_patches:].detach()),
                                                         dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[
                                                          :mask_index_start],
                                                          cur_image_features,
                                                          cur_input_embeds[
                                                          mask_index_start + num_patches:]),
                                                         dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


def add_spatial_token(tokenizer):
    spi_tokens = ['<bbox>', '<point>']
    num_spi_tokens = tokenizer.add_tokens(spi_tokens, special_tokens=True)

    return tokenizer, num_spi_tokens


class SPILlavaMPTForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config):
        super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = SPILlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size,
                                 bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            *args,
            img_metas=None,
            bboxes=None,
            **kwargs
    ):
        self.model.orig_forward = self.model.forward
        self.model.forward = partial(self.model.orig_forward,
                                     img_metas=img_metas,
                                     bboxes=bboxes)

        outputs = super().forward(*args, **kwargs)
        self.model.forward = self.model.orig_forward
        return outputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer,
                                    device,
                                    tune_mm_mlp_adapter=False,
                                    pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        tokenizer, num_spi_tokens = add_spatial_token(tokenizer)
        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
                special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            num_new_tokens = num_new_tokens + num_spi_tokens
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[
                                        :-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(
                        device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter,
                                                  map_location='cpu')
                embed_tokens_weight = mm_projector_weights[
                    'model.embed_tokens.weight']
                num_new_tokens = num_new_tokens - num_spi_tokens
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                                                         -num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f'Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.')

        vision_config.im_patch_token = \
            tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.bbox_token = tokenizer.convert_tokens_to_ids(['<bbox>'])[
            0]
        vision_config.point_token = \
            tokenizer.convert_tokens_to_ids(['<point>'])[0]
        # broadcast the tokenizer to all modules
        for m in self.modules():
            m.tokenizer = tokenizer
