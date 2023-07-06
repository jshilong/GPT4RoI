import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.integrations import is_fairscale_available
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import ShardedDDPOption
from transformers.utils import is_sagemaker_mp_enabled


def unwrap_model(model: nn.Module) -> nn.Module:
    """Recursively unwraps a model from potential containers (as used in
    distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    else:
        return model


class LLaVATrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder,
                                                   'mm_projector')
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder,
                                                        f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save,
                           os.path.join(output_dir, f'mm_projector.bin'))

        super(LLaVATrainer, self)._save(output_dir, state_dict)

    def create_optimizer(self):

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if 'bias' not in name]

            train_str = 'spi_module'

            if os.environ.get('ONLY_SPI', None) and (not os.environ.get('PROJ', None)):
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters() if (train_str in n and p.requires_grad)
                        ],
                        # TODO ab this
                        'weight_decay': 0.01,
                    },
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters() if (train_str not in n and p.requires_grad)
                        ],
                        'weight_decay': 0.0,
                        'lr': 0.

                    },
                ]

            elif os.environ.get('ONLY_SPI', None) and os.environ.get('PROJ', None):
                proj_train_str = 'proj'
                spi_train_str = 'spi_module'
                print('Only training SPI and PROJ')
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters() if
                            ((spi_train_str in n or proj_train_str in n) and p.requires_grad)
                        ],
                        # TODO ab this
                        'weight_decay': 0.0,
                    },
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters() if
                            ((proj_train_str not in n and spi_train_str not in n)
                             and p.requires_grad)
                        ],
                        'weight_decay': 0.0,
                        'lr': 0.

                    },
                ]

            else:
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        'weight_decay': self.args.weight_decay,
                    },
                    {
                        'params': [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        'weight_decay': 0.0,

                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                if is_fairscale_available():
                    from fairscale.optim import OSS
                else:
                    raise ImportError()
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == 'Adam8bit':
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f'skipped {module}: {skipped / 2 ** 20}M params')
                            manager.register_module_override(module, 'weight', {'optim_bits': 32})
                            logger.debug(f'bitsandbytes: will optimize {module} in fp32')
                    print(f'skipped: {skipped / 2 ** 20}M params')

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
