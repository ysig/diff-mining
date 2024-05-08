import logging
import math
import os
from os.path import join
from pathlib import Path
from itertools import chain

import accelerate
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami, upload_folder
from packaging import version

import diffusers
from diffusers.models import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

from args import parse_args
from PIL import Image

def concatenate_pil_images_width(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im

def concatenate_pil_images_height(images):
    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]

    return new_im


class BaseTrainer(object):
    def init_parse_args(self):
        return parse_args()

    def __init__(self):    
        self.args = self.init_parse_args()

        # check_min_version("0.15.0.dev0")
        self.check_ema()
        self.logger = get_logger(__name__, log_level="INFO")
        self.logging_dir = os.path.join(self.args.output_dir, self.args.logging_dir)
        self.init_accelerator()
        self.init_logging()
        self.seed()
        self.init_hub()
        self.ampere()

    def check_ema(self):
        if self.args.non_ema_revision is not None:
            deprecate(
                "non_ema_revision!=None",
                "0.15.0",
                message=(
                    "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                    " use `--variant=non_ema` instead."
                ),
            )    

    def init_accelerator(self):
        args = self.args
        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
        print('args.mixed_precision', args.mixed_precision)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=self.logging_dir,
            project_config=accelerator_project_config,
        )

    def seed(self):
        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

    def init_logging(self):
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        self.logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    def init_hub(self):
        # Handle the repository creation
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)

            if self.args.push_to_hub:
                self.repo_id = create_repo(repo_id=self.args.hub_model_id or Path(self.args.output_dir).name, exist_ok=True, token=self.args.hub_token).repo_id

    def customized_saving_accelerate(self):
        accelerator = self.accelerator
        # `accelerate` 0.16.0 will have better support for customized saving

        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if self.args.use_ema:
                    self.ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    print('Saving', i, type(model))
                    if isinstance(model, UNet2DConditionModel):
                        path = os.path.join(output_dir, "unet")
                    elif isinstance(model, transformers.models.clip.modeling_clip.CLIPTextModel):
                        continue
                    else:
                        raise ValueError(f"Model {type(model)} unrecognized.")

                    # make sure to pop weight so that corresponding model is not saved again
                    print('Saving model', path)
                    model.save_pretrained(path)
                    weights.pop()

            def load_model_hook(models, input_dir):
                if self.args.use_ema:
                    load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                    self.ema_unet.load_state_dict(load_model.state_dict())
                    self.ema_unet.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    if isinstance(model, UNet2DConditionModel):
                        object, folder = UNet2DConditionModel, "unet"
                    elif isinstance(model, transformers.models.clip.modeling_clip.CLIPTextModel):
                        continue
                    else:
                        raise ValueError(f"Model {type(model)} unrecognized.")

                    load_model = object.from_pretrained(input_dir, subfolder=folder)
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

            accelerator.register_save_state_pre_hook(save_model_hook)
            accelerator.register_load_state_pre_hook(load_model_hook)

    def init_xformers(self):
        if self.args.xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.logger.info(f"Using xFormers version {xformers_version}")
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

    def ampere(self):
        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    def init_lora(self):
        if self.args.lora_rank > 0:
            print('Initializing LoRA')
            from peft import LoraConfig
            unet_lora_config = LoraConfig(r=self.args.lora_rank, init_lora_weights="gaussian", target_modules=["to_k", "to_q", "to_v", "to_out.0"])
            self.unet.add_adapter(unet_lora_config)
            lora_layers = filter(lambda p: p.requires_grad, self.unet.parameters())

    def init_optimizer(self):
        args = self.args
        if args.scale_lr:
            args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * self.accelerator.num_processes)

        # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`")
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        parameters = self.unet.parameters()
        if self.args.lora_rank > 0:
            parameters = filter(lambda p: p.requires_grad, parameters)

        self.optimizer = optimizer_cls(
            parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    def init_scheduler(self):
        # Scheduler and math around the number of training steps.
        args = self.args
        self.overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)

    def end_training(self):
        # Create the pipeline using the trained modules and save it.
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            pipeline = self.to_pipeline()
            pipeline.save_pretrained(self.args.export_dir if self.args.export_dir is not None else join(self.args.output_dir, 'export'))

            if self.args.push_to_hub:
                upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )
        self.accelerator.end_training()
