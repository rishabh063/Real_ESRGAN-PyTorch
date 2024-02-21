# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import time
from copy import deepcopy
from pathlib import Path

import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from real_esrgan.data.degenerated_image_dataset import DegeneratedImageDataset
from real_esrgan.data.degradations import degradation_process
from real_esrgan.data.paired_image_dataset import PairedImageDataset
from real_esrgan.data.prefetcher import CUDAPrefetcher, CPUPrefetcher
from real_esrgan.data.transforms import random_crop_torch, random_rotate_torch, random_vertically_flip_torch, random_horizontally_flip_torch
from real_esrgan.layers.ema import ModelEMA
from real_esrgan.models import RRDBNet
from real_esrgan.utils.checkpoint import load_state_dict, save_checkpoint
from real_esrgan.utils.diffjepg import DiffJPEG
from real_esrgan.utils.envs import select_device, set_seed_everything
from real_esrgan.utils.events import LOGGER, AverageMeter, ProgressMeter
from real_esrgan.utils.general import increment_name, find_last_checkpoint
from real_esrgan.utils.imgproc import USMSharp
from real_esrgan.utils.torch_utils import get_model_info
from .evaler import Evaler


def init_train_env(config_dict: DictConfig) -> [DictConfig, torch.device]:
    r"""Initialize the training environment.

    Args:
        config_dict (DictConfig): The configuration dictionary.

    Returns:
        device (torch.device): The device to be used for training.
    """

    def _resume(config_dict: DictConfig, weights_path: str | Path):
        checkpoint_path = weights_path if isinstance(weights_path, str) else find_last_checkpoint()
        assert Path(checkpoint_path).is_file(), f"the checkpoint path is not exist: {checkpoint_path}"
        LOGGER.info(f"Resume training from the checkpoint file: `{checkpoint_path}`")
        resume_config_file_path = Path(checkpoint_path).parent.parent / save_config_name
        if resume_config_file_path.exists():
            config_dict = OmegaConf.load(resume_config_file_path)
        else:
            LOGGER.warning(f"Can not find the path of `{Path(checkpoint_path).parent.parent / save_config_name}`, will save exp log to"
                           f" {Path(checkpoint_path).parent.parent}")
            LOGGER.warning(f"In this case, make sure to provide configuration, such as datasets, batch size.")
            config_dict.TRAIN.SAVE_DIR = str(Path(checkpoint_path).parent.parent)
        return checkpoint_path

    # Define the name of the configuration file
    save_config_name = "config.yaml"

    resume_g = config_dict.get("RESUME_G", "")
    resume_d = config_dict.get("RESUME_D", "")

    # Handle the resume training case
    if resume_g:
        checkpoint_path = _resume(config_dict, resume_g)
        config_dict.TRAIN.RESUME = checkpoint_path  # set the args.resume to checkpoint path.
    elif resume_d:
        checkpoint_path = _resume(config_dict, resume_d)
        config_dict.TRAIN.RESUME = checkpoint_path
    else:
        save_dir = config_dict.TRAIN.OUTPUT_DIR / Path(config_dict.EXP_NAME)
        config_dict.TRAIN.SAVE_DIR = str(increment_name(save_dir))
        Path(config_dict.TRAIN.SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # Select the device for training
    device = select_device(config_dict.DEVICE)

    # Set the random seed
    set_seed_everything(1 + config_dict.TRAIN.RANK, deterministic=(config_dict.TRAIN.RANK == -1))

    # Save the configuration
    OmegaConf.save(config_dict, config_dict.TRAIN.SAVE_DIR / Path(save_config_name))

    return config_dict, device


class Trainer:
    def __init__(self, config_dict: DictConfig, device: torch.device):
        self.config_dict = config_dict
        self.device = device
        self.exp_name = self.config_dict.EXP_NAME
        self.phase = self.config_dict.PHASE
        self.upscale_factor = self.config_dict.UPSCALE_FACTOR
        self.degradation_model_parameters_dict = self.config_dict.get("DEGRADATION_MODEL_PARAMETERS_DICT")
        self.degradation_process_parameters_dict = self.config_dict.get("DEGRADATION_PROCESS_PARAMETERS_DICT")
        self.dataset_config_dict = self.config_dict.DATASET
        self.model_config_dict = self.config_dict.MODEL
        self.train_config_dict = self.config_dict.TRAIN
        self.eval_config_dict = self.config_dict.EVAL

        # ========== Init all config ==========
        # datasets
        self.dataset_mode = self.dataset_config_dict.MODE
        self.dataset_train_gt_images_dir = self.dataset_config_dict.TRAIN_GT_IMAGES_DIR
        self.dataset_train_lr_images_dir = self.dataset_config_dict.get("TRAIN_LR_IMAGES_DIR")
        self.dataset_val_gt_images_dir = self.dataset_config_dict.VAL_GT_IMAGES_DIR
        self.dataset_val_lr_images_dir = self.dataset_config_dict.VAL_LR_IMAGES_DIR

        # model
        self.model_g_type = self.model_config_dict.G.TYPE
        self.model_g_in_channels = self.model_config_dict.G.IN_CHANNELS
        self.model_g_out_channels = self.model_config_dict.G.OUT_CHANNELS
        self.model_g_channels = self.model_config_dict.G.CHANNELS
        self.model_g_growth_channels = self.model_config_dict.G.GROWTH_CHANNELS
        self.model_g_num_rrdb = self.model_config_dict.G.NUM_RRDB

        # train
        self.resume_g = self.train_config_dict.get("RESUME_G", "")
        self.resume_d = self.train_config_dict.get("RESUME_D", "")
        # train weights
        self.g_weights_path = self.train_config_dict.get("G_WEIGHTS_PATH", "")
        self.d_weights_path = self.train_config_dict.get("D_WEIGHTS_PATH", "")
        # train dataset
        self.train_image_size = self.train_config_dict.IMAGE_SIZE
        self.train_batch_size = self.train_config_dict.BATCH_SIZE
        self.train_num_workers = self.train_config_dict.NUM_WORKERS
        # train solver
        self.solver_g_optim = self.train_config_dict.SOLVER.G.OPTIM
        self.solver_g_lr = self.train_config_dict.SOLVER.G.LR
        self.solver_g_betas = list(self.train_config_dict.SOLVER.G.BETAS)
        self.solver_g_eps = self.train_config_dict.SOLVER.G.EPS
        self.solver_g_weight_decay = self.train_config_dict.SOLVER.G.WEIGHT_DECAY
        self.solver_g_lr_scheduler_type = self.train_config_dict.SOLVER.G.LR_SCHEDULER.TYPE
        self.solver_g_lr_scheduler_step_size = self.train_config_dict.SOLVER.G.LR_SCHEDULER.STEP_SIZE
        self.solver_g_lr_scheduler_gamma = self.train_config_dict.SOLVER.G.LR_SCHEDULER.GAMMA
        # train loss
        self.loss_pixel = self.train_config_dict.LOSS.get("PIXEL", "")
        self.loss_feature = self.train_config_dict.LOSS.get("FEATURE", "")
        self.loss_gan = self.train_config_dict.LOSS.get("GAN", "")
        if self.loss_pixel:
            self.loss_pixel_type = self.loss_pixel.get("TYPE", "")
            self.loss_pixel_weight = list(self.loss_pixel.get("WEIGHT", []))
        if self.loss_feature:
            self.loss_feature_type = self.loss_feature.get("TYPE", "")
            self.loss_feature_weight = list(self.loss_feature.get("WEIGHT", []))
        if self.loss_gan:
            self.loss_gan_type = self.loss_gan.get("TYPE", "")
            self.loss_gan_weight = list(self.loss_gan.get("WEIGHT", []))
        # train hyper-parameters
        self.epochs = self.train_config_dict.EPOCHS
        # train setup
        self.local_rank = self.train_config_dict.LOCAL_RANK
        self.rank = self.train_config_dict.RANK
        self.world_size = self.train_config_dict.WORLD_SIZE
        self.dist_url = self.train_config_dict.DIST_URL
        self.save_dir = self.train_config_dict.SAVE_DIR
        # train results
        self.output_dir = self.train_config_dict.OUTPUT_DIR
        self.verbose = self.train_config_dict.VERBOSE

        # ========== Init all objects ==========
        if self.phase not in ["psnr", "gan"]:
            raise NotImplementedError(f"Phase {self.phase} is not implemented. Only support `psnr` and `gan`.")

        # datasets
        self.train_dataloader, self.val_dataloader = self.get_dataloader()
        self.num_train_batch = len(self.train_dataloader)
        if self.dataset_mode == "degradation":
            # Define JPEG compression method and USM sharpening method
            jpeg_operation = DiffJPEG()
            usm_sharpener = USMSharp()
            self.jpeg_operation = jpeg_operation.to(device=self.device)
            self.usm_sharpener = usm_sharpener.to(device=self.device)

        # model
        self.g_model = self.get_g_model()
        self.ema = ModelEMA(self.g_model)

        # optimizer and scheduler
        self.g_optimizer = self.get_g_optimizer()
        self.g_lr_scheduler = self.get_g_lr_scheduler()

        self.start_epoch = 0
        # resume model for training
        if self.resume_g:
            self.g_checkpoint = torch.load(self.resume_g, map_location=self.device)
            if self.g_checkpoint:
                self.resume_g_model()
            else:
                LOGGER.warning(f"Loading state_dict from {self.resume_g} failed, train from scratch...")

        # losses
        if self.phase == "psnr":
            self.pixel_criterion = self.define_psnr_loss()

        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir)

        # Initialize the mixed precision method
        self.scaler = amp.GradScaler()

        # eval for training
        self.evaler = Evaler(config_dict, device)
        # metrics
        self.best_psnr: float = 0.0
        self.best_ssim: float = 0.0
        self.best_niqe: float = 100.0

    def get_dataloader(self):
        if self.dataset_mode not in ["degradation", "paired"]:
            raise NotImplementedError(f"Dataset mode {self.dataset_mode} is not implemented. Only support `degradation` and `paired`.")

        if self.dataset_mode == "degradation":
            train_datasets = DegeneratedImageDataset(self.dataset_train_gt_images_dir,
                                                     self.degradation_model_parameters_dict)
        else:
            train_datasets = PairedImageDataset(self.dataset_train_gt_images_dir,
                                                self.dataset_train_lr_images_dir)
        val_datasets = PairedImageDataset(self.dataset_val_gt_images_dir,
                                          self.dataset_val_lr_images_dir)
        # generate dataset iterator
        train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                       batch_size=self.train_batch_size,
                                                       shuffle=True,
                                                       num_workers=self.train_num_workers,
                                                       pin_memory=True,
                                                       drop_last=True,
                                                       persistent_workers=True)
        val_dataloader = torch.utils.data.DataLoader(val_datasets,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True)

        # Replace the data set iterator with CUDA to speed up
        if self.device.type == "cuda":
            train_dataloader = CUDAPrefetcher(train_dataloader, self.device)
            val_dataloader = CUDAPrefetcher(val_dataloader, self.device)
        else:
            train_dataloader = CPUPrefetcher(train_dataloader)
            val_dataloader = CPUPrefetcher(val_dataloader)
        return train_dataloader, val_dataloader

    def get_g_model(self):
        if self.model_g_type == "rrdbnet_x4":
            g_model = RRDBNet(self.model_g_in_channels,
                              self.model_g_out_channels,
                              self.model_g_channels,
                              self.model_g_growth_channels,
                              self.model_g_num_rrdb)
        else:
            raise NotImplementedError(f"Model type {self.model_g_type} is not implemented.")
        g_model = g_model.to(self.device)
        if self.g_weights_path:
            LOGGER.info(f"Loading state_dict from {self.g_weights_path} for fine-tuning...")
            g_model = load_state_dict(self.g_weights_path, g_model, map_location=self.device)

        if self.verbose:
            LOGGER.info(f"G model: {g_model}")
            model_info = get_model_info(g_model, self.train_config_dict.IMAGE_SIZE, self.device)
            LOGGER.info(f"G model summary: {model_info}")
        return g_model

    def get_g_optimizer(self):
        if self.solver_g_optim not in ["Adam"]:
            raise NotImplementedError(f"Optimizer {self.solver_g_optim} is not implemented. Only support `Adam`.")

        g_optimizer = optim.Adam(self.g_model.parameters(),
                                 lr=self.solver_g_lr,
                                 betas=self.solver_g_betas,
                                 eps=self.solver_g_eps,
                                 weight_decay=self.solver_g_weight_decay)
        LOGGER.info(f"G optimizer: {g_optimizer}")

        return g_optimizer

    def get_g_lr_scheduler(self):
        if self.solver_g_lr_scheduler_type not in ["StepLR"]:
            raise NotImplementedError(f"Scheduler {self.solver_g_lr_scheduler_type} is not implemented. Only support `StepLR`.")

        g_lr_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer,
                                                   step_size=self.solver_g_lr_scheduler_step_size,
                                                   gamma=self.solver_g_lr_scheduler_gamma)
        LOGGER.info(f"G LR scheduler: ``{self.solver_g_lr_scheduler_type}``")
        return g_lr_scheduler

    def resume_g_model(self):
        resume_state_dict = self.g_checkpoint["model"].float().state_dict()
        self.g_model.load_state_dict(resume_state_dict, strict=True)
        self.start_epoch = self.g_checkpoint["epoch"] + 1
        self.g_optimizer.load_state_dict(self.g_checkpoint["optimizer"])
        self.g_lr_scheduler.load_state_dict(self.g_checkpoint["scheduler"])
        self.ema.ema.load_state_dict(self.g_checkpoint["ema"].float().state_dict())
        self.ema.updates = self.g_checkpoint["updates"]
        LOGGER.info(f"Resumed G model from epoch {self.start_epoch}")

    def define_psnr_loss(self) -> nn.L1Loss:
        if self.loss_pixel_type not in ["l1", "l2"]:
            raise NotImplementedError(f"Loss type {self.loss_pixel_type} is not implemented. Only support `l1` and `l2`.")

        if self.loss_pixel_type == "l1":
            LOGGER.info(f"Pixel-wise loss: ``L1 loss``.")
            pixel_criterion = nn.L1Loss()
        else:
            LOGGER.info(f"Pixel-wise loss: ``MSE loss``.")
            pixel_criterion = nn.MSELoss()

        return pixel_criterion.to(device=self.device)

    def train(self):
        if self.phase == "psnr":
            self.train_psnr()
        else:
            self.train_gan()

        if self.device != "cpu":
            torch.cuda.empty_cache()

    def train_psnr(self):
        for epoch in range(self.start_epoch, self.epochs):
            # The information printed by the progress bar
            batch_time = AverageMeter("Time", ":6.3f")
            data_time = AverageMeter("Data", ":6.3f")
            losses = AverageMeter("Loss", ":6.6f")
            progress = ProgressMeter(self.num_train_batch, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch}]")
            # train pipeline
            # Put the generator in training mode
            self.g_model.train()

            # Initialize the number of data batches to print logs on the terminal
            batch_index = 0

            # Initialize the data loader and load the first batch of data
            self.train_dataloader.reset()
            batch_data = self.train_dataloader.next()

            # Get the initialization training time
            end = time.time()

            while batch_data is not None:
                # Calculate the time it takes to load a batch of data
                data_time.update(time.time() - end)

                gt = batch_data["gt"].to(device=self.device, non_blocking=True)
                gaussian_kernel1 = batch_data["gaussian_kernel1"].to(device=self.device, non_blocking=True)
                gaussian_kernel2 = batch_data["gaussian_kernel2"].to(device=self.device, non_blocking=True)
                sinc_kernel = batch_data["sinc_kernel"].to(device=self.device, non_blocking=True)
                loss_weight = torch.Tensor(self.loss_pixel_weight).to(device=self.device)

                # Get the degraded low-resolution image
                gt_usm, gt, lr = degradation_process(gt,
                                                     gaussian_kernel1,
                                                     gaussian_kernel2,
                                                     sinc_kernel,
                                                     self.upscale_factor,
                                                     self.degradation_process_parameters_dict,
                                                     self.jpeg_operation,
                                                     self.usm_sharpener)

                # image data augmentation
                (gt_usm, gt), lr = random_crop_torch([gt_usm, gt], lr, self.train_image_size, self.upscale_factor)
                (gt_usm, gt), lr = random_rotate_torch([gt_usm, gt], lr, self.upscale_factor, [0, 90, 180, 270])
                (gt_usm, gt), lr = random_vertically_flip_torch([gt_usm, gt], lr)
                (gt_usm, gt), lr = random_horizontally_flip_torch([gt_usm, gt], lr)

                # Initialize the generator gradient
                self.g_model.zero_grad(set_to_none=True)

                # Mixed precision training
                with amp.autocast():
                    sr = self.g_model(lr)
                    loss = self.pixel_criterion(sr, gt_usm)
                    loss = torch.sum(torch.mul(loss_weight, loss))

                # Backpropagation
                self.scaler.scale(loss).backward()
                # update generator weights
                self.scaler.step(self.g_optimizer)
                self.scaler.update()

                # update exponential average model weights
                self.ema.update(self.g_model)

                # Statistical loss value for terminal data output
                losses.update(loss.item(), lr.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # Record training log information
                if batch_index % 100 == 0:
                    # Writer Loss to file
                    self.tblogger.add_scalar("Train/Loss", loss.item(), batch_index + epoch * self.train_batch_size + 1)
                    progress.display(batch_index)

                # Preload the next batch of data
                batch_data = self.train_dataloader.next()

                # Add 1 to the number of data batches
                batch_index += 1

            # Update the learning rate after each training epoch
            self.g_lr_scheduler.step()

            # Evaluate the model after each training epoch
            psnr, ssim, _ = self.evaler.evaluate(self.val_dataloader, self.g_model)
            # update attributes for ema model
            self.ema.update_attr(self.g_model)

            is_best = psnr > self.best_psnr or ssim > self.best_ssim
            is_last = (epoch + 1) == self.epochs

            # save ckpt
            ckpt = {
                "model": deepcopy(self.g_model).half(),
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": self.g_optimizer.state_dict(),
                "scheduler": self.g_lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            save_ckpt_dir = Path(self.save_dir) / "weights"
            save_checkpoint(ckpt, is_best, save_ckpt_dir, model_name="g_last_checkpoint", best_model_name="g_best_checkpoint")

            del ckpt

    def train_gan(self):
        pass
