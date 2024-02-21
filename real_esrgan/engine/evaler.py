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
from typing import Any

import torch.utils.data
from omegaconf import DictConfig
from torch import nn

from real_esrgan.data.paired_image_dataset import PairedImageDataset
from real_esrgan.data.prefetcher import CUDAPrefetcher, CPUPrefetcher
from real_esrgan.evaluation.metrics import PSNR, SSIM, NIQE
from real_esrgan.utils.checkpoint import load_checkpoint
from real_esrgan.utils.events import LOGGER, AverageMeter, ProgressMeter
from real_esrgan.utils.torch_utils import get_model_info


class Evaler:
    def __init__(self, config_dict: DictConfig, device: torch.device) -> None:
        self.config_dict = config_dict
        self.device = device
        self.upscale_factor = self.config_dict.UPSCALE_FACTOR
        self.dataset_config_dict = self.config_dict.DATASET
        self.eval_config_dict = self.config_dict.EVAL

        self.weights_path = self.eval_config_dict.WEIGHTS_PATH
        self.niqe_weights_path = self.eval_config_dict.NIQE_WEIGHTS_PATH
        self.half = self.eval_config_dict.HALF
        self.only_test_y_channel = self.eval_config_dict.ONLY_TEST_Y_CHANNEL

        # IQA model
        self.psnr_model = PSNR(crop_border=self.upscale_factor, only_test_y_channel=self.only_test_y_channel, data_range=1.0)
        self.ssim_model = SSIM(crop_border=self.upscale_factor, only_test_y_channel=self.only_test_y_channel, data_range=255.0)
        self.niqe_model = NIQE(crop_border=self.upscale_factor, niqe_weights_path=self.niqe_weights_path)
        self.psnr_model = self.psnr_model.to(self.device)
        self.ssim_model = self.ssim_model.to(self.device)
        self.niqe_model = self.niqe_model.to(self.device)

    def get_dataloader(self):
        val_datasets = PairedImageDataset(self.dataset_config_dict.VAL_GT_IMAGES_DIR, self.dataset_config_dict.VAL_LR_IMAGES_DIR)
        val_dataloader = torch.utils.data.DataLoader(val_datasets,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1,
                                                     pin_memory=True,
                                                     drop_last=False,
                                                     persistent_workers=True)

        # Replace the data set iterator with CUDA to speed up
        if self.device.type == "cuda":
            val_dataloader = CUDAPrefetcher(val_dataloader, self.device)
        else:
            val_dataloader = CPUPrefetcher(val_dataloader)
        return val_dataloader

    def load_model(self) -> nn.Module:
        model = load_checkpoint(self.weights_path, map_location=self.device)
        model_info = get_model_info(model, device=self.device)
        LOGGER.info(f"Model Summary: {model_info}")

        model.half() if self.half else model.float()
        model.eval()
        return model

    def eval_model(self, dataloader: Any, model: nn.Module, device: torch.device) -> tuple[Any, Any, Any]:
        # The information printed by the progress bar
        batch_time = AverageMeter("Time", ":6.3f")
        psnres = AverageMeter("PSNR", ":4.2f")
        ssimes = AverageMeter("SSIM", ":4.4f")
        niqees = AverageMeter("NIQE", ":4.2f")
        progress = ProgressMeter(len(dataloader), [batch_time, psnres, ssimes, niqees], prefix=f"Eval: ")

        # Set the model as validation model
        model.eval()

        # Initialize data batches
        batch_index = 0

        # Set the data set iterator pointer to 0 and load the first batch of data
        dataloader.reset()
        batch_data = dataloader.next()

        # Record the start time of verifying a batch
        end = time.time()

        # Disable gradient propagation
        with torch.no_grad():
            while batch_data is not None:
                # Load batches of data
                gt = batch_data["gt"].to(device=device, non_blocking=True)
                lr = batch_data["lr"].to(device=device, non_blocking=True)

                if self.half:
                    lr = lr.half()
                    gt = gt.half()

                # inference
                sr = model(lr)

                # Calculate the image IQA
                psnr = self.psnr_model(sr, gt)
                ssim = self.ssim_model(sr.float(), gt.float())
                niqe = self.niqe_model(sr)
                psnres.update(psnr.item(), lr.size(0))
                ssimes.update(ssim.item(), lr.size(0))
                niqees.update(niqe.item(), lr.size(0))

                # Record the total time to verify a batch
                batch_time.update(time.time() - end)
                end = time.time()

                # Output a verification log information
                progress.display(batch_index + 1)

                # Preload the next batch of data
                batch_data = dataloader.next()

                # Add 1 to the number of data batches
                batch_index += 1

            # Print the performance index of the model at the current epoch
            progress.display_summary()

        return psnres.avg, ssimes.avg, niqees.avg

    def evaluate(self, dataloader: Any = None, model: nn.Module = None):
        if dataloader is None:
            dataloader = self.get_dataloader()
        if model is None:
            model = self.load_model()
        psnr, ssim, niqe = self.eval_model(dataloader, model, self.device)

        LOGGER.info(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, NIQE: {niqe:.2f}")
        return psnr, ssim, niqe