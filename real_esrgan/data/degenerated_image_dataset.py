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
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data
from torch import Tensor

from real_esrgan.data.degradations import random_mixed_kernels, generate_sinc_kernel
from real_esrgan.utils.imgproc import image_to_tensor

__all__ = [
    "DegeneratedImageDataset",
]


class DegeneratedImageDataset(torch.utils.data.Dataset):
    r"""Define degenerate dataset loading method.

    Args:
        gt_images_dir (str or Path): Ground-truth dataset address
        degradation_model_parameters_dict (dict): Parameter dictionary with degenerate model
    """

    def __init__(
            self,
            gt_images_dir: str | Path,
            degradation_model_parameters_dict: dict
    ) -> None:
        super(DegeneratedImageDataset, self).__init__()
        # Get a list of all image filenames
        if isinstance(gt_images_dir, str):
            gt_images_dir = Path(gt_images_dir)
        self.gt_image_file_names = [p for p in gt_images_dir.glob("*")]
        # Define the probability of each processing operation in the first-order degradation
        self.degradation_model_parameters_dict = degradation_model_parameters_dict
        # Define the size of the sinc filter kernel
        self.sinc_tensor = torch.zeros([degradation_model_parameters_dict["sinc_kernel_size"],
                                        degradation_model_parameters_dict["sinc_kernel_size"]]).float()
        self.sinc_tensor[degradation_model_parameters_dict["sinc_kernel_size"] // 2,
                         degradation_model_parameters_dict["sinc_kernel_size"] // 2] = 1

    def __getitem__(
            self,
            batch_index: int
    ) -> [Tensor, Tensor, Tensor] or [Tensor, Tensor]:
        # Generate a first-order degenerate Gaussian kernel
        gaussian_kernel_size1 = random.choice(self.degradation_model_parameters_dict["gaussian_kernel_range"])
        if np.random.uniform() < self.degradation_model_parameters_dict["sinc_kernel_probability1"]:
            # This sinc filter setting applies to kernels in the range [7, 21] and can be adjusted dynamically
            if gaussian_kernel_size1 < int(np.median(self.degradation_model_parameters_dict["gaussian_kernel_range"])):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            gaussian_kernel1 = generate_sinc_kernel(
                omega_c,
                gaussian_kernel_size1,
                padding=False)
        else:
            gaussian_kernel1 = random_mixed_kernels(
                self.degradation_model_parameters_dict["gaussian_kernel_type"],
                self.degradation_model_parameters_dict["gaussian_kernel_probability1"],
                gaussian_kernel_size1,
                self.degradation_model_parameters_dict["gaussian_sigma_range1"],
                self.degradation_model_parameters_dict["gaussian_sigma_range1"],
                [-math.pi, math.pi],
                self.degradation_model_parameters_dict["generalized_kernel_beta_range1"],
                self.degradation_model_parameters_dict["plateau_kernel_beta_range1"],
                noise_range=None)
        # First-order degenerate Gaussian fill kernel size
        pad_size = (self.degradation_model_parameters_dict["gaussian_kernel_range"][-1] - gaussian_kernel_size1) // 2
        gaussian_kernel1 = np.pad(gaussian_kernel1, ((pad_size, pad_size), (pad_size, pad_size)))

        # Generate a second-order degenerate Gaussian kernel
        gaussian_kernel_size2 = random.choice(self.degradation_model_parameters_dict["gaussian_kernel_range"])
        if np.random.uniform() < self.degradation_model_parameters_dict["sinc_kernel_probability2"]:
            # This sinc filter setting applies to kernels in the range [7, 21] and can be adjusted dynamically
            if gaussian_kernel_size2 < int(np.median(self.degradation_model_parameters_dict["gaussian_kernel_range"])):
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            gaussian_kernel2 = generate_sinc_kernel(
                omega_c,
                gaussian_kernel_size2,
                padding=False)
        else:
            gaussian_kernel2 = random_mixed_kernels(
                self.degradation_model_parameters_dict["gaussian_kernel_type"],
                self.degradation_model_parameters_dict["gaussian_kernel_probability2"],
                gaussian_kernel_size2,
                self.degradation_model_parameters_dict["gaussian_sigma_range2"],
                self.degradation_model_parameters_dict["gaussian_sigma_range2"],
                [-math.pi, math.pi],
                self.degradation_model_parameters_dict["generalized_kernel_beta_range2"],
                self.degradation_model_parameters_dict["plateau_kernel_beta_range2"],
                noise_range=None)

        # second-order degenerate Gaussian fill kernel size
        pad_size = (self.degradation_model_parameters_dict["gaussian_kernel_range"][-1] - gaussian_kernel_size2) // 2
        gaussian_kernel2 = np.pad(gaussian_kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # Sinc filter kernel
        if np.random.uniform() < self.degradation_model_parameters_dict["sinc_kernel_probability3"]:
            gaussian_kernel_size2 = random.choice(self.degradation_model_parameters_dict["gaussian_kernel_range"])
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = generate_sinc_kernel(
                omega_c,
                gaussian_kernel_size2,
                padding=self.degradation_model_parameters_dict["sinc_kernel_size"])
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.sinc_tensor

        gaussian_kernel1 = torch.FloatTensor(gaussian_kernel1)
        gaussian_kernel2 = torch.FloatTensor(gaussian_kernel2)
        sinc_kernel = torch.FloatTensor(sinc_kernel)

        # read a batch of images
        gt_image = cv2.imread(str(self.gt_image_file_names[batch_index])).astype(np.float32) / 255.

        # BGR image data to RGB image data
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        # Convert the RGB image data channel to a data format supported by PyTorch
        gt_tensor = image_to_tensor(gt_image, False, False)

        return {"gt": gt_tensor,
                "gaussian_kernel1": gaussian_kernel1,
                "gaussian_kernel2": gaussian_kernel2,
                "sinc_kernel": sinc_kernel}

    def __len__(self) -> int:
        return len(self.gt_image_file_names)
