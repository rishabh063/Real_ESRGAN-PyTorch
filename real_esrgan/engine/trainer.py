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
from pathlib import Path

import torch.utils.data
from omegaconf import DictConfig, OmegaConf

from real_esrgan.utils.envs import select_device, set_seed_everything
from real_esrgan.utils.events import LOGGER
from real_esrgan.utils.general import increment_name, find_last_checkpoint


def init_train_env(config_dict: DictConfig) -> [DictConfig, torch.device]:
    r"""Initialize the training environment.

    Args:
        config_dict (DictConfig): The configuration dictionary.

    Returns:
        device (torch.device): The device to be used for training.
    """
    # Define the name of the configuration file
    save_config_name = "config.yaml"

    # Handle the resume training case
    if config_dict.TRAIN.RESUME:
        # Do not care!
        checkpoint_path = config_dict.TRAIN.RESUME if isinstance(config_dict.TRAIN.RESUME, str) else find_last_checkpoint()
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
        config_dict.TRAIN.RESUME = checkpoint_path  # set the args.resume to checkpoint path.
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
        pass

    def train_psnr(self):
        pass

    def train_ga(self):
        pass
