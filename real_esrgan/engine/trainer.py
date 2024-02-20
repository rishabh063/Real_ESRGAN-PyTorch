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

from real_esrgan.data.degenerated_image_dataset import DegeneratedImageDataset
from real_esrgan.data.paired_image_dataset import PairedImageDataset
from real_esrgan.data.prefetcher import CUDAPrefetcher, CPUPrefetcher
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
        self.degradation_model_parameters_dict = self.config_dict.get("DEGRADATION_MODEL_PARAMETERS_DICT")
        self.degradation_process_parameters_dict = self.config_dict.get("DEGRADATION_PROCESS_PARAMETERS_DICT")
        self.mode = self.config_dict.MODE
        self.dataset_config_dict = self.config_dict.DATASET
        self.model_config_dict = self.config_dict.MODEL
        self.train_config_dict = self.config_dict.TRAIN
        self.eval_config_dict = self.config_dict.EVAL

        # ========== Init all config ==========
        # datasets
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
        self.solver_optim = self.train_config_dict.SOLVER.OPTIM
        self.solver_lr = self.train_config_dict.SOLVER.LR
        self.solver_betas = list(self.train_config_dict.SOLVER.BETAS)
        self.solver_eps = self.train_config_dict.SOLVER.EPS
        self.solver_weight_decay = self.train_config_dict.SOLVER.WEIGHT_DECAY
        self.solver_lr_scheduler_type = self.train_config_dict.SOLVER.LR_SCHEDULER.TYPE
        self.solver_lr_scheduler_step_size = self.train_config_dict.SOLVER.LR_SCHEDULER.STEP_SIZE
        self.solver_lr_scheduler_gamma = self.train_config_dict.SOLVER.LR_SCHEDULER.GAMMA
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
        # datasets
        self.train_dataloader, self.val_dataloader = self.get_dataloader()
        self.num_train_batch = len(self.train_dataloader)

    def get_dataloader(self):
        if self.mode == "degradation":
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

    def train_psnr(self):
        pass

    def train_ga(self):
        pass
