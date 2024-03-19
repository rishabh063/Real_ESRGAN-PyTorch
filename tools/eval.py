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
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from real_esrgan.engine.evaler import Evaler
from real_esrgan.utils.envs import select_device


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path",
        metavar="FILE",
        help="path to config file",
    )
    return parser.parse_args()


def main() -> None:
    opts = get_opts()
    config_path = opts.config_path

    config_dict = OmegaConf.load(config_path)
    # merge _BASE_ config
    base_config_path = config_dict.get("_BASE_", False)
    if base_config_path:
        base_config_dict = OmegaConf.load(Path(config_path).absolute().parent / Path(base_config_path))
        config_dict = OmegaConf.merge(base_config_dict, config_dict)
    device = select_device(config_dict.DEVICE)

    evaler = Evaler(config_dict, device)
    psnr, ssim, niqe = evaler.evaluate()
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim:.4f}, NIQE: {niqe:.2f}")


if __name__ == "__main__":
    main()
