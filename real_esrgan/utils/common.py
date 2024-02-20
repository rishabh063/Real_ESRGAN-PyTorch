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
import logging
from pathlib import Path
from typing import Union

from torchvision.datasets.folder import IMG_EXTENSIONS

__all__ = [
    "check_dir", "get_all_filenames"
]

logger = logging.getLogger(__name__)


def check_dir(dir_path: Union[str, Path]) -> None:
    r"""Check if the input directory exists and is a directory.

    Args:
        dir_path (str or Path): Input directory path.
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if not dir_path.exists():
        raise FileExistsError(f"Input directory '{dir_path}' not exists.")

    if not dir_path.is_dir():
        raise TypeError(f"'{dir_path}' is not a directory.")


def get_all_filenames(path: str | Path, image_extensions: tuple = None) -> list:
    r"""Get all file names in the input folder.

    Args：
        path (str or Path): Input directory path.
        image_extensions (tuple): Supported image format. Default: ``None``.
    """
    if isinstance(path, str):
        path = Path(path)

    if image_extensions is None:
        image_extensions = IMG_EXTENSIONS

    # Only get file names with specified extensions
    file_paths = path.iterdir()
    file_names = [p.name for p in file_paths if p.suffix in image_extensions]

    return file_names
