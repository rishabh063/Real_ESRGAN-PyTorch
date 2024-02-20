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
import os
import shutil
from typing import Optional

__all__ = [
    "configure_logging", "LOGGER", "NCOLS",
]


def configure_logging(name: Optional[str] = None) -> logging.Logger:
    r"""Configures the logging module.

    Args:
        name (Optional[str], optional): The name of the logger. Defaults to None.

    Returns:
        logging.Logger: The configured logger.
    """
    # Get the rank from the environment variables. If not found, default to -1
    rank = int(os.getenv("RANK", -1))

    # Set the basic configuration for the logging module
    # If the rank is -1 or 0, set the logging level to INFO, otherwise set it to WARNING
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)

    # Return the logger with the specified name
    return logging.getLogger(name)


LOGGER = configure_logging(__name__)
NCOLS = min(100, shutil.get_terminal_size().columns)
