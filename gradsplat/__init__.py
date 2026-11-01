# Copyright (c) 2026 The GradSplat Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

from .model import GradSplat, GradSplatConfig
from .render import render

__all__ = ["GradSplat", "GradSplatConfig", "render"]
__version__ = "1.0.0"
