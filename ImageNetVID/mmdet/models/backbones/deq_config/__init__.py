# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from .default import _C as deq_config
from .default import update_config
from .models import MODEL_EXTRAS

__all__ = ['deq_config', 'update_config', 'MODEL_EXTRAS']
