from typing import Optional, Tuple

import torch
from transformers.models.qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


class WrappedQwen2Block(Qwen2DecoderLayer):
    pass