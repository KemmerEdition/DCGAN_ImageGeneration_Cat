from typing import List, Tuple

import torch


def collate_fn(batch: List[Tuple]):
    return {
        "img": torch.stack(batch),
    }
