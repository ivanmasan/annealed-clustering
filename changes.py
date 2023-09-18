from dataclasses import dataclass

import numpy as np


@dataclass
class MatrixChange:
    sku_id: int
    cluster_id: int
    delta: float


