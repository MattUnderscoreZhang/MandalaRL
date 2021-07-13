from jax import numpy as np
from enum import IntEnum


Key = np.ndarray
ColorCount = np.ndarray  # [n x 6] int array

Color = IntEnum('Color', ['RED', 'BLACK', 'PURPLE', 'YELLOW', 'GREEN', 'ORANGE'], start=0)
Zone = IntEnum('Zone', ['DECK', 'DISCARD',
                        'M1_MOUNTAIN', 'M1_P1_FIELD', 'M1_P2_FIELD',
                        'M2_MOUNTAIN', 'M2_P1_FIELD', 'M2_P2_FIELD',
                        'P1_HAND', 'P2_HAND', 'P1_RIVER', 'P2_RIVER', 'P1_CUP', 'P2_CUP'], start=0)

n_colors = 6
n_zones = 14
