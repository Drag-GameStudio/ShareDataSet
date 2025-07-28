import numpy as np

def liner_opacity(start, end, kof):
    return (end * kof + start * (1 - kof)).astype(np.uint8)