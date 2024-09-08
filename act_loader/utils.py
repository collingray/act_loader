from dataclasses import dataclass
from random import random
from math import sqrt, log, exp, ceil
import mmap

import numpy as np
import torch


def truncate_seq(seq, max_length):
    offset = int(random() * (len(seq) - max_length))
    return seq[offset : offset + max_length]


TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

NP_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
}


def β(n, k):
    return sqrt(n * (k - 1) / (2 * log(k) * k**2))


def µ(n, k):
    return n / k + β(n, k)


def p(m, n, k):
    return 1 - exp(-exp((µ(n, k) - m) / β(n, k)))


def k_bins(n, max_mem, p_overflow=1e-12):
    if n <= max_mem:
        return 1

    k = ceil(n / max_mem)
    while p(max_mem, n, k) > p_overflow:
        k += 1
    return k


def get_hugepage_size():
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("Hugepagesize:"):
                    return int(line.split()[1]) * 1024  # Convert KB to bytes
    except FileNotFoundError:
        pass  # Not a Linux system or /proc not available
    return None  # Unable to determine hugepage size


def auto_device():
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"
