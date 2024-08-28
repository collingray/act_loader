import math
import mmap
import os
import random
import uuid

import torch
import numpy as np

from utils import NP_DTYPES


class MemoryMappedTensor:
    def __init__(self, dir, length, tensor_shape, dtype):
        self.tensor_shape = tensor_shape
        self.tensor_size = math.prod(tensor_shape) * NP_DTYPES[dtype]().itemsize
        self.dtype = dtype
        self.max_length = length
        self.page_size = os.sysconf("SC_PAGE_SIZE")

        self.size = self.max_length * self.tensor_size

        self.path = os.path.join(dir, f"{uuid.uuid4()}.bin")
        with open(self.path, "wb") as f:
            f.seek(self.size - 1)
            f.write(b"\0")

        self.file = open(self.path, "r+b")
        self.mmap = mmap.mmap(
            self.file.fileno(), length=0, access=mmap.ACCESS_WRITE, offset=0
        )

        self.mmap.madvise(mmap.MADV_SEQUENTIAL)

        # Write head location
        self.pointer = 0

        # Bytes of data written but not yet flushed
        self.unflushed = 0

    def append(self, tensor):
        # Add a batch dimension if missing
        if tensor.dim() != len(self.tensor_shape) + 1:
            tensor = tensor.unsqueeze(0)

        # Ensure the tensor is the correct shape
        if tensor.shape[1:] != self.tensor_shape:
            raise ValueError(
                f"Tensor shape mismatch. Expected {self.tensor_shape}, got {tensor.shape}"
            )

        # Ensure the tensor will fit
        if self.pointer >= self.max_length:
            raise IndexError("Memory-mapped tensor is full")

        # start = self.pointer * self.tensor_size
        data = tensor.cpu().numpy().tobytes()

        # if start % self.page_size != 0:
        #     i = min(self.page_size - (start % self.page_size), len(bytes))
        #     self.mmap[start : start + i] = bytes[:i]
        #     bytes = bytes[i:]
        #     start += i

        # if start % self.page_size == 0:
        #     self.mmap.flush()

        # for i in range(0, len(bytes), self.page_size):
        #     j = min(self.page_size, len(bytes) - i)
        #     self.mmap[start + i : start + i + j] = bytes[i : i + j]
        #     self.mmap.flush()
        #
        # self.mmap.

        if self.unflushed + len(data) > self.page_size:
            self.mmap.write(data[: self.page_size - self.unflushed])
            data = data[self.page_size - self.unflushed :]
            self.mmap.flush()
            self.unflushed = 0

        self.mmap.write(data)
        self.unflushed += len(data)
        del data

        # self.mmap.madvise(mmap.MADV_DONTNEED, 0, len(self.mmap))

        # MAdvise DONTNEED on all pages just written
        # start = self.pointer * self.tensor_size
        # start_page = start // self.page_size
        # end_page = (start + len(bytes)) // self.page_size
        #
        # if start_page != end_page:  # Only call if we crossed a page boundary
        # self.mmap.madvise(
        #     mmap.MADV_DONTNEED,
        #     0,
        #     (len(self.mmap) // self.page_size) * self.page_size,
        # )

        # Increment the write head
        # self.pointer += tensor.shape[0]

        # self.data._mmap.madvise(mmap.MADV_DONTNEED, 0, end * self.data.itemsize)

    def get_data(self):
        # Ensure all data is written to disk
        if self.unflushed > 0:
            self.mmap.flush()

        return (
            np.frombuffer(
                self.mmap[: self.mmap.tell()],
                dtype=NP_DTYPES[self.dtype],
            )
            .reshape((-1, *self.tensor_shape))
            .copy()
        )

    def __len__(self):
        return self.mmap.tell()

    def close(self):
        self.mmap.flush()
        self.mmap.close()
        self.file.close()
        os.remove(self.path)
