import math
import mmap
import os
import uuid

import numpy as np

import async_mmap
from fflags import FeatureFlags
from utils import NP_DTYPES, get_hugepage_size


class MemoryMappedTensor:
    def __init__(
        self,
        dir,
        length,
        tensor_shape,
        dtype,
        fflags: FeatureFlags,
    ):
        self.tensor_shape = tensor_shape
        self.tensor_size = math.prod(tensor_shape) * NP_DTYPES[dtype]().itemsize
        self.dtype = dtype
        self.max_length = length
        self.page_size = mmap.PAGESIZE * fflags.pages_per_flush
        self.size = self.max_length * self.tensor_size
        self.closed = False

        self.path = os.path.join(dir, f"{uuid.uuid4()}.bin")
        with open(self.path, "wb") as f:
            f.seek(self.size - 1)
            f.write(b"\0")

        self.file = open(self.path, "r+b")

        if fflags.use_async_mmap:
            self.mmap = async_mmap.async_mmap(
                self.file.fileno(),
                length=0,
                offset=0,
                # flags=fflags.mmap_flags,
                # prot=mmap.PROT_WRITE,
                access=mmap.ACCESS_WRITE,
            )
        else:
            self.mmap = mmap.mmap(
                self.file.fileno(),
                length=0,
                offset=0,
                # flags=fflags.mmap_flags,
                # prot=mmap.PROT_WRITE,
                access=mmap.ACCESS_WRITE,
            )

        if fflags.use_madv_sequential:
            self.mmap.madvise(mmap.MADV_SEQUENTIAL)

        if fflags.use_madv_hugepage:
            if hasattr(
                mmap, "MADV_HUGEPAGE"
            ):  # mmap.MADV_HUGEPAGE only exists if the current system supports huge pages
                self.mmap.madvise(mmap.MADV_HUGEPAGE)
                self.page_size = get_hugepage_size()
            else:
                print(
                    "Warning: MADV_HUGEPAGE specified, but not supported on current architecture"
                )

        # We track this here to avoid reads from async_mmap
        self.length = 0

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

        data = tensor.cpu().numpy().tobytes()

        self.length += len(data) // self.tensor_size

        # Ensure the tensor will fit
        if self.length >= self.max_length:
            raise IndexError("Memory-mapped tensor is full")

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
        return self.length

    def close(self):
        if self.closed:
            return

        self.mmap.flush()
        self.mmap.close()
        self.file.close()
        os.remove(self.path)
        self.closed = True
