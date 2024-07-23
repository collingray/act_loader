import math
import os

import torch
import numpy as np

from act_loader.utils import NP_DTYPES


class MemoryMappedTensor:
    def __init__(self, path, length, tensor_shape, dtype):
        self.tensor_shape = tensor_shape
        self.tensor_size = math.prod(tensor_shape)
        self.length = length

        total_size = length * self.tensor_size

        # Create the file if it doesn't exist
        if not os.path.exists(path):
            empty_data = np.zeros(total_size, dtype=NP_DTYPES[dtype])
            empty_data.tofile(path)

        # Open the file in memory-map mode
        self.data = np.memmap(
            path, dtype=NP_DTYPES[dtype], mode="r+", shape=(total_size,)
        )

        # Write head location
        self.pointer = 0

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
        if self.pointer >= self.length:
            raise IndexError("Memory-mapped tensor is full")

        # Calculate the start and end indices for this tensor
        start = self.pointer * self.tensor_size
        end = start + self.tensor_size * tensor.shape[0]

        # Write the tensor data to the file
        self.data[start:end] = tensor.cpu().flatten()

        # Flush to ensure data is written to disk
        self.data.flush()

        # Increment the write head
        self.pointer += tensor.shape[0]

    def read(self, idx):
        if idx >= self.pointer:
            raise IndexError("Tensor index out of range")

        # Calculate the start and end indices for this tensor
        start = idx * self.tensor_size
        end = start + self.tensor_size

        # Read the tensor data from the file
        tensor_data = self.data[start:end]

        # Reshape and convert to PyTorch tensor
        return torch.from_numpy(tensor_data.reshape(self.tensor_shape))

    def read_batch(self, start, end):
        if start > self.pointer:
            raise IndexError("Batch start index out of range")

        # Calculate the start and end indices for this batch
        start = start * self.tensor_size
        end = end * self.tensor_size

        # Clip the end index to the current length, otherwise the batch will be padded with zeros
        end = min(end, self.pointer * self.tensor_size)

        # Read the tensor data from the file
        tensor_data = self.data[start:end]

        # Reshape and convert to PyTorch tensor
        return torch.from_numpy(tensor_data.reshape(-1, *self.tensor_shape))

    def get_data(self):
        return self.data[: self.pointer * self.tensor_size].reshape(
            (-1, *self.tensor_shape)
        )

    def shuffle_data(self, generator=None):
        if generator is None:
            generator = np.random.Generator(np.random.PCG64())

        reshaped = self.data[: self.pointer * self.tensor_size].reshape(
            (-1, *self.tensor_shape)
        )
        generator.shuffle(reshaped)
        self.data[: self.pointer * self.tensor_size] = reshaped.flatten()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.read_batch(idx.start, idx.stop)
        else:
            return self.read(idx)

    def __len__(self):
        return self.pointer

    def __iter__(self):
        for i in range(self.pointer):
            yield self.read(i)

    def batch_iter(self, batch_size):
        for i in range(0, self.pointer, batch_size):
            yield self.read_batch(i, i + batch_size)

    def close(self):
        del self.data
