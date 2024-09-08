import asyncio
import mmap
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import time
import os


class async_mmap:
    def __init__(self, fileno, length, **kwargs):
        self.mmap = mmap.mmap(fileno, length, **kwargs)
        self.executor = ThreadPoolExecutor()
        self.operation_queue = deque()
        self.lock = asyncio.Lock()
        self.loop = asyncio.get_event_loop()
        self.background_task = self.loop.create_task(self._process_operations())

    def write(self, data, offset=0):
        self.operation_queue.append(("write", data, offset))
        return len(data)

    def flush(self):
        self.operation_queue.append(("flush",))
        return

    async def _process_operations(self):
        while True:
            if self.operation_queue:
                operation = self.operation_queue.popleft()
                if operation[0] == "write":
                    await self._perform_write(operation[1], operation[2])
                elif operation[0] == "flush":
                    await self._perform_flush()
            else:
                await asyncio.sleep(0.001)  # Small delay to prevent busy-waiting

    async def _perform_write(self, data, offset):
        async with self.lock:
            await self.loop.run_in_executor(
                self.executor, self.mmap.write, data, offset
            )

    async def _perform_flush(self):
        async with self.lock:
            await self.loop.run_in_executor(self.executor, self.mmap.flush)

    def close(self):
        self.background_task.cancel()  # Cancel the background task
        self.mmap.close()
        self.loop.close()
        self.executor.shutdown()

    def tell(self):
        return self.mmap.tell()

    def madvise(self, option, **kwargs):
        self.mmap.madvise(option, **kwargs)

    def __getitem__(self, item):
        return self.mmap[item]


async def _speed_test_async_mmap(filename, size, data):
    with open(filename, "r+b") as f:
        mm = async_mmap(f.fileno(), size)
        start_time = time.time()

        chunk_size = 1024 * 1024  # 1 MB chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            mm.write(chunk, i)

        mm.flush()
        mm.close()

        end_time = time.time()
    return end_time - start_time


def _speed_test_default_mmap(filename, size, data):
    with open(filename, "r+b") as f:
        mm = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_WRITE)
        start_time = time.time()

        chunk_size = 1024 * 1024  # 1 MB chunks
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            mm.write(chunk)

        mm.flush()
        mm.close()

        end_time = time.time()
    return end_time - start_time


async def _run_speed_test():
    size = 500 * 1024 * 1024  # 500 MB
    data = os.urandom(size)  # Generate random data

    async_filename = "async_test.bin"
    default_filename = "default_test.bin"

    for filename in [async_filename, default_filename]:
        with open(filename, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")

    print(f"Testing with {size / (1024 * 1024):.2f} MB of data")

    # Test async_mmap
    async_time = await _speed_test_async_mmap(async_filename, size, data)
    print(f"async_mmap write time: {async_time:.4f} seconds")

    # Test mmap.mmap
    default_time = _speed_test_default_mmap(default_filename, size, data)
    print(f"Default mmap write time: {default_time:.4f} seconds")

    # Compare results
    speedup = (default_time - async_time) / default_time * 100
    print(f"async_mmap is {speedup:.2f}% faster than default mmap")

    # Clean up test files
    os.remove(async_filename)
    os.remove(default_filename)


if __name__ == "__main__":
    asyncio.run(_run_speed_test())
