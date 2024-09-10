import asyncio
import mmap
import os
import queue
import tempfile
import threading
import time

from dotenv import load_dotenv
from tqdm import tqdm


class async_mmap:
    def __init__(self, fileno, length, **kwargs):
        self.mmap = mmap.mmap(fileno, length, **kwargs)
        self.write_queue = queue.Queue()
        self.write_thread = threading.Thread(
            target=self._background_writer, daemon=True
        )
        self.lock = threading.Lock()
        self.write_thread.start()
        self.closed = False

    def _background_writer(self):
        while True:
            data = self.write_queue.get()

            if data is None:
                break

            self.mmap.write(data)

    def write(self, data):
        with self.lock:
            if self.closed:
                raise ValueError("Cannot write to closed async_mmap")

            self.write_queue.put(data)

    def flush(self):
        with self.lock:
            self.mmap.flush()

    def close(self):
        with self.lock:
            if not self.closed:
                self.write_queue.put(None)  # Signal to stop the background thread
                self.write_thread.join()
                self.mmap.flush()
                self.mmap.close()
                self.closed = True

    def tell(self):
        return self.mmap.tell()

    def madvise(self, option, **kwargs):
        return self.mmap.madvise(option, **kwargs)

    def __getitem__(self, item):
        return self.mmap[item]


def _speed_test_default_mmap(filename, size, data, delay=0.0):
    file = open(filename, "r+b")

    mm = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_WRITE)
    start_time = time.time()

    chunk_size = 1024 * 1024  # 1 MB chunks
    for i in tqdm(range(0, len(data), chunk_size), desc="Writing data (default)"):
        chunk = data[i : i + chunk_size]
        mm.write(chunk)
        time.sleep(delay)

    mm.flush()
    mm.close()

    end_time = time.time()

    file.close()

    return end_time - start_time


def _speed_test_async_mmap(filename, size, data, delay=0.0):
    file = open(filename, "r+b")

    mm = async_mmap(file.fileno(), size, access=mmap.ACCESS_WRITE)
    start_time = time.time()

    chunk_size = 1024 * 1024  # 1 MB chunks
    for i in tqdm(range(0, len(data), chunk_size), desc="Writing data (async)"):
        chunk = data[i : i + chunk_size]
        mm.write(chunk)
        time.sleep(delay)

    mm.flush()
    mm.close()

    end_time = time.time()

    file.close()

    return end_time - start_time


async def _run_speed_test():
    load_dotenv()

    size = 1 * 1024 * 1024 * 1024  # 1 GiB
    data = os.urandom(size)  # Generate random data

    output_dir = os.getenv("OUTPUT_DIR", ".")

    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        default_filename = os.path.join(tmp_dir, "default_test.bin")
        async_filename = os.path.join(tmp_dir, "async_test.bin")

        for filename in [default_filename, async_filename]:
            with open(filename, "wb") as f:
                f.seek(size - 1)
                f.write(b"\0")

        print(f"Testing with {size / (1024 * 1024 * 1024):.2f} GiB of data")

        for delay in [0.0, 0.01, 0.1]:
            print(f"Testing with delay of {delay:.2f} seconds")

            # Test mmap.mmap
            default_time = _speed_test_default_mmap(default_filename, size, data, delay)
            print(f"Default mmap write time: {default_time:.4f} seconds")

            # Verify that the default file was written correctly
            with open(default_filename, "rb") as f:
                assert f.read() == data, "Default mmap did not write the correct data"

            # Test async_mmap
            async_time = _speed_test_async_mmap(async_filename, size, data, delay)
            print(f"async_mmap write time: {async_time:.4f} seconds")

            # Verify that the async file was written correctly
            with open(async_filename, "rb") as f:
                assert f.read() == data, "Async mmap did not write the correct data"

            # Compare results
            speedup = (default_time - async_time) / default_time * 100
            print(f"async_mmap is {speedup:.2f}% faster than default mmap")


if __name__ == "__main__":
    asyncio.run(_run_speed_test())
