import os
from time import time

import datasets
from dotenv import load_dotenv

from fflags import FeatureFlags
from utils import auto_device
from shuffle import tl_generate_acts, shuffle_acts


load_dotenv()

n_tokens = int(1e8)
layers = [1]
sites = ["hook_mlp_out"]
d_mlp = 768
dtype = "float16"
max_bytes = 2 * (1024**3)  # 2 GiB
p_overflow = 1e-12

model_name = "gpt2"
dataset_name = "wikitext"
dataset_config = "wikitext-103-v1"
dataset_split = "train"

output_dir = os.getenv("OUTPUT_DIR", ".")

dataset = datasets.load_dataset(dataset_name, dataset_config, split=dataset_split)[
    "text"
]

device = auto_device()


def test_feature_flags():
    model_batch = 48
    act_batch = int(2**19)

    flag_sets = [
        ("all flags", FeatureFlags()),
        ("w/o async mmap", FeatureFlags(use_async_mmap=False)),
        ("w/o MADV_SEQUENTIAL", FeatureFlags(use_madv_sequential=False)),
        ("w/o MADV_DONTNEED", FeatureFlags(use_madv_dontneed=False)),
        ("w/o MADV_HUGEPAGE", FeatureFlags(use_madv_hugepage=False)),
        ("w/o synchronized flushes", FeatureFlags(sync_flushes=False)),
        (
            "no flags",
            FeatureFlags(
                use_async_mmap=False,
                use_madv_sequential=False,
                use_madv_dontneed=False,
                use_madv_hugepage=False,
                sync_flushes=False,
            ),
        ),
    ]

    for name, flags in flag_sets:
        gen = tl_generate_acts(
            model_name,
            dataset,
            layers,
            sites,
            model_batch,
            act_batch,
            dtype,
            device,
        )

        print(f"Testing {name}...")
        start = time()
        shuffle_acts(
            gen,
            n_tokens=n_tokens,
            max_bytes=max_bytes,
            layers=layers,
            sites=sites,
            n_dim=d_mlp,
            act_batch=act_batch,
            output_dir=output_dir,
            p_overflow=p_overflow,
            dtype=dtype,
            fflags=flags,
        )
        end = time()
        print(f"{name}: {end - start:.2f}s")


def test_activation_batch():
    model_batch = 48

    for act_batch in [256, 512, 1024, 2048]:
        gen = tl_generate_acts(
            model_name,
            dataset,
            layers,
            sites,
            model_batch,
            act_batch,
            dtype,
            device,
        )

        print(f"Testing activation batch of {act_batch}...")
        start = time()
        shuffle_acts(
            gen,
            n_tokens=n_tokens,
            max_bytes=max_bytes,
            layers=layers,
            sites=sites,
            n_dim=d_mlp,
            act_batch=act_batch,
            output_dir=output_dir,
            p_overflow=p_overflow,
            dtype=dtype,
        )
        end = time()
        print(f"{act_batch}: {end - start:.2f}s")


if __name__ == "__main__":
    print("Testing feature flags...")
    test_feature_flags()

    print("Testing activation batch sizes...")
    test_activation_batch()
