import os
from time import time

import datasets
from dotenv import load_dotenv

from fflags import FeatureFlags
from utils import auto_device
from shuffle import tl_generate_acts, shuffle_acts


load_dotenv()

n_tokens = int(2e7)
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
    act_batch = 512

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

    flag_sets = [
        ("all flags", FeatureFlags()),
        (
            "no flags",
            FeatureFlags(
                use_async_mmap=False,
                use_madv_sequential=False,
                use_madv_hugepage=False,
                use_map_private=False,
            ),
        ),
        ("w/o async mmap", FeatureFlags(use_async_mmap=False)),
        ("w/o synchronized flushes", FeatureFlags(sync_flushes=True)),
        ("w/o MADV_SEQUENTIAL", FeatureFlags(use_madv_sequential=False)),
        ("w/o MADV_HUGEPAGE", FeatureFlags(use_madv_hugepage=False)),
        # ("w/o MADV_DONTNEED", FeatureFlags(use_madv_dontneed=False)),
        # ("w/o MAP_PRIVATE", FeatureFlags(use_map_private=False)),
    ]

    for name, flags in flag_sets:
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
    act_batch = 512

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


if __name__ == "__main__":
    print("Testing feature flags...")
    test_feature_flags()

    # print("Testing activation batch sizes...")
    # test_activation_batch()
