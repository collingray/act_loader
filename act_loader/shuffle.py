import math
import os
import random
import resource
import tempfile
from time import time_ns

import datasets
import h5py
import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from fflags import FeatureFlags
from mmap_tensor import MemoryMappedTensor
from utils import k_bins, TORCH_DTYPES, auto_device


@torch.no_grad()
def tl_generate_acts(
    model_name: str,
    dataset,
    layers: int | list[int],
    sites: str | list[str],
    model_batch: int,
    act_batch: int,
    dtype: str,
    device: str,
):
    """
    Generate activations for a given model, dataset, at each layer and site

    :param model_name: The model to generate activations for
    :param dataset: The dataset to generate activations from
    :param layers: The layer(s) to generate activations from
    :param sites: The site(s) to generate activations from
    :param model_batch: The batch size of sequences passed to the model when generating activations
    :param act_batch: The batch size of activations to return
    :param dtype: The data type to use for activations
    :param device: The device to generate activations on
    :return: An iterator over activations
    """

    assert isinstance(layers, int) or (
        isinstance(layers, list) and len(layers) > 0
    ), "'layers' must be an int or non-empty list of ints"
    assert dtype in TORCH_DTYPES, f"'dtype' must be one of {TORCH_DTYPES}"

    layers = [layers] if isinstance(layers, int) else layers
    sites = [sites] if isinstance(sites, str) else sites

    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model_name, device=device, dtype=dtype
    )

    final_layer = max(layers)
    act_names = [f"blocks.{layer}.{site}" for layer in layers for site in sites]

    # while True:
    #     yield torch.rand(len(layers), len(sites), 768, dtype=TORCH_DTYPES[dtype])

    act_buffer = torch.zeros(
        0,
        len(layers),
        len(sites),
        model.cfg.d_model,
        dtype=TORCH_DTYPES[dtype],
        device=device,
    )

    for batch in DataLoader(
        dataset, batch_size=model_batch, shuffle=True, pin_memory=True, num_workers=8
    ):
        out, cache = model.run_with_cache(
            batch, stop_at_layer=final_layer + 1, names_filter=act_names
        )

        del out

        acts = torch.stack(
            [  # [n_layers, n_sites, batch_size, seq_len, d_model]
                torch.stack([cache[f"blocks.{layer}.{site}"] for site in sites])
                for layer in layers
            ]
        )

        del cache

        acts = acts.permute(
            2, 3, 0, 1, 4
        )  # [batch_size, seq_len, n_layers, n_sites, d_model]
        acts = acts.flatten(0, 1)  # [batch_size * seq_len, n_layers, n_sites, d_model]
        act_buffer = torch.cat([act_buffer, acts], dim=0)

        while act_buffer.shape[0] >= act_batch:
            yield act_buffer[:act_batch]
            act_buffer = act_buffer[act_batch:]


def shuffle_acts(
    act_generator,
    n_tokens,
    max_bytes,
    layers: int | list[int],
    sites: str | list[str],
    n_dim: int,
    dtype: str,
    act_batch: int,
    output_dir: str = ".",
    p_overflow=1e-12,
    seed=None,
    fflags=FeatureFlags(),
):
    layers = [layers] if isinstance(layers, int) else layers
    sites = [sites] if isinstance(sites, str) else sites

    shape = (len(layers), len(sites), n_dim)

    # Number of token's worth of activations that can fit in memory
    m = max_bytes // (math.prod(shape) * TORCH_DTYPES[dtype].itemsize)

    # Number of bins to split the activations into, with a `p_overflow` chance that at least one bin will grow larger than `m`
    k = k_bins(n_tokens, m, p_overflow)

    if fflags.set_file_handle_limits:
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        new_limit = soft_limit + k
        if new_limit > hard_limit:
            raise ValueError(
                f"Number of bins ({k}) exceeds the file descriptor limit ({hard_limit})"
            )

        resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))

    if fflags.log_bin_data:
        print(
            f"n_tokens: {n_tokens}, # bins: {k}, bin size: {m}, p_overflow: {p_overflow}"
        )

    timing_data = {}

    if fflags.record_timing:
        timing_data["act_gen"] = np.empty(0, dtype=np.int64)
        timing_data["bin_append"] = np.empty(0, dtype=np.int64)
        timing_data["bin_shuffle"] = np.empty(0, dtype=np.int64)
        timing_data["bin_write"] = np.empty(0, dtype=np.int64)

    try:
        with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
            bins = [
                MemoryMappedTensor(tmp_dir, m, shape, dtype, fflags) for _ in range(k)
            ]

            rng = np.random.Generator(np.random.PCG64(seed))

            for t in tqdm(range(0, n_tokens, act_batch)):
                if fflags.record_timing and random.randint(0, 1000) == 0:
                    start = time_ns()

                    acts = next(act_generator)

                    if t + act_batch >= n_tokens:
                        acts = acts[: n_tokens - t]

                    end = time_ns()
                    timing_data["act_gen"] = np.append(
                        timing_data["act_gen"], (end - start) // act_batch
                    )
                    start = time_ns()

                    for i in range(acts.shape[0]):
                        bins[rng.integers(k)].append(acts[i])

                    end = time_ns()
                    timing_data["bin_append"] = np.append(
                        timing_data["bin_append"], (end - start) // act_batch
                    )
                else:
                    acts = next(act_generator)

                    if t + act_batch >= n_tokens:
                        acts = acts[: n_tokens - t]

                    for i in range(acts.shape[0]):
                        bins[rng.integers(k)].append(acts[i])

            if fflags.log_final_bin_shapes:
                for bin in bins:
                    data = bin.get_data()
                    print(data.shape)
                    print(len(data))
                    print("~~~~~")

            with h5py.File(f"{output_dir}/acts.h5", "w") as f:
                for l in layers:
                    layer_group = f.create_group(f"layer_{l}")
                    for s in sites:
                        layer_group.create_dataset(
                            s, shape=(n_tokens, n_dim), dtype=dtype
                        )

                curr_len = 0
                for i, bin in enumerate(bins):
                    data = bin.get_data()

                    record_timing = fflags.record_timing and random.randint(0, 10) == 0

                    if record_timing:
                        start = time_ns()
                        rng.shuffle(data)
                        end = time_ns()
                        timing_data["bin_shuffle"] = np.append(
                            timing_data["bin_shuffle"], end - start
                        )
                        start = time_ns()
                    else:
                        rng.shuffle(data)

                    for j, l in enumerate(layers):
                        for k, s in enumerate(sites):
                            f[f"layer_{l}"][s][curr_len : curr_len + data.shape[0]] = (
                                data[:, j, k]
                            )

                    if record_timing:
                        end = time_ns()
                        timing_data["bin_write"] = np.append(
                            timing_data["bin_write"], end - start
                        )

                    curr_len += data.shape[0]
                    bin.close()
                    del data

                if fflags.log_info:
                    print(f"Saved shuffled activations to {output_dir}/acts.h5")
    finally:
        for bin in bins:
            bin.close()

        if fflags.set_file_handle_limits:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

        if fflags.record_timing:
            return timing_data


if __name__ == "__main__":
    load_dotenv()

    n_tokens = int(2e9)
    layers = [1, 2, 3]
    sites = ["hook_mlp_out", "hook_attn_out"]
    d_mlp = 768
    dtype = "float16"
    max_bytes = 32 * (1024**3)  # 32 GiB
    p_overflow = 1e-12

    model_name = "gpt2"
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-v1"
    dataset_split = "train"

    model_batch = 16
    act_batch = 512

    device = auto_device()

    output_dir = os.getenv("OUTPUT_DIR", ".")

    dataset = datasets.load_dataset(dataset_name, dataset_config, split=dataset_split)[
        "text"
    ]

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
