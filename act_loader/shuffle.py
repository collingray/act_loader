import math
import tempfile

import datasets
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from act_loader.mmap_tensor import MemoryMappedTensor
from act_loader.utils import k_bins, TORCH_DTYPES


@torch.no_grad()
def tl_generate_acts(
    model: str,
    dataset_name: str,
    layer: int | list[int],
    site: str | list[str],
    batch_size: int,
    dtype: str,
    device: str,
):
    """
    Generate activations for a given model, dataset, and layer/site combination

    :param model: The model to generate activations for
    :param dataset_name: The hf dataset to generate activations from
    :param layer: The layer(s) to generate activations from
    :param site: The site(s) to generate activations from
    :param batch_size: The batch size to use when generating activations
    :param dtype: The data type to use for activations
    :param device: The device to generate activations on
    :return: An iterator over activations
    """

    assert isinstance(layer, int) or (
        isinstance(layer, list) and len(layer) > 0
    ), "'layer' must be an int or non-empty list of ints"
    assert dtype in TORCH_DTYPES, f"'dtype' must be one of {TORCH_DTYPES}"

    layers = [layer] if isinstance(layer, int) else layer
    sites = [site] if isinstance(site, str) else site

    model = HookedTransformer.from_pretrained_no_processing(
        model_name=model, device=device, dtype=dtype
    )

    dataset = datasets.load_dataset(dataset_name)["train"]["text"]

    final_layer = max(layers)
    act_names = [f"blocks.{l}.{s}" for l in layers for s in sites]

    # while True:
    #     yield torch.rand(len(layers), len(sites), 768, dtype=TORCH_DTYPES[dtype])

    for batch in DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8
    ):
        out, cache = model.run_with_cache(
            batch, stop_at_layer=final_layer + 1, names_filter=act_names
        )

        del out

        acts = torch.stack(
            [  # [n_layers, n_sites, batch_size, seq_len, d_model]
                torch.stack([cache[f"blocks.{l}.{s}"] for s in sites]) for l in layers
            ]
        )

        del cache

        acts = acts.permute(
            2, 3, 0, 1, 4
        )  # [batch_size, seq_len, n_layers, n_sites, d_model]
        acts = acts.flatten(0, 1)  # [batch_size * seq_len, n_layers, n_sites, d_model]

        for i in range(acts.shape[0]):
            yield acts[i]


def shuffle_acts(
    act_generator,
    n_tokens,
    out_name,
    max_bytes,
    layers: int | list[int],
    sites: str | list[str],
    n_dim: int,
    dtype: str,
    p_overflow=1e-12,
    seed=None,
):

    layers = [layers] if isinstance(layers, int) else layers
    sites = [sites] if isinstance(sites, str) else sites

    shape = (len(layers), len(sites), n_dim)

    # Number of token's worth of activations that can fit in memory
    m = max_bytes // (math.prod(shape) * TORCH_DTYPES[dtype].itemsize)

    print(f"n_tokens: {n_tokens}, m: {m}, p_overflow: {p_overflow}")

    # Number of bins to split the activations into, with a `p_overflow` chance that one bin will be greater than `m`
    k = k_bins(n_tokens, m, p_overflow)

    with tempfile.TemporaryDirectory() as tmp_dir:
        bins = [MemoryMappedTensor(tmp_dir, m, shape, dtype) for i in range(k)]

        print(f"Splitting activations into {k} bins...")

        rng = np.random.Generator(np.random.PCG64(seed))

        for i in tqdm(range(n_tokens)):
            acts = next(act_generator)
            bins[rng.integers(k)].append(acts)

        with h5py.File(f"{out_name}.h5", "w") as f:
            for l in layers:
                layer_group = f.create_group(f"layer_{l}")
                for s in sites:
                    layer_group.create_dataset(s, shape=(n_tokens, n_dim), dtype=dtype)

            curr_len = 0
            for i, bin in enumerate(bins):
                data = bin.get_data()
                rng.shuffle(data)

                for j, l in enumerate(layers):
                    for k, s in enumerate(sites):
                        f[f"layer_{l}"][s][curr_len : curr_len + data.shape[0]] = data[
                            :, j, k
                        ]

                curr_len += data.shape[0]
                bin.close()
                del data

            print(f"Saved shuffled activations to {out_name}.h5")


if __name__ == "__main__":
    n_tokens = 2e11
    layers = [1, 2, 3]
    sites = ["mlp_in", "mlp_out"]
    d_mlp = 768
    dtype = torch.float16
    max_bytes = 32 * (1024**3)  # 32 GB
    shape = (len(layers), len(sites), d_mlp)
