from typing import List, Optional

from torch.utils.data import Dataset, DataLoader

import utils


class ActLoader:
    def __init__(
        self,
        model: str,
        dataset: Dataset,
        layer: int | List[int],
        site: str | List[str],
        shuffling: bool = True,
        batch_size: int = 1,
        primary_device: str = "cuda",
        offload_device: Optional[str] = "cpu",
        offload_model: bool = False,
        dtype: str = "float32",
        lazy: bool = False,
        cache: bool = False,
        seed: int = 42,
    ):
        """

        :param model:
        :param dataset:
        :param layer:
        :param site:
        :param shuffling:
        :param batch_size:
        :param primary_device: The primary device to generate and return activations on
        :param offload_device: The secondary device to store activations on
        :param offload_model: If enabled, the model will be offloaded to the offload_device when not in use
        :param dtype: The datatype to use for activations
        :param lazy: If enabled, activations will not be loaded until the first call to `next`
        :param cache: If enabled, activations will be cached to disk after being loaded
        :param seed: The seed to use for shuffling
        """

        assert isinstance(layer, int) or (
            isinstance(layer, list) and len(layer) > 0
        ), "'layer' must be an int or non-empty list of ints"
        assert (
            dtype in utils.TORCH_DTYPES
        ), f"'dtype' must be one of {utils.TORCH_DTYPES}"

        self.model = model
        self.dataset = dataset
        self.layer = layer
        self.site = site
        self.shuffling = shuffling
        self.batch_size = batch_size
        self.primary_device = primary_device
        self.offload_device = offload_device
        self.offload_model = offload_model
        self.cache = cache
        self.seed = seed

    def next(self): ...
