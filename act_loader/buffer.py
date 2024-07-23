import torch
import datasets
from transformer_lens import HookedTransformer
from tqdm.autonotebook import tqdm
from multiprocessing import Pool
import gc

from act_loader import utils


class ActBuffer:
    """
    :param model_name: the hf model name
    :param layers: which layers to get activations from, passed as a list of ints
    :param dataset_name: the name of the hf dataset to use
    :param act_site: the tl key to get activations from
    :param dataset_split: the split of the dataset to use
    :param dataset_config: the config to use when loading the dataset
    :param buffer_size: the size of the buffer, in number of activations
    :param min_capacity: the minimum guaranteed capacity of the buffer, in number of activations, used to determine
    when to refresh the buffer
    :param model_batch_size: the batch size to use in the model when generating activations
    :param samples_per_seq: the number of activations to randomly sample from each sequence. If None, all
    activations will be used
    :param max_seq_length: the maximum sequence length to use when generating activations. If None, the sequences
    will not be truncated
    :param act_size: the size of the activations vectors. If None, it will guess the size from the model's cfg
    :param shuffle_buffer: if True, the buffer will be shuffled after each refresh
    :param seed: the seed to use for dataset shuffling and activation sampling
    :param device: the device to use for the model
    :param dtype: the dtype to use for the buffer and model
    :param buffer_device: the device to use for the buffer. If None, it will use the same device as the model
    :param offload_device: the device to offload the model to when not generating activations. If None, offloading
    is disabled. If using this, make sure to use a large enough buffer to avoid frequent offloading
    :param refresh_progress: If True, a progress bar will be displayed when refreshing the buffer

    A data buffer to store MLP activations for training the autoencoder.
    """
    def __init__(
        self,
        model,
        layers,
        dataset,
        act_site,
        buffer_size=256,
        min_capacity=128,
        model_batch_size=8,
        samples_per_seq=None,
        max_seq_length=None,
        act_size=None,
        shuffle_buffer=False,
        seed=None,
        device="cuda",
        dtype=torch.bfloat16,
        buffer_device=None,
        offload_device=None,
        refresh_progress=False,
        hf_model=None
    ):

        self.layers = layers
        self.act_names = [f"blocks.{layer}.{act_site}" for layer in layers]  # the tl keys to grab activations from todo
        self.buffer_size = buffer_size
        self.min_capacity = min_capacity
        self.model_batch_size = model_batch_size
        self.samples_per_seq = samples_per_seq
        self.max_seq_length = max_seq_length
        self.act_size = act_size
        self.shuffle_buffer = shuffle_buffer
        self.device = device
        self.dtype = dtype
        self.buffer_device = buffer_device or device
        self.offload_device = offload_device
        self.refresh_progress = refresh_progress
        self.final_layer = max(layers)  # the final layer that needs to be run

        assert isinstance(layers, list) and len(layers) > 0, "layers must be a non-empty list of ints"

        if seed:
            torch.manual_seed(seed)

        # pointer to the current position in the dataset
        self.dataset_pointer = 0

        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=model_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=8
        )
        self.data_generator = iter(self.data_loader)

        # load the model into a HookedTransformer
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name=model,
            hf_model=hf_model,
            device=device,
            dtype=dtype
        )

        # if the act_size is not provided, use the size from the model's cfg
        if act_size is None:
            if act_site[:3] == "mlp":
                self.act_size = self.model.cfg.d_mlp
            elif act_site == "hook_mlp_out":
                self.act_size = self.model.cfg.d_model
            else:
                raise ValueError(f"Cannot determine act_size from act_site {act_site}, please provide it manually")

        # if the buffer is on the cpu, pin it to memory for faster transfer to the gpu
        pin_memory = buffer_device == "cpu"

        # the buffer to store activations in, with shape (buffer_size, len(layers), act_size)
        self.buffer = torch.zeros(
            (buffer_size, len(self.layers), act_size),
            dtype=dtype,
            pin_memory=pin_memory,
            device=buffer_device
        )

        # pointer to read/write location in the buffer, reset to 0 after refresh is called
        # starts at buffer_size to be fully filled on first refresh
        self.buffer_pointer = self.buffer_size

        # initial buffer fill
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        """
        Whenever the buffer is refreshed, we remove the first `buffer_pointer` activations that were used, shift the
        remaining activations to the start of the buffer, and then fill the rest of the buffer with `buffer_pointer` new
        activations from the model.
        """

        # shift the remaining activations to the start of the buffer
        self.buffer = torch.roll(self.buffer, -self.buffer_pointer, 0)

        # if offloading is enabled, move the model to `device` before generating activations
        if self.offload_device:
            self.model.to(self.device)

        # start a progress bar if `refresh_progress` is enabled
        if self.refresh_progress:
            pbar = tqdm(total=self.buffer_pointer)
        else:
            pbar = None

        # fill the rest of the buffer with `buffer_pointer` new activations from the model
        while self.buffer_pointer > 0:
            # get the next batch of seqs
            try:
                seqs = next(self.data_generator)
            except StopIteration:
                print("Data generator exhausted, resetting...")
                self.reset_dataset()
                seqs = next(self.data_generator)

            if self.max_seq_length:
                with Pool(8) as p:
                    seqs = p.starmap(utils.truncate_seq, [(seq, self.max_seq_length) for seq in seqs])

            # run the seqs through the model to get the activations
            out, cache = self.model.run_with_cache(seqs, stop_at_layer=self.final_layer + 1,
                                                   names_filter=self.act_names)

            # clean up logits in order to free the graph memory
            del out
            torch.cuda.empty_cache()

            # store the activations in the buffer
            acts = torch.stack([cache[name] for name in self.act_names], dim=-2)
            # (batch, pos, layers, act_size) -> (batch*samples_per_seq, layers, act_size)
            if self.samples_per_seq:
                acts = acts[:, torch.randperm(acts.shape[-3])[:self.samples_per_seq]].flatten(0, 1)
            else:
                acts = acts.flatten(0, 1)

            write_pointer = self.buffer_size - self.buffer_pointer

            new_acts = min(acts.shape[0], self.buffer_pointer)  # the number of acts to write, capped by buffer_pointer
            self.buffer[write_pointer:write_pointer + acts.shape[0]].copy_(acts[:new_acts], non_blocking=True)
            del acts

            # update the buffer pointer by the number of activations we just added
            self.buffer_pointer -= new_acts

            # update the progress bar
            if pbar:
                pbar.update(new_acts)

        # close the progress bar
        if pbar:
            pbar.close()

        # sync the buffer to ensure async copies are complete
        torch.cuda.synchronize()

        # if shuffle_buffer is enabled, shuffle the buffer
        if self.shuffle_buffer:
            self.buffer = self.buffer[torch.randperm(self.buffer_size)]

        # if offloading is enabled, move the model back to `offload_device`, and clear the cache
        if self.offload_device:
            self.model.to(self.offload_device)
            torch.cuda.empty_cache()

        gc.collect()

        assert self.buffer_pointer == 0, "Buffer pointer should be 0 after refresh"

    @torch.no_grad()
    def next(self, batch: int = None):
        # if this batch read would take us below the min_capacity, refresh the buffer
        if self.will_refresh(batch):
            self.refresh()

        if batch is None:
            out = self.buffer[self.buffer_pointer]
        else:
            out = self.buffer[self.buffer_pointer:self.buffer_pointer + batch]

        self.buffer_pointer += batch or 1

        return out

    def reset_dataset(self):
        """
        Reset the buffer to the beginning of the dataset without reshuffling.
        """
        self.data_generator = iter(self.data_loader)

    def will_refresh(self, batch: int = None):
        return self.buffer_size - (self.buffer_pointer + (batch or 1)) < self.min_capacity
