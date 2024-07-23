import pandas as pd
import time
from tqdm.autonotebook import tqdm

from act_loader.loader import ActLoader

MODELS = [
    'gpt2',

]

OPTIONS = {
    'batch_size': [1, 16, 256, 4096],
    'cached': [True, False],
    'shuffling': [True, False],
}


def profile_loader_timing(models, options, num_acts=10000):
    configs = {}
    for batch_size in options['batch_size']:
        for cached in options['cached']:
            for shuffling in options['shuffling']:
                cfg_name = f"{'shuffled' if shuffling else 'unshuffled'}_{'cached' if cached else 'uncached'}_{batch_size}x"

                configs[cfg_name] = (batch_size, cached, shuffling)

    times = pd.DataFrame(index=models, columns=list(configs.keys()))

    for model in models:
        for cfg_name, (batch_size, cache, shuffling) in configs.items():
            print(f"Profiling {model} with config {cfg_name}")
            loader = ActLoader(
                model,
                dataset='wikitext-2',
                layer=0,
                site='mlp_out',
                batch_size=batch_size,
                lazy=True,
                shuffling=shuffling,
                cache=cache,
            )

            start = time.time()
            for _ in tqdm(range(num_acts)):
                loader.next()
            end = time.time()

            times.loc[model, cfg_name] = end - start

    return times


def profile_loader_memory(models, configs):
    pass


if __name__ == '__main__':
    print(f"Models: {MODELS}")
    print(f"Options: {OPTIONS}")
    times = profile_loader_timing(MODELS, OPTIONS)
    print(times.to_string())
