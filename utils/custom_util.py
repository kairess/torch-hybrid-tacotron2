import torch
import numpy as np
import math
from hparams import hparams as hps
from utils.util import mode, to_arr

def memoize(func):
    """
    memoize decorator
    """
    class Memodict(dict):
        """
        Memoization decorator for a function taking one or more arguments.
        """
        def __getitem__(self, *key):
            return dict.__getitem__(self, key)

        def __missing__(self, key):
            ret = self[key] = func(*key)
            return ret

    return Memodict().__getitem__


@memoize
def guided_attention(N, max_N, T, max_T, g, use_eos):
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n in range(int(N)):
        for t in range(int(T)):
            ti = t
            if use_eos and t > int(T) - 3:
                t = 0

            # W[n, ti] = 1 - np.exp(-(n / N - t / T) ** 2 / (2 * g * g))
            W[n, ti] = 1 - math.exp(-(n / N - t / T) ** 2 / (2 * g * g))

    return W


def guided_attentions(input_lengths, target_lengths, max_target_len, g=0.25, use_eos=False):
    input_lengths = to_arr(input_lengths)
    target_lengths = to_arr(target_lengths)
    B = len(input_lengths)
    max_input_len = int(input_lengths.max())
    W = np.zeros((B, max_target_len, max_input_len), dtype=np.float32)
    for b in range(B):
        target_length = target_lengths[b]
        if target_length % hps.n_frames_per_step != 0:
            target_length += hps.n_frames_per_step - target_length % hps.n_frames_per_step

        W[b] = guided_attention(input_lengths[b], max_input_len,
                                target_length // hps.n_frames_per_step, max_target_len, g, use_eos).T

    return mode(torch.from_numpy(W))
