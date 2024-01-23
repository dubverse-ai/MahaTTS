import torch
import numpy as np
from scipy.signal import get_window
from scipy.io.wavfile import read

TACOTRON_MEL_MAX = 2.4
TACOTRON_MEL_MIN = -11.5130


def normalize_and_denormalize_tacotron_mel(mel, denormalize=True):
    if denormalize:
        return ((mel + 1) / 2) * (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN) + TACOTRON_MEL_MIN
    else:
        return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def get_mask_from_lengths(lengths, max_len=None):
    max_len = max_len or torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_mask(lengths, max_len=None):
    max_len = max_len or torch.max(lengths).item()
    lens = torch.arange(max_len)
    mask = lens[:max_len].unsqueeze(0) < lengths.unsqueeze(1)
    return mask


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    return torch.exp(x) / C


def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = np.square(np.pad(win_sq, (0, n_fft - len(win_sq))))

    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data), sampling_rate


if __name__ == "__main__":
    lens = torch.tensor([2, 3, 7, 5, 4])
    mask = get_mask(lens)
    print(mask)
    print(mask.shape)
