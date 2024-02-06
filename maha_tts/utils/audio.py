import torch
import numpy as np
from scipy.signal import get_window
from scipy.io.wavfile import read

TACOTRON_MEL_MAX = 2.4
TACOTRON_MEL_MIN = -11.5130

def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def get_mask(lengths, max_len=None):
    """
    Generate a mask for sequences based on lengths.

    Parameters:
    - lengths: Torch tensor, lengths of sequences
    - max_len: Optional, maximum length for padding

    Returns:
    - Torch tensor, mask for sequences
    """
    max_len = max_len or torch.max(lengths).item()
    lens = torch.arange(max_len)
    mask = lens[:max_len].unsqueeze(0) < lengths.unsqueeze(1)
    return mask

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    Perform dynamic range compression on input tensor.

    Parameters:
    - x: Torch tensor, input tensor
    - C: Compression factor
    - clip_val: Minimum value to clamp input tensor

    Returns:
    - Torch tensor, compressed tensor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression(x, C=1):
    """
    Perform dynamic range decompression on input tensor.

    Parameters:
    - x: Torch tensor, input tensor
    - C: Compression factor used for compression

    Returns:
    - Torch tensor, decompressed tensor
    """
    return torch.exp(x) / C

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    Compute the sum-square envelope of a window function at a given hop length.

    Parameters:
    - window: String, tuple, number, callable, or list-like; window specification
    - n_frames: Int, number of analysis frames
    - hop_length: Int, number of samples to advance between frames
    - win_length: Int, length of the window function
    - n_fft: Int, length of each analysis frame
    - dtype: Numpy data type of the output
    - norm: Normalization type for the window function

    Returns:
    - Numpy array, sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = np.square(librosa.util.normalize(win_sq, norm=norm))
    win_sq = librosa.util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def load_wav_to_torch(full_path):
    """
    Load WAV file into Torch tensor.

    Parameters:
    - full_path: String, path to the WAV file

    Returns:
    - Torch tensor, audio data
    - Int, sampling rate
    """
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data), sampling_rate

if __name__ == "__main__":
    lens = torch.tensor([2, 3, 7, 5, 4])
    mask = get_mask(lens)
    print(mask)
    print(mask.shape)
