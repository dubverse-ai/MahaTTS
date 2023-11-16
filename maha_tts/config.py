import os
class config:
  
    semantic_model_centroids = 10000 + 1
    seed_value = 3407

    # Text to Semantic
    t2s_position = 2048

    # Semantic to acoustic
    sa_timesteps_max = 1000

    #Acoustic Properties
    CLIP_LENGTH = 500
    MAX_WAV_VALUE=32768.0
    filter_length=1024
    hop_length=256 #256
    window = 'hann'
    win_length=1024
    n_mel_channels=80
    sampling_rate=22050
    mel_fmin=0.0
    mel_fmax=8000.0