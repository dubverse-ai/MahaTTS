<div align="center">

<h1>MahaTTS: An Open-Source Large Speech Generation Model in the making</h1>
a <a href = "https://black.dubverse.ai">Dubverse Black</a> initiative <br> <br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-eOQqznKWwAfMdusJ_LDtDhjIyAlSMrG?usp=sharing)
[![Discord Shield](https://discordapp.com/api/guilds/1162007551987171410/widget.png?style=shield)](https://discord.gg/4VGnrgpBN)
</div>

------

## Description

MahaTTS, with Maha signifying 'Great' in Sanskrit, is a Text to Speech Model developed by [Dubverse.ai](https://dubverse.ai). We drew inspiration from the [tortoise-tts](https://github.com/neonbjb/tortoise-tts) model, but our model uniquely utilizes seamless M4t wav2vec2 for semantic token extraction. As this specific variant of wav2vec2 is trained on multilingual data, it enhances our model's scalability across different languages.

We are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.

<img width="993" alt="MahaTTS Architecture" src="https://github.com/dubverse-ai/MahaTTS/assets/32906806/7429d3b6-3f19-4bd8-9005-ff9e16a698f8">

## Updates

**2023-11-13**

- MahaTTS Released! Open sourced Smolie
- Community and access to new features on our [Discord](https://discord.gg/uFPrzBqyF2)

## Features

1. Multilinguality (coming soon)
2. Realistic Prosody and intonation
3. Multi-voice capabilities

## Installation

```bash
pip install git+https://github.com/dubverse-ai/MahaTTS.git
```

```bash
pip install maha-tts
```

## api usage

```bash
!gdown --folder 1-HEc3V4f6X93I8_IfqExLfL3s8I_dXGZ -q # download speakers ref files

import torch,glob
from maha_tts import load_models,infer_tts
from scipy.io.wavfile import write
from IPython.display import Audio,display

# PATH TO THE SPEAKERS WAV FILES
speaker =['/content/infer_ref_wavs/2272_152282_000019_000001/',
          '/content/infer_ref_wavs/2971_4275_000049_000000/',
          '/content/infer_ref_wavs/4807_26852_000062_000000/',
          '/content/infer_ref_wavs/6518_66470_000014_000002/']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie',device)
print('Using:',device)

speaker_num = 0 # @param ["0", "1", "2", "3"] {type:"raw"}
text = "I freakin love how Elon came to life the moment they started talking about gaming and specifically diablo, you can tell that he didn't want that part of the discussion to end, while Lex to move on to the next subject! Once a true gamer, always a true gamer!" # @param {type:"string"}

ref_clips = glob.glob(speaker[speaker_num]+'*.wav')
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder)

write('/content/test.wav',sr,audio)
```
## Roadmap
- [x] Smolie - eng (trained on 200 hours of LibriTTS)
- [ ] Smolie - indic (Train on Indian languages, estimated by 20th Dec)
- [ ] Optimizations for inference (looking for contributors, check issues)

## Some Generated Samples
0 -> "I seriously laughed so much hahahaha (seals with headphones...) and appreciate both the interviewer and the subject. Major respect for two extraordinary humans - and in this time of gratefulness, I'm thankful for you both and this forum!"

1 -> "I freakin love how Elon came to life the moment they started talking about gaming and specifically diablo, you can tell that he didn't want that part of the discussion to end, while Lex to move on to the next subject! Once a true gamer, always a true gamer!"

2 -> "hello there! how are you?" (This one didn't work well, M1 model hallucinated)

3 -> "Who doesn't love a good scary story, something to send a chill across your skin in the middle of summer's heat or really, any other time? And this year, we're celebrating the two hundredth birthday of one of the most famous scary stories of all time: Frankenstein."

https://github.com/dubverse-ai/MahaTTS/assets/32906806/ab6798c7-b749-44e7-93bd-d74fda065b1f



https://github.com/dubverse-ai/MahaTTS/assets/32906806/b9e39028-3ecb-48ad-99cd-e6e5822b2406




https://github.com/dubverse-ai/MahaTTS/assets/32906806/ad9c79e1-a206-4be2-aa8f-eda788a10d78



https://github.com/dubverse-ai/MahaTTS/assets/32906806/10e5925a-b26c-4bda-81c0-450a7906bc8b



## Technical Details

### Model Params
|      Model (Smolie)       | Parameters | Model Type |       Output      |  
|:-------------------------:|:----------:|------------|:-----------------:|
|   Text to Semantic (M1)   |    69 M    | Causal LM  |   10,001 Tokens   |
|  Semantic to MelSpec(M2)  |    108 M   | Diffusion  |   2x 80x Melspec  |
|      Hifi Gan Vocoder     |    13 M    |    GAN     |   Audio Waveform  |

### Languages Supported
| Language | Status |
| --- | :---: |
| English (en) | ✅ |

## License

MahaTTS is licensed under the Apache 2.0 License. 

## 🙏 Appreciation

- [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
- [M4t Seamless](https://github.com/facebookresearch/seamless_communication) [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of MahaTTS
- [Diffusion training](https://github.com/openai/guided-diffusion) for training diffusion model
- [Huggingface](https://huggingface.co/docs/transformers/index) for related training and inference code
