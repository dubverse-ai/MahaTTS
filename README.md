<div align="center">

<h1>MahaTTS: An Open-Source Large Speech Generation Model in the making</h1>
a <a href = "https://black.dubverse.ai">Dubverse Black</a> initiative <br> <br>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-eOQqznKWwAfMdusJ_LDtDhjIyAlSMrG?usp=sharing)
[![Discord Shield](https://discordapp.com/api/guilds/1162007551987171410/widget.png?style=shield)](https://discord.gg/4VGnrgpBN)
</div>

------

## Community

Join our [Discord Community](https://discord.gg/4VGnrgpBN) and select the role `Github Guru` to get early access to upcoming model releases. We have two major model releases planned before March 2024.  

## Description

MahaTTS, with Maha signifying 'Great' in Sanskrit, is a Text to Speech Model developed by [Dubverse.ai](https://dubverse.ai). We drew inspiration from the [Tortoise TTS](https://github.com/neonbjb/tortoise-tts) model, but our model uniquely utilizes seamless M4t wav2vec2 for semantic token extraction. As this specific variant of wav2vec2 is trained on multilingual data, it enhances our model's scalability across different languages.

We are providing access to pretrained model checkpoints, which are ready for inference and available for commercial use.

<img width="993" alt="MahaTTS Architecture" src="https://github.com/dubverse-ai/MahaTTS/assets/32906806/7429d3b6-3f19-4bd8-9005-ff9e16a698f8">

## Updates

**13-11-2023**

- MahaTTS Open Sourced!

**7-01-2024**

- Smolie English (`smolie-en`) and Smolie Indic (`smolie-in`) released!

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
!gdown --folder 1-HEc3V4f6X93I8_IfqExLfL3s8I_dXGZ -q #download speakers ref files

import torch, glob
from maha_tts import load_models,infer_tts,config
from scipy.io.wavfile import write
from IPython.display import Audio,display

# PATH TO THE SPEAKERS WAV FILES
speaker =['/content/infer_ref_wavs/2272_152282_000019_000001/',
          '/content/infer_ref_wavs/2971_4275_000049_000000/',
          '/content/infer_ref_wavs/4807_26852_000062_000000/',
          '/content/infer_ref_wavs/6518_66470_000014_000002/']
```

### Inferring `smolie-en`, the English Model
```bash
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-en',device)
print('Using:',device)

speaker_num = 0 # @param ["0", "1", "2", "3"] {type:"raw"}
text = "I freakin love how Elon came to life the moment they started talking about gaming and specifically diablo, you can tell that he didn't want that part of the discussion to end, while Lex to move on to the next subject! Once a true gamer, always a true gamer!" # @param {type:"string"}

ref_clips = glob.glob(speaker[speaker_num]+'*.wav')
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder)

write('/content/test.wav',sr,audio)
```

### Inferring `smolie-in`, the Indic Multilingual Model
```bash
# SMOLIE-IN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-in',device)
print('Using:',device)

speaker_num = 0 # @param ["0", "1", "2", "3"] {type:"raw"}
text = "शाम के समय, आसमान में बिखरी हुई रंग-बिरंगी रौशनी से सजा हुआ नगर दृश्य एक रोमांटिक माहौल बना रहा था।" # @param {type:"string"}

langauge = 'hindi' # ['hindi','english','tamil', 'telugu', 'punjabi', 'marathi', 'gujarati', 'bengali', 'assamese']
language = torch.tensor(config.lang_index[langauge]).to(device).unsqueeze(0)

ref_clips = glob.glob(speaker[speaker_num]+'*.wav')
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder,language)

write('/content/test.wav',sr,audio)
```

## Roadmap
- [x] Smolie English (`smolie-en`): Trained on 9k hours of English Podcast data
- [x] Smolie Indic (`smolie-in`): Trained on 400 hour of IIT Madras audio data across 9 Indian languages
- [ ] Smolie Indic Scaled (coming soon!)
- [ ] Optimizations for inference (looking for contributors, check issues)

## Some Generated Samples
0 -> "I seriously laughed so much hahahaha (seals with headphones...) and appreciate both the interviewer and the subject. Major respect for two extraordinary humans - and in this time of gratefulness, I'm thankful for you both and this forum!"

1 -> "I freakin love how Elon came to life the moment they started talking about gaming and specifically diablo, you can tell that he didn't want that part of the discussion to end, while Lex to move on to the next subject! Once a true gamer, always a true gamer!"

2 -> "hello there! how are you?" (This one didn't work well, M1 model hallucinated)

3 -> "Who doesn't love a good scary story, something to send a chill across your skin in the middle of summer's heat or really, any other time? And this year, we're celebrating the two hundredth birthday of one of the most famous scary stories of all time: Frankenstein."



https://github.com/dubverse-ai/MahaTTS/assets/32906806/462ee134-5d8c-43c8-a425-3b6cabd2ff85




https://github.com/dubverse-ai/MahaTTS/assets/32906806/40c62402-7f65-4a35-b739-d8b8a082ad62



https://github.com/dubverse-ai/MahaTTS/assets/32906806/f0a9628c-ef81-450d-ab82-2f4c4626864e



https://github.com/dubverse-ai/MahaTTS/assets/32906806/15476151-72ea-410d-bcdc-177433df7884


## Technical Details

### Model Params
|      Model (Smolie)       | Parameters | Model Type |       Output      |  
|:-------------------------:|:----------:|------------|:-----------------:|
|   Text to Semantic (M1)   |    84 M    | Causal LM  |   10,001 Tokens   |
|  Semantic to MelSpec(M2)  |    430 M   | Diffusion  |   2x 80x Melspec  |
|      Hifi Gan Vocoder     |    13 M    |    GAN     |   Audio Waveform  |

### Languages Supported
| Language | Status |
| --- | :---: |
| English (en) | ✅ |
| Hindi (in) | ✅ |
| Indian English (in) | ✅ |
| Bengali (in) | ✅ |
| Tamil (in) | ✅ |
| Telugu (in) | ✅ |
| Punjabi (in) | ✅ |
| Marathi (in) | ✅ |
| Gujarati (in) | ✅ |
| Assamese (in) | ✅ |

## License

MahaTTS is licensed under the Apache 2.0 License. 

## 🙏 Appreciation

- [tortoise-tts](https://github.com/neonbjb/tortoise-tts)
- [M4t Seamless](https://github.com/facebookresearch/seamless_communication) [AudioLM](https://arxiv.org/abs/2209.03143) and many other ground-breaking papers that enabled the development of MahaTTS
- [Diffusion training](https://github.com/openai/guided-diffusion) for training diffusion model
- [Huggingface](https://huggingface.co/docs/transformers/index) for related training and inference code
