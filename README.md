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

<img width="1251" alt="Screenshot 2023-11-23 at 1 49 45 PM" src="https://github.com/dubverse-ai/MahaTTS/assets/32906806/9992b5a9-1ca7-4e64-9c27-6244e9fd4b15">

## Roadmap
- [x] Smolie - eng (trained on 200 hours of LibriTTS)
- [ ] Smolie - indic (Train on Indian languages, estimated by 20th Dec)
- [ ] Optimizations for inference (looking for contributors, check issues)

## Some Generated Samples
0 -> "I seriously laughed so much hahahaha (seals with headphones...) and appreciate both the interviewer and the subject. Major respect for two extraordinary humans - and in this time of gratefulness, I'm thankful for you both and this forum!"

1 -> "I freakin love how Elon came to life the moment they started talking about gaming and specifically diablo, you can tell that he didn't want that part of the discussion to end, while Lex to move on to the next subject! Once a true gamer, always a true gamer!"

2 -> "hello there! how are you?" (This one didn't work well, M1 model hallucinated)

3 -> "Who doesn't love a good scary story, something to send a chill across your skin in the middle of summer's heat or really, any other time? And this year, we're celebrating the two hundredth birthday of one of the most famous scary stories of all time: Frankenstein."

https://github.com/dubverse-ai/MahaTTS/assets/32906806/66fc7a08-3e8a-4d63-a3fa-88bc705a172a

https://github.com/dubverse-ai/MahaTTS/assets/32906806/5acf5a4b-aeb8-4f14-94fe-45811868a886

https://github.com/dubverse-ai/MahaTTS/assets/32906806/0af2ce6e-4172-4aac-9322-4fd545f1d4ac

https://github.com/dubverse-ai/MahaTTS/assets/32906806/2d5b0335-d1fc-473a-aea8-c5bb6afbce27

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
- 
