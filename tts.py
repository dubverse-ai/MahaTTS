import torch,glob
from maha_tts import load_diffuser,load_models,infer_tts,config
from scipy.io.wavfile import write

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:',device)
text = 'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition.'
langauge = 'english'
language = torch.tensor(config.lang_index[langauge]).to(device).unsqueeze(0)
ref_clips = glob.glob('models/Smolie-en/ref_clips/part0_1_1/*.wav')
# print(len(ref_clips))

# diffuser = load_diffuser()
diff_model,ts_model,vocoder,diffuser = load_models('Smolie-in',device)
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder,language)
write('test.wav',sr,audio)