import torch,glob
from maha_tts import load_diffuser,load_models,infer_tts
from scipy.io.wavfile import write

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using:',device)
text = 'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition.'
ref_clips = glob.glob('/Users/jaskaransingh/Desktop/NeuralSpeak/ref_clips/*.wav')
# print(len(ref_clips))

# diffuser = load_diffuser()
diff_model,ts_model,vocoder,diffuser = load_models('Smolie',device)
audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder)
write('test.wav',sr,audio)