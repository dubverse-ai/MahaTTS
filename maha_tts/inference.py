import torch,glob,os,requests
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import write
from scipy.special import softmax
from maha_tts.models.diff_model import load_diff_model
from maha_tts.models.autoregressive import load_TS_model
from maha_tts.models.vocoder import load_vocoder_model,infer_wav
from maha_tts.utils.audio import denormalize_tacotron_mel,normalize_tacotron_mel,load_wav_to_torch,dynamic_range_compression
from maha_tts.utils.stft import STFT
from maha_tts.utils.diffusion import SpacedDiffusion,get_named_beta_schedule,space_timesteps
from maha_tts.text.symbols import labels,text_labels,code_labels,text_enc,text_dec,code_enc,code_dec
from maha_tts.text.cleaners import  english_cleaners
from maha_tts.config import config

DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'maha_tts', 'models')

stft_fn = STFT(config.filter_length, config.hop_length, config.win_length)

mel_basis = librosa_mel_fn(
        sr=config.sampling_rate, n_fft=config.filter_length, n_mels=config.n_mel_channels, fmin=config.mel_fmin, fmax=config.mel_fmax)

mel_basis = torch.from_numpy(mel_basis).float()

model_dirs= {
    'Smolie':['https://huggingface.co/Dubverse/MahaTTS/resolve/main/maha_tts/pretrained_models/smolie/S2A/s2a_latest.pt',
                'https://huggingface.co/Dubverse/MahaTTS/resolve/main/maha_tts/pretrained_models/smolie/T2S/t2s_best.pt'],
    'hifigan':['https://huggingface.co/Dubverse/MahaTTS/resolve/main/maha_tts/pretrained_models/hifigan/g_02500000',
                'https://huggingface.co/Dubverse/MahaTTS/resolve/main/maha_tts/pretrained_models/hifigan/config.json']
}

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Check if the response was successful (status code 200)
    response.raise_for_status()

    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            # Write data to the file
            file.write(data)
            # Update the progress bar
            bar.update(len(data))

    print(f"Download complete: {filename}\n")

def download_model(name):
    print('Downloading ',name," ....")
    checkpoint_diff = os.path.join(DEFAULT_MODELS_DIR,name,'s2a_latest.pt')
    checkpoint_ts = os.path.join(DEFAULT_MODELS_DIR,name,'t2s_best.pt')
    checkpoint_voco = os.path.join(DEFAULT_MODELS_DIR,'hifigan','g_02500000')
    voco_config_path = os.path.join(DEFAULT_MODELS_DIR,'hifigan','config.json')

    os.makedirs(os.path.join(DEFAULT_MODELS_DIR,name),exist_ok=True)
        
    if name == 'hifigan':
        download_file(model_dirs[name][0],checkpoint_voco)
        download_file(model_dirs[name][1],voco_config_path)
    
    else:
        download_file(model_dirs[name][0],checkpoint_diff)
        download_file(model_dirs[name][1],checkpoint_ts)

def load_models(name,device=torch.device('cpu')):
    '''
    Load pre-trained models for different components of a text-to-speech system.

    Args:
    device (str): The target device for model loading (e.g., 'cpu' or 'cuda').
    checkpoint_diff (str): File path to the pre-trained model checkpoint for the diffusion model.
    checkpoint_ts (str): File path to the pre-trained model checkpoint for the text-to-semantic model.
    checkpoint_voco (str): File path to the pre-trained model checkpoint for the vocoder model.
    voco_config_path (str): File path to the configuration file for the vocoder model.

    Returns:
    diff_model (object): Loaded diffusion model for semantic-to-acoustic tokens.
    ts_model (object): Loaded text-to-semantic model for converting text-to-semantic tokens.
    vocoder (object): Loaded vocoder model for generating waveform from acoustic tokens.
    diffuser (object): Configured diffuser object for use in the diffusion model.
    '''

    assert name in model_dirs, "no model name "+name

    checkpoint_diff = os.path.join(DEFAULT_MODELS_DIR,name,'s2a_latest.pt')
    checkpoint_ts = os.path.join(DEFAULT_MODELS_DIR,name,'t2s_best.pt')
    checkpoint_voco = os.path.join(DEFAULT_MODELS_DIR,'hifigan','g_02500000')
    voco_config_path = os.path.join(DEFAULT_MODELS_DIR,'hifigan','config.json')
    
    # for i in [checkpoint_diff,checkpoint_ts,checkpoint_voco,voco_config_path]:
    if not os.path.exists(checkpoint_diff) or not os.path.exists(checkpoint_ts):
        download_model(name)
    
    if not os.path.exists(checkpoint_voco) or not os.path.exists(voco_config_path):
        download_model('hifigan')

    diff_model = load_diff_model(checkpoint_diff,device)
    ts_model = load_TS_model(checkpoint_ts,device)
    vocoder = load_vocoder_model(voco_config_path,checkpoint_voco,device)
    diffuser = load_diffuser()

    return diff_model,ts_model,vocoder,diffuser

def infer_mel(model,timeshape,code,ref_mel,diffuser,temperature=1.0):
    device = next(model.parameters()).device
    code = code.to(device)
    ref_mel =ref_mel.to(device)
    output_shape = (1,80,timeshape)
    noise = torch.randn(output_shape, device=code.device) * temperature
    mel = diffuser.p_sample_loop(model, output_shape, noise=noise,
                                      model_kwargs={'code_emb': code,'ref_clips':ref_mel},
                                     progress=True)
    return denormalize_tacotron_mel(mel)

def generate_semantic_tokens(
    text,
    model,
    ref_mels,
    temp = 0.7,
    top_p= None,
    top_k= None,
    n_tot_steps = 1000,
    device = None
    ):
    semb = []
    with torch.no_grad():
        for n in range(n_tot_steps):
            x = get_inputs(text,semb,ref_mels,device)
            _,result = model(**x)
            relevant_logits = result[0,:,-1]
            if top_p is not None:
                # faster to convert to numpy
                original_device = relevant_logits.device
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(original_device)

            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")

            probs = F.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
            semb.append(str(code_dec[item_next.item()]))
            if semb[-1] == '<EST>' or semb[-1] == '<PAD>':
                break

            del relevant_logits, probs, item_next

    semb = torch.tensor([int(i) for i in semb[:-1]])
    return semb,result

def get_inputs(text,semb=[],ref_mels=[],device=torch.device('cpu')):
  text = text.lower()
  text_ids=[text_enc['<S>']]+[text_enc[i] for i in text.strip()]+[text_enc['<E>']]
  semb_ids=[code_enc['<SST>']]+[code_enc[i] for i in semb]#+[tok_enc['<EST>']]

  input_ids = text_ids+semb_ids
  # pad_length = config.t2s_position-(len(text_ids)+len(semb_ids))

  token_type_ids = [0]*len(text_ids)+[1]*len(semb_ids)#+[0]*pad_length
  positional_ids = [i for i in range(len(text_ids))]+[i for i in range(len(semb_ids))]#+[0]*pad_length
  # labels = [-100]*len(text_ids)+semb_ids+[-100]*pad_length
  attention_mask = [1]*len(input_ids)#+[0]*pad_length
  # input_ids += [tok_enc['<PAD>']]*pad_length
  return {'text_ids':torch.tensor(text_ids).unsqueeze(0).to(device),'codes_ids':torch.tensor(semb_ids).unsqueeze(0).to(device),'ref_clips':normalize_tacotron_mel(ref_mels).to(device)}

def get_ref_mels(ref_clips):
    ref_mels = []
    for i in ref_clips:
        ref_mels.append(get_mel(i)[0][:,:500])
    
    ref_mels_padded = (torch.randn((len(ref_mels), 80, 500)))*1e-8
    for i,mel in enumerate(ref_mels):
        ref_mels_padded[i, :, :mel.size(1)] = mel
    return ref_mels_padded.unsqueeze(0)

def get_mel(filepath):
    audio, sampling_rate = load_wav_to_torch(filepath)
    audio_norm = audio / config.MAX_WAV_VALUE
    audio_norm = audio_norm.unsqueeze(0)
    y = torch.autograd.Variable(audio_norm, requires_grad=False)

    assert(torch.min(y.data) >= -1)
    assert(torch.max(y.data) <= 1)
    magnitudes, phases = stft_fn.transform(y)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_output = dynamic_range_compression(mel_output)
    melspec = torch.squeeze(mel_output, 0)
    energy = torch.norm(magnitudes, dim=1).squeeze(0)
    return melspec,list(energy)

def infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder):
    '''
    Generate audio from the given text using a text-to-speech (TTS) pipeline.

    Args:
    text (str): The input text to be synthesized into speech.
    ref_clips (list): A list of paths to reference audio clips, preferably more than 3 clips.
    diffuser (object): A diffusion object used for denoising and guidance in the diffusion model. It should be obtained using load_diffuser.
    diff_model: diffusion model for semantic-to-acoustic tokens.
    ts_model: text-to-semantic model for converting text-to-semantic tokens.
    vocoder: vocoder model for generating waveform from acoustic tokens.

    Returns:
    audio (numpy.ndarray): Generated audio waveform.
    sampling_rate (int): Sampling rate of the generated audio.

    Description:
    The `infer_tts` function takes input text and reference audio clips, and processes them through a TTS pipeline.
    It first performs text preprocessing and generates semantic tokens using the specified text synthesis model.
    Then, it infers mel-spectrogram features using the diffusion model and the provided diffuser.
    Finally, it generates audio from the mel-spectrogram using the vocoder.

    Note: The function requires properly configured diff_model, ts_model, and vocoder objects for successful TTS.

    Example usage:
    audio, sampling_rate = infer_tts("Hello, how are you?", ref_clips, diffuser, diff_model, ts_model, vocoder)
    '''
    device = next(ts_model.parameters()).device
    text = english_cleaners(text)
    ref_mels = get_ref_mels(ref_clips)
    with torch.no_grad():
        sem_tok,_ = generate_semantic_tokens(
                        text,
                        ts_model,
                        ref_mels,
                        temp = 0.7,
                        top_p= 0.8,
                        top_k= 5,
                        n_tot_steps = 1000,
                        device = device
                    )
        mel = infer_mel(diff_model,int(((sem_tok.shape[-1] * 320 / 16000) * 22050/256)+1),sem_tok.unsqueeze(0) + 1,
                        ref_mels,diffuser,temperature=1.0)

        audio = infer_wav(mel,vocoder)
    
    return audio,config.sampling_rate

def load_diffuser(timesteps = 100, gudiance=3):
    '''
    Load and configure a diffuser for denoising and guidance in the diffusion model.

    Args:
    timesteps (int): Number of denoising steps out of 1000. Default is 100.
    guidance (int): Conditioning-free guidance parameter. Default is 3.

    Returns:
    diffuser (object): Configured diffuser object for use in the diffusion model.

    Description:
    The `load_diffuser` function initializes a diffuser with specific settings for denoising and guidance.
    '''
    betas = get_named_beta_schedule('cosine',config.sa_timesteps_max)
    diffuser = SpacedDiffusion(use_timesteps=space_timesteps(1000, [timesteps]), model_mean_type='epsilon',
                        model_var_type='learned_range', loss_type='rescaled_mse', betas=betas,
                        conditioning_free=True, conditioning_free_k=gudiance)
    diffuser.training=False
    return diffuser

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    text = 'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition.'
    ref_clips = glob.glob('/Users/jaskaransingh/Desktop/maha_tts/ref_clips/*.wav')

    checkpoint_diff = 'maha_tts/pretrained_models/S2A/s2a_latest.pt'
    checkpoint_ts = 'maha_tts/pretrained_models/T2S/t2s_best.pt'
    checkpoint_voco = 'maha_tts/pretrained_models/hifigan/g_02500000'
    voco_config_path = 'maha_tts/pretrained_models/hifigan/config.json'

    diffuser = load_diffuser()
    diff_model,ts_model,vocoder = load_models(device,checkpoint_diff,checkpoint_ts,checkpoint_voco,voco_config_path)
    audio,sr = infer_tts(text,ref_clips,diffuser,diff_model,ts_model,vocoder)
    write('test.wav',sr,audio)

    