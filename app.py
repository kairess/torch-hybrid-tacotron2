import gradio as gr
import torch
import numpy as np
import soundfile as sf

from model.model_inference_v2 import Tacotron2
from vocoder.model.waveglow import WaveGlow
from vocoder.denoiser_librosa import Denoiser
from korean_text.korean_cleaner_cls import KoreanCleaner

from text import text_to_sequence
from utils.util import to_var

device = 'cpu' # cuda

# Tacotron2
ckpt_dict = torch.load('logs/model/acoustic.ckpt', map_location=torch.device(device))
model = Tacotron2()
model.load_state_dict(ckpt_dict['model'])
model = model.eval()

# Vocoder
ckpt_dict = torch.load('logs/model/vocoder.ckpt', map_location=torch.device(device))
vocoder = WaveGlow()
vocoder.load_state_dict(ckpt_dict['model'])
vocoder = vocoder.remove_weightnorm(vocoder)
vocoder.eval()
denoiser = Denoiser(vocoder, 0.1)

korean_cleaner = KoreanCleaner()

def inference(text):
    text = korean_cleaner.clean_text(text)

    sequence = text_to_sequence(text, ['multi_cleaner'])
    sequence = to_var(torch.IntTensor(sequence)[None, :]).long()

    sigma = 0.5
    strength = 10
    sample_rate = 22050

    with torch.no_grad():
        _, mel_outputs_postnet, linear_outputs, _, alignments = model.inference(sequence)
        wav = vocoder.infer(mel_outputs_postnet, sigma=sigma)

        wav *= 32767. / max(0.01, torch.max(torch.abs(wav)))
        wav = wav.squeeze()
        wav = wav.cpu().detach().numpy().astype('float32')

        wav = denoiser(wav, strength=strength)

    wav = np.append(wav, np.array([[0.0] * (sample_rate // 2)]))

    wav_file = wav.astype(np.int16)
    sf.write('temp.wav', wav_file, sample_rate)

    return 'temp.wav'

demo = gr.Interface(fn=inference, inputs="text", outputs="audio")
demo.launch()
