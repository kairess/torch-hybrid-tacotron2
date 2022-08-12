import torch
import numpy as np
from vocoder.utils.audio import _stft, _istft
from vocoder.hparams import hparams as hps


class Denoiser(torch.nn.Module):
    def __init__(self, waveglow, power=0.1):
        super(Denoiser, self).__init__()
        if power > 0.0:
            noise_mel = torch.zeros((1, 80, 88), dtype=waveglow.upsample.weight.dtype, device=waveglow.upsample.weight.device)
            noise_mel[0, :, :] = power

        else:
            noise_mel = torch.zeros((1, 80, 88), dtype=waveglow.upsample.weight.dtype, device=waveglow.upsample.weight.device)

        with torch.no_grad():
            noise_audio = waveglow.infer(noise_mel, sigma=0.0).float()
            noise_audio = noise_audio * hps.max_wav_value / max(0.01, torch.max(torch.abs(noise_audio)))

        noise_audio = noise_audio.squeeze()
        noise_audio = noise_audio.cpu().detach().numpy().astype('float32')
        noise_magnitude = np.abs(_stft(noise_audio))

        self.noise_magnitude = noise_magnitude[:, 0][:, None]

    def forward(self, audio, strength=10):
        D = _stft(audio)
        magnitude = np.abs(D)
        phase = np.exp(1j * np.angle(D))
        denoised_magnitude = magnitude - self.noise_magnitude * strength
        denoised_magnitude = np.clip(denoised_magnitude, 0, np.max(np.abs(denoised_magnitude)))
        denoised_audio = _istft(denoised_magnitude * phase)
        return denoised_audio