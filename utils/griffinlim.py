import torch
from torchaudio import functional as F
import math
from hparams import hparams as hps
from utils.util import to_arr


# def griffin_lim(spectrogram):
# 	denormalize_spec = _denormalize_torch(spectrogram)
# 	amplitude = _db_to_amp_torch(denormalize_spec + hps.ref_level_db)
# 	# S = torch.pow(amplitude, hps.power)
# 	S = torch.pow(amplitude, 1.2)
# 	waveform = to_arr(_griffin_lim_torch(S)[0])
# 	return waveform

def griffin_lim(spectrogram):
	magnitude = (spectrogram * hps.max_level_db) - hps.max_level_db + hps.ref_level_db
	magnitude = _db_to_amp_torch(magnitude)
	magnitude = torch.pow(magnitude, 1.2)
	waveform = to_arr(_griffin_lim_torch(magnitude)[0])
	return waveform


def _griffin_lim_torch(S):
	spectrogram = S.unsqueeze(0)
	n_fft = hps.n_fft
	hop_length = hps.hop_length
	win_length = hps.win_length

	shape = spectrogram.size()
	spectrogram = spectrogram.view([-1] + list(shape[-2:]))

	batch, freq, frames = spectrogram.size()

	rand_init = True

	if rand_init:
		angles = 2 * math.pi * torch.rand(batch, freq, frames)
	else:
		angles = torch.zeros(batch, freq, frames)


	angles = torch.stack([angles.cos(), angles.sin()], dim=-1).to(dtype=spectrogram.dtype, device=spectrogram.device)
	spectrogram = spectrogram.unsqueeze(-1).expand_as(angles)

	rebuilt = torch.tensor(0.)

	momentum = 0.99
	momentum = momentum / (1 + momentum)
	window = torch.hann_window(win_length).to(device=spectrogram.device)

	for _ in range(hps.gl_iters):
		tprev = rebuilt

		inverse = F.istft(spectrogram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).float()

		rebuilt = torch.stft(inverse, n_fft, hop_length, win_length, window, True, 'reflect', False, True)

		angles = rebuilt - tprev.mul_(momentum / (1 + momentum))
		angles = angles.div_(F.complex_norm(angles).add_(1e-16).unsqueeze(-1).expand_as(angles))


	waveform = F.istft(spectrogram * angles, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window).float()
	waveform = waveform.view(shape[:-2] + waveform.shape[-1:])

	return waveform


def _db_to_amp_torch(x):
	ones = torch.ones(x.size()).to(dtype=x.dtype, device=x.device)
	return torch.pow(ones * 10.0, x * 0.05)


def _denormalize_torch(spectrogram):
	return (torch.clamp(spectrogram, min=0, max=1) * -hps.min_level_db) + hps.min_level_db
