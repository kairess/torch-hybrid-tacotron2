import librosa
import numpy as np
from scipy.io import wavfile
from hparams import hparams as hps

def load_wav(path):
    wav, sr = librosa.load(path, sr=hps.sample_rate)
    wav = wav / np.abs(wav).max() * 0.999

    try:
        assert sr == hps.sample_rate
    except:
        print('Error:', path, 'has wrong sample rate')

    return wav

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hps.sample_rate, wav.astype(np.int16))


def melspectrogram(y):
    linear = _stft(y)
    magnitude = np.abs(linear)

    mel_basis = librosa.filters.mel(hps.sample_rate, hps.n_fft, hps.num_mels, hps.fmin, hps.fmax)
    mel = np.dot(mel_basis, magnitude)
    mel = 20 * np.log10(np.maximum(1e-5, mel))

    mel = np.maximum((mel + hps.max_level_db - hps.ref_level_db) / hps.max_level_db, 1e-8)
    return mel


def inv_melspectrogram(magnitude):
    magnitude = (magnitude * hps.max_level_db) - hps.max_level_db + hps.ref_level_db
    S = np.power(10.0, magnitude * 0.05)
    mel_basis = librosa.filters.mel(hps.sample_rate, hps.n_fft, hps.num_mels, hps.fmin, hps.fmax)
    inv_mel_basis = np.linalg.pinv(mel_basis)
    inverse = np.dot(inv_mel_basis, S)
    S = np.maximum(1e-10, inverse)
    return griffin_lim(S ** 1.2)


def spectrogram(y):
    linear = _stft(y)
    magnitude = np.abs(linear)
    magnitude = 20 * np.log10(np.maximum(1e-5, magnitude))
    magnitude = np.maximum((magnitude + hps.max_level_db - hps.ref_level_db) / hps.max_level_db, 1e-8)
    return magnitude


def inv_spectrogram(magnitude):
    magnitude = (magnitude * hps.max_level_db) - hps.max_level_db + hps.ref_level_db
    S = np.power(10.0, magnitude * 0.05)
    return griffin_lim(S ** 1.2)


def griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hps.gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)

    return y


def _istft(y):
    return librosa.istft(y, hop_length=hps.hop_length, win_length=hps.win_length, window=hps.window)


def _stft(y):
    return librosa.stft(y=y, n_fft=hps.n_fft, window=hps.window, hop_length=hps.hop_length, win_length=hps.win_length)
