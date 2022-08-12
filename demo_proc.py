import torch
import argparse
import numpy as np
import librosa
import hashlib
import struct
import time
import re
import os
# import setproctitle
import base64

from text import text_to_sequence, sequence_to_text
from model.model_inference_v2 import Tacotron2
from hparams import hparams as hps
from utils.util import mode, to_var, to_arr
from utils.audio import save_wav, inv_melspectrogram, inv_spectrogram
from utils.griffinlim import griffin_lim
from korean_text.korean_cleaner_cls import KoreanCleaner

from vocoder.model.waveglow import WaveGlow
from vocoder.denoiser_librosa import Denoiser

from flask import Flask, request, Response, jsonify

from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading


app = Flask(__name__)

device = 'cuda' # cpu

loaded_model = None
loaded_vocoder = None
loaded_denoiser = None

executor = None
korean_cleaner = None

@app.route('/tts/stream')
def test():
	text = request.args.get('text')
	sigma = float(request.args.get('sigma'))
	strength = float(request.args.get('strength'))

	task = partial(_infer, text, sigma, strength)
	future = executor.submit(task)
	text_to_wav, _, _ = future.result()

	return Response(text_to_wav, mimetype="audio/wav")

@app.route('/tts/lipsync')
def test_lipsync():
	text = request.args.get('text')
	sigma = float(request.args.get('sigma'))
	strength = float(request.args.get('strength'))

	task = partial(_infer, text, sigma, strength)
	future = executor.submit(task)
	text_to_wav, alignment, audio_duration = future.result()

	jamo_sync = _get_lipsync(text, text_to_wav, alignment, audio_duration)

	return jsonify(setOutput({'raw_base64': base64.b64encode(text_to_wav).decode('utf-8'), 'duration': round(audio_duration, 3), 'sync': jamo_sync}))


def _infer(text, sigma, strength):
	start_time = time.time()
	print('text : ', text)
	text = korean_cleaner.clean_text(text)
	print('clean text : ', text)

	sequence = text_to_sequence(text, hps.text_cleaners)
	sequence = to_var(torch.IntTensor(sequence)[None, :]).long()
	_, mel_outputs_postnet, linear_outputs, _, alignments = loaded_model.inference(sequence)

	with torch.no_grad():
		if loaded_vocoder is None:
			wav = griffin_lim(linear_outputs)
			wav *= 32767 / max(0.01, np.max(np.abs(wav)))

		else:
			wav = loaded_vocoder.infer(mel_outputs_postnet, sigma=sigma)
			wav *= 32767. / max(0.01, torch.max(torch.abs(wav)))
			wav = wav.squeeze()
			wav = wav.cpu().detach().numpy().astype('float32')
			wav = loaded_denoiser(wav, strength=strength)

	wav = np.append(wav, np.array([[0.0] * (hps.sample_rate // 2)]))
	audio_duration = librosa.get_duration(wav, hps.sample_rate)
	wav = wav.astype(np.int16)
	wav = _convert_to_pcm16(wav, 22050)

	print('{} seconds'.format(time.time() - start_time))

	return wav, alignments[0], audio_duration

def _get_lipsync(text, wav, alignment, audio_duration):
	text_seq = sequence_to_text(text_to_sequence(text, hps.text_cleaners))
	alignment = to_arr(alignment.transpose(0, 1))
	audio_duration = audio_duration * 1000 - 500

	step_size = audio_duration / len(alignment[1])
	max_indices = alignment.argmax(axis=1)

	jamo_sync = []

	for i, (symbol, max_idx) in enumerate(zip(text_seq, max_indices)):
		sync_time = max_idx * step_size
		jamo_sync.append({'id': i, 'jamo': str(symbol), 'sync_time': round(sync_time, 3)})

	return jamo_sync




def _convert_to_pcm16(wav_int16, sr):
		data = wav_int16
		dkind = data.dtype.kind
		fs = sr

		header_data = b''
		header_data += b'RIFF'
		header_data += b'\x00\x00\x00\x00'
		header_data += b'WAVE'
		header_data += b'fmt '

		format_tag = 0x0001  # WAVE_FORMAT_PCM
		channels = 1
		bit_depth = data.dtype.itemsize * 8
		bytes_per_second = fs * (bit_depth // 8) * channels
		block_align = channels * (bit_depth // 8)

		fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs, bytes_per_second, block_align, bit_depth)
		header_data += struct.pack('<I', len(fmt_chunk_data))
		header_data += fmt_chunk_data

		if ((len(header_data) - 4 - 4) + (4 + 4 + data.nbytes)) > 0xFFFFFFFF:
			raise ValueError("Data exceeds wave file size limit")

		data_chunk_data = b'data'
		data_chunk_data += struct.pack('<I', data.nbytes)
		header_data += data_chunk_data

		data_bytes = data.tobytes()

		data_pcm16 = header_data
		data_pcm16 += data_bytes

		return data_pcm16

def setOutput(result, errorCode=0, errorMessage=''):
	return {'result': result, 'errorCode': errorCode, 'errorMessage': errorMessage}


def _load_model(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth, map_location=torch.device(device))
	model = Tacotron2()
	model.load_state_dict(ckpt_dict['model'])
	model = mode(model, True).eval()
	# model.decoder.train()
	# model.postnet.train()
	return model

def _load_vocoder(ckpt_pth):
	ckpt_dict = torch.load(ckpt_pth, map_location=torch.device(device))
	vocoder = WaveGlow()
	vocoder.load_state_dict(ckpt_dict['model'])
	vocoder = vocoder.remove_weightnorm(vocoder)
	vocoder.to(device).eval()

	denoiser = Denoiser(vocoder, 0.1)
	return vocoder, denoiser


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int)
	parser.add_argument('--model_path', type=str, required=False, default='logs/model/acoustic.ckpt')
	parser.add_argument('--vocoder_path', type=str, required=False, default='logs/model/vocoder.ckpt')
	args = parser.parse_args()

	# proc_name = 'tts-proc-{0}-port-{1}'.format('kyeongsang', args.port)
	# setproctitle.setproctitle(proc_name)

	loaded_model = _load_model(args.model_path)
	loaded_vocoder, loaded_denoiser = _load_vocoder(args.vocoder_path)

	executor = ThreadPoolExecutor(max_workers=3)
	korean_cleaner = KoreanCleaner()

	app.run(host='0.0.0.0', port=args.port, threaded=True)
