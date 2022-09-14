from text import symbols


class hparams:
	################################
	# Data Parameters              #
	################################
	text_cleaners=['multi_cleaner']

	################################
	# Audio                        #
	################################
	num_mels = 80
	sample_rate = 22050
	n_fft = 2048
	win_length = 1024
	hop_length = 256
	window = 'hann'
	fmin = 50
	fmax = 11000
	max_level_db = 100
	min_level_db = -100
	ref_level_db = 20
	gl_iters = 16
	# num_mels = 80
	# num_freq = 1025
	# sample_rate = 22050
	# frame_length_ms = 50
	# frame_shift_ms = 12.5
	# preemphasis = 0.97
	# min_level_db = -100
	# ref_level_db = 20
	# power = 1.5
	# gl_iters = 30

	################################
	# Model Parameters             #
	################################
	n_symbols = len(symbols)
	symbols_embedding_dim = 512

	# Encoder parameters
	encoder_kernel_size = 5
	encoder_n_convolutions = 3
	encoder_embedding_dim = 512

	# Decoder parameters
	n_frames_per_step = 2
	decoder_rnn_dim = 1024
	prenet_dim = 256
	max_decoder_steps = 1000
	gate_threshold = 0.5
	p_attention_dropout = 0.1
	p_decoder_dropout = 0.1

	# Attention parameters
	attention_rnn_dim = 1024
	attention_dim = 256

	# Location Layer parameters
	attention_location_n_filters = 32
	attention_location_kernel_size = 31

	# Mel-post processing network parameters
	postnet_embedding_dim = 512
	postnet_kernel_size = 5
	postnet_n_convolutions = 5

	# Linear post processing network parameters
	cbhg_kernel_size = 8
	cbhg_projections = [256, num_mels]

	################################
	# Train                        #
	################################
	is_cuda = False
	pin_mem = True
	n_workers = 8
	lr = 2e-3
	betas = (0.9, 0.999)
	eps = 1e-6
	sch = True
	sch_step = 4000
	use_guided_attention = False
	guided_attention_sigma = 0.1
	use_eos = True
	max_iter = 8e5
	batch_size = 16
	iters_per_log = 10
	iters_per_sample = 1000
	iters_per_ckpt = 1000
	weight_decay = 1e-6
	grad_clip_thresh = 1.0
	mask_padding = True
	p = 10 # mel spec loss penalty
	eg_text = '존경하고 사랑하는 광주 시민 여러분, 광주 시장 인사드리게습니다.'
