

class hparams:
    # 학습용 설정
    is_cuda = False
    gpu_id = '1'
    use_benchmark = True
    n_workers = 8
    batch_size = 8
    pin_mem = True
    lr = 5e-5
    betas = (0.9, 0.999)
    eps = 1e-8
    weight_decay = 1e-6
    use_scheduler = False
    scheduler_step = 4000
    grad_clip_thresh = 1.0
    iters_per_sample = 1000
    iters_per_save = 1000
    iters_per_log = 10
    max_train_iter = 3e6
    sample_audio = './data/kss/wavs/1_0000.wav'


    # 데이터 설정
    metadata = 'metadata.csv'
    file_index = 0
    segment_length = 16384 # default 16000 # 16384

    # 오디오 설정
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
    max_wav_value = 32768.0

    # num_mels = 80
    # num_freq = 1025
    # max_wav_value = 32768.0
    # sample_rate = 22050
    # frame_length_ms = 50
    # frame_shift_ms = 12.5
    # preemphasis = 0.97
    # min_level_db = -100
    # ref_level_db = 20
    # power = 1.5
    # gl_iters = 30
    # # STFT 설정
    # filter_length = 1024
    # hop_length = 256
    # win_length = 1024
    # mel_fmin = 0.0
    # mel_fmax = 8000.0



    # Model 설정
    # upsample_kerner_size = 1024
    # upsample_stride = 256
    upsample_kerner_size = win_length
    upsample_stride = hop_length

    n_flows = 12
    n_group = 8
    n_early_every = 4
    n_early_size = 2

    wn_n_layers = 8
    wn_n_channels = 256 # paper : 512, small 256
    wn_kernel_size = 3

    sigma = 1.0