

class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create spk_embedder hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "epochs": 1000,
        "iters_per_checkpoint": 1000,
        "seed": 16807,
        "dynamic_loss_scaling": True,
        "fp16_run": False,
        "distributed_run": False,
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        # "cudnn_enabled": True,
        # "cudnn_benchmark": False,
        "output_directory": "PPG2Mel-72-uni-MultiSPK-24k-VCC-w30-large-dc1024",  # Directory to save checkpoints.
        # "output_directory": "PPG2Mel-514-MultiSPK-16k-LibriTTS-fixed-w30-large",  # Directory to save checkpoints.
        # "output_directory": "PPG2Mel-514-MultiSPK-16k-MF-zeros-w30-large",  # Directory to save checkpoints.
        "log_directory": 'log',
        "checkpoint_path": 'PPG2Mel-72-MultiSPK-24k-VCC-w30-large-dc1024/ckpt/checkpoint_30000.pt',  # Path to a checkpoint file. 121000 for 1.12 alignment
        # "checkpoint_path": 'PPG2Mel-514-MultiSPK-16k-LibriTTS-fixed-w30-large/ckpt/checkpoint_67000.pt',  # Path to a checkpoint file. 121000 for 1.12 alignment
        # "checkpoint_path": 'PPG2Mel-514-MultiSPK-16k-VCTK-fixed-w30-large/ckpt/checkpoint_136000.pt',  # Path to a checkpoint file. 121000 for 1.12 alignment
        "warm_start": True,  # Load the model only (warm start)
        "n_gpus": 1,  # Number of GPUs
        "rank": 0,  # Rank of current gpu
        "group_name": 'group_name',  # Distributed group name

        ################################
        # Data Parameters             #
        ################################
        # Passed as a txt file, see data/filelists/training-set.txt for an
        # example.
        "training_files": '',
        # Passed as a txt file, see data/filelists/validation-set.txt for an
        # example.
        "validation_files": '',
        "is_full_ppg": True,  # Whether to use the full PPG or not.
        "is_append_f0": False,  # Currently only effective at sentence level
        "ppg_subsampling_factor": 1,  # Sub-sample the ppg & acoustic sequence.
        # Cases
        # |'load_feats_from_disk'|'is_cache_feats'|Note
        # |True                  |True            |Error
        # |True                  |False           |Please set cache path
        # |False                 |True            |Overwrite the cache path
        # |False                 |False           |Ignores the cache path
        "load_feats_from_disk": False,  # Remember to set the path.
        # Mutually exclusive with 'load_feats_from_disk', will overwrite
        # 'feats_cache_path' if set.
        "is_cache_feats": False,
        "feats_cache_path": '',

        ################################
        # Audio Parameters             #
        ################################
        "max_wav_value": 32768.0,
        "sampling_rate": 16000,
        "n_acoustic_feat_dims": 80,
        "filter_length": 1024, # not using
        "hop_length": 160, # not using
        "win_length": 1024, # not using
        "mel_fmin": 0.0,
        "mel_fmax": 8000.0,

        ################################
        # Model Parameters             #
        ################################
        "n_symbols": 72, # 514
        # "n_symbols": 5816,
        "symbols_embedding_dim": 512,

        # Encoder parameters
        "encoder_kernel_size": 5,
        "encoder_n_convolutions": 3,
        "encoder_embedding_dim": 512,
        "spk_embedding_dim": 256,

        # Decoder parameters
        "decoder_rnn_dim": 1024, # 1024
        "prenet_dim": 256,
        "max_decoder_steps": 8000,
        "gate_threshold": 0.5,
        "p_attention_dropout": 0.1,
        "p_decoder_dropout": 0.1,

        # Attention parameters
        "attention_rnn_dim": 1024,
        "attention_dim": 128,
        # +- time steps to look at when computing the attention. Set to None
        # to block it.
        "attention_window_size": 30,

        # Location Layer parameters
        "attention_location_n_filters": 32,
        "attention_location_kernel_size": 31,

        # Mel-post processing network parameters
        "postnet_embedding_dim": 512,
        "postnet_kernel_size": 5,
        "postnet_n_convolutions": 5,

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip_thresh": 1.0,
        "batch_size": 8,
        "mask_padding": True,  # set model's padded outputs to padded values
        "mel_weight": 1,
        "gate_weight": 0, # 0.005

        #######################
        # GST Hyperparameters #
        #######################
        "token_dim": 256,
        # reference encoder
        "ref_enc_filters": [32, 32, 64, 64, 128, 128],
        "ref_enc_size": [3, 3],
        "ref_enc_strides": [2, 2],
        "ref_enc_pad": [1, 1],
        "ref_enc_gru_size": 128,
        # style token layer
        "token_num": 20,
        # token_emb_size: 256,
        "num_heads": 1, # 8
        # multihead_attn_num_unit: 256,
        # style_att_type: 'mlp_attention',
        # attn_normalize: True,
        "dropout_p": 0.5,
        "n_mels": 80  # Number of Mel banks to generate
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view

