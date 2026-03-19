class ModelArgs:
    def __init__(self):
        # model args for baguan_v2.py
        self.img_size = [721, 1440]
        self.patch_size = 8
        self.depth = 48
        self.embed_dim = 1536
        self.num_heads = 24
        self.window_size = 18
        self.use_checkpoint = True
        self.use_reentrant = False
        self.ape = False


class OptimArgs:
    def __init__(self):
        self.lr = 7e-4
        self.betas = [0.9, 0.95]
        self.weight_decay = 0.1


class DataArgs:
    def __init__(self):
        self.batch_size = 1
        self.num_workers = 4
        self.prefetch_factor = 2


class SchedulerArgs:
    def __init__(self):
        self.warmup_epochs = 0
        self.max_epochs = 100000
        self.warmup_start_lr = 1e-7
        self.eta_min = 1e-7


class LightningTrainerArgs:
    def __init__(self):
        self.num_nodes = 2
        self.accelerator = 'gpu'
        self.precision = '16-mixed'
        self.max_steps = 100000
        self.log_every_n_steps = 10
        self.val_check_interval = 1000
        # self.gradient_clip_val = 32.0
        self.devices = 8
        self.enable_progress_bar = True


class Args:
    def __init__(self):
        # TODO: parse the args
        # data args
        self.root = '/jupyter/data_025/'
        # ds args
        self.ds_config = 'configs/ds_config.json'
        self.seed = 1997
        self.save_interval = 1000
        # self.pretrained_path = '/jupyter/BaguanV2/checkpoint/patch6_hid1536_head16_window20_swinv2_singleframe/epoch_epoch=009-v9.ckpt'
        self.pretrained_path = ''

        self.data = DataArgs()
        self.model = ModelArgs()
        self.optim = OptimArgs()
        self.scheduler = SchedulerArgs()
        self.trainer = LightningTrainerArgs()