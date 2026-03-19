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
        self.lr = 2e-4
        self.betas = [0.9, 0.99]
        self.weight_decay = 1e-5


class DataArgs:
    def __init__(self):
        self.batch_size = 1
        self.num_workers = 2
        self.prefetch_factor = 2


class SchedulerArgs:
    def __init__(self):
        self.warmup_epochs = 4000
        self.max_epochs = 40000
        self.warmup_start_lr = 1e-7
        self.eta_min = 1e-7


class LightningTrainerArgs:
    def __init__(self):
        self.num_nodes = 1
        self.accelerator = 'gpu'
        self.precision = '16-mixed'
        self.max_steps = 40000
        self.log_every_n_steps = 10
        self.val_check_interval = 1000
        # self.gradient_clip_val = 32.0
        self.devices = 8
        self.enable_progress_bar = True


class ArgsV1:
    def __init__(self):
        # TODO: parse the args
        # data args
        self.root = '/jupyter/data_025/'
        # ds args
        self.ds_config = 'configs/ds_config.json'
        self.seed = 1997
        self.save_interval = 1000
        # self.pretrained_path = '/jupyter/BaguanV2/checkpoint/continue_PT_11201133.ckpt'
        self.pretrained_path = '/jupyter/BaguanV1/checkpoint/baguan_v1_finetune_24h/epoch_epoch=005-v3.ckpt'

        self.data = DataArgs()
        self.model = ModelArgs()
        self.optim = OptimArgs()
        self.scheduler = SchedulerArgs()
        self.trainer = LightningTrainerArgs()