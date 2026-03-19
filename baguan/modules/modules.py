import lightning as L

import torch
from torch import optim, nn

import os
import numpy as np

# from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from copy import deepcopy
from baguan.models.baguan_v2 import BaguanV2
# from baguan.models.fuxi import Fuxi
from baguan.utils import Args, LinearWarmupCosineAnnealingLR
from baguan.utils import lat_weighted_mae, lat_weighted_rmse


class BaguanV2Module(L.LightningModule):
    def __init__(
        self, 
        args, 
        len_const_vars, 
        len_inp_vars, 
        len_out_vars,
        out_surface_vars,
        out_upper_vars,
        const_vars,
        transform,
        model_type: str = 'fuxi',
        pretrained_path: str = '',
    ):
        super().__init__()

        self.args = args
        self.out_surface_vars = out_surface_vars
        self.out_upper_vars = out_upper_vars
        self.transform = transform
        self.model_type = model_type

        variables = out_surface_vars + out_upper_vars + const_vars
        self.in_variables = []
        for v in variables:
            if 'tp' not in v:
                self.in_variables.append(v)
        self.out_variables = out_surface_vars + out_upper_vars
        self.ids = []
        for i, v in enumerate(self.out_variables):
            if 'tp' not in v:
                self.ids.append(i)

        self.net = BaguanV2()
        # self.net = Fuxi()
        self.lat = np.load(os.path.join(self.args.root, "lat.npy"))
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)
        print("finish loading in modules.py ......")

    def load_pretrained_weights(self, pretrained_path):
        # if os.path.isdir(pretrained_path):
        #     checkpoint_model = get_fp32_state_dict_from_zero_checkpoint(pretrained_path)
        # else:
        checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]

        state_dict = self.state_dict()

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def training_step(self, batch, batch_idx):
        surface_lst, upper_lst, in_constant, lead_time, hours, dates, months = batch
        inp_lst = torch.cat([surface_lst[:, :, :, 1:], upper_lst[:, :, :, 1:]], dim=2)
        in_constant = in_constant[:, :, 1:]
        lat = self.lat
        loss_lst = []
        
        inp = inp_lst[:, 0] # b, c, h, w
        for s in range(0, surface_lst.shape[1] - 1):
            y = inp_lst[:, s + 1]
            pred = self.net(
                torch.cat([inp[:, self.ids], in_constant], dim=1), 
                hours[:, s], dates[:, s], 
                self.in_variables, self.out_variables,
            )
            inp = pred

            # training loss
            loss_dict = lat_weighted_mae(
                pred, y, self.out_surface_vars + self.out_upper_vars, lat
            )
            for var in loss_dict.keys():
                if s > 1:
                    self.log(
                        "train/" + var + f"_{s * 6}_hours", loss_dict[var],
                        on_step=True, on_epoch=False, prog_bar=True,
                    )
                else:
                    self.log(
                        "train/" + var, loss_dict[var],
                        on_step=True, on_epoch=False, prog_bar=True,
                    )
            loss_lst.append(loss_dict['loss'])

            # check rmse
            loss_dict = lat_weighted_rmse(
                pred, y, self.out_surface_vars + self.out_upper_vars, lat, 
                transform=self.transform
            )
            for var in loss_dict.keys():
                if s > 1:
                    self.log(
                        "train/" + var + f"_{s * 6}_hours", loss_dict[var],
                        on_step=True, on_epoch=False, prog_bar=True,
                    )
                else:
                    self.log(
                        "train/" + var, loss_dict[var],
                        on_step=True, on_epoch=False, prog_bar=True,
                    )
        
        loss = torch.stack(loss_lst).mean()

        return loss

    def validation_step(self, batch, batch_idx):
        surface_lst, upper_lst, in_constant, lead_time, hours, dates, months = batch
        inp_lst = torch.cat([surface_lst[:, :, :, 1:], upper_lst[:, :, :, 1:]], dim=2)
        in_constant = in_constant[:, :, 1:]

        lat = self.lat
        loss_lst = []

        inp = inp_lst[:, 0] # b, c, h, w
        for s in range(0, surface_lst.shape[1] - 1):
            y = inp_lst[:, s + 1]
            pred = self.net(
                torch.cat([inp[:, self.ids], in_constant], dim=1), 
                hours[:, s], dates[:, s], 
                self.in_variables, self.out_variables,
            )
            inp = pred

            # check rmse
            loss_dict = lat_weighted_rmse(
                pred, y, self.out_surface_vars + self.out_upper_vars, lat, 
                transform=self.transform
            )
            for var in loss_dict.keys():
                self.log(
                    "val/" + var + f"_{s * 6}_hours", loss_dict[var],
                    on_step=False, on_epoch=True, prog_bar=True,
                    sync_dist=True
                )

        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "absolute_pos_embed" in name or \
                "cpb_mlp" in name or "logit_scale" in name or \
                    'relative_position_bias_table' in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.args.optim.lr,
                    "betas": (self.args.optim.betas[0], self.args.optim.betas[1]),
                    "weight_decay": self.args.optim.weight_decay,
                    "fused": True,
                },
                {
                    "params": no_decay,
                    "lr": self.args.optim.lr,
                    "betas": (self.args.optim.betas[0], self.args.optim.betas[1]),
                    "weight_decay": 0,
                    "fused": True,
                },
            ]
        )

        # return optimizer

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            **self.args.scheduler.__dict__
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


if __name__ == '__main__':
    module = BaguanV2Module()
    module.configure_optimizers()