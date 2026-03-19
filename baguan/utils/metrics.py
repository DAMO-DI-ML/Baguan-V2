import numpy as np
import torch

def lat_weighted_mae(pred, y, vars, lat):
    error = (pred - y).abs()

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    if pred.shape[2] == 720:
        w_lat = w_lat[1:]
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)
    w_lat = w_lat.abs() ** 0.5

    aug = 1.6
    weight_dict = {
        't2m': 0.2, 'u10': 0.15, 'v10': 0.15, 'msl': 0.15, 'u100': 0.15, 'v100': 0.15,
        'd2m': 0.08, 'sp': 0.08, 'sst': 0.08, 'tcc': 0.08, 'lcc': 0.08, 'tcw': 0.08, 'tcwv': 0.08, 
        'avg_sdswrf': 0.08, 'avg_sdirswrf': 0.08, 'tp1h': 0.08, 'tp6h': 0.08,

        'z_50': 0.008298755186721992, 'z_100': 0.016597510373443983, 'z_150': 0.024896265560165973, 'z_200': 0.03319502074688797 * aug * 2, 'z_250': 0.04149377593360996, 
        'z_300': 0.04979253112033195, 'z_400': 0.06639004149377593, 'z_500': 0.08298755186721991 * aug, 'z_600': 0.0995850622406639, 'z_700': 0.11618257261410789, 
        'z_850': 0.14107883817427386, 'z_925': 0.15352697095435686, 'z_1000': 0.16597510373443983,

        'u_50': 0.008298755186721992, 'u_100': 0.016597510373443983, 'u_150': 0.024896265560165973, 'u_200': 0.03319502074688797 * aug * 2, 'u_250': 0.04149377593360996, 
        'u_300': 0.04979253112033195, 'u_400': 0.06639004149377593, 'u_500': 0.08298755186721991 * aug, 'u_600': 0.0995850622406639, 'u_700': 0.11618257261410789, 
        'u_850': 0.14107883817427386, 'u_925': 0.15352697095435686, 'u_1000': 0.16597510373443983,

        'v_50': 0.008298755186721992, 'v_100': 0.016597510373443983, 'v_150': 0.024896265560165973, 'v_200': 0.03319502074688797 * aug * 2, 'v_250': 0.04149377593360996, 
        'v_300': 0.04979253112033195, 'v_400': 0.06639004149377593, 'v_500': 0.08298755186721991 * aug, 'v_600': 0.0995850622406639, 'v_700': 0.11618257261410789, 
        'v_850': 0.14107883817427386, 'v_925': 0.15352697095435686, 'v_1000': 0.16597510373443983,

        't_50': 0.008298755186721992, 't_100': 0.016597510373443983, 't_150': 0.024896265560165973, 't_200': 0.03319502074688797 * aug * 2, 't_250': 0.04149377593360996, 
        't_300': 0.04979253112033195, 't_400': 0.06639004149377593, 't_500': 0.08298755186721991 * aug, 't_600': 0.0995850622406639, 't_700': 0.11618257261410789, 
        't_850': 0.14107883817427386, 't_925': 0.15352697095435686, 't_1000': 0.16597510373443983,
        
        'q_50': 0.008298755186721992, 'q_100': 0.016597510373443983, 'q_150': 0.024896265560165973, 'q_200': 0.03319502074688797 * aug * 2, 'q_250': 0.04149377593360996, 
        'q_300': 0.04979253112033195, 'q_400': 0.06639004149377593, 'q_500': 0.08298755186721991 * aug, 'q_600': 0.0995850622406639, 'q_700': 0.11618257261410789, 
        'q_850': 0.14107883817427386, 'q_925': 0.15352697095435686, 'q_1000': 0.16597510373443983,
    }
    for v in ['z', 'q', 'u', 'v', 't']:
        weight_dict[v + '_50'] *= 0.1
        weight_dict[v + '_100'] *= 0.2
        weight_dict[v + '_150'] *= 0.4
        weight_dict[v + '_250'] *= 0.6
        weight_dict[v + '_300'] *= 0.8

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[var + '_mae'] = (error[:, i] * w_lat).mean()

    weights = torch.Tensor([weight_dict[var] for var in vars]).to(device=error.device).view(1, -1, 1, 1)
    weights = weights / weights.sum()
    loss_dict["loss"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()
    
    # loss_dict["loss"] = torch.stack([loss_dict[k] for k in loss_dict.keys()]).mean()

    return loss_dict

def lat_weighted_rmse(pred, y, vars, lat, transform=None):

    if transform is not None:
        pred = transform(pred.to(torch.float32))
        y = transform(y.to(torch.float32))

    error = (pred - y) ** 2

    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    if pred.shape[2] == 720:
        w_lat = w_lat[1:]
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f'w_rmse_{var}'] = torch.mean(
            torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
        )
    
    return loss_dict


def lat_weighted_acc(pred, y, vars, lat, clim, transform=None):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    if transform is not None:
        pred = transform(pred.to(torch.float32))
        y = transform(y.to(torch.float32))

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].float().cpu() for k in loss_dict.keys()])

    return loss_dict