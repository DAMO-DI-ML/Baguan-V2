import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ERA5Datasets(Dataset):
    """
    1. 每n_steps次迭代之后更新一次步长，每个步长产出steps+2条数据
    2. valid全部步长，train当前步长
    3. 可以跨年度
    """
    NEW_VARS = [
        '100m_u_component_of_wind', '100m_v_component_of_wind', '2m_dewpoint_temperature',
        'surface_pressure', 'total_cloud_cover', 'low_cloud_cover',
        'mean_surface_direct_short_wave_radiation_flux', 'mean_surface_downward_short_wave_radiation_flux',
        'total_column_water', 'total_column_water_vapour', 'total_precipitation_1hr', 'total_precipitation_6hr',
        'sea_surface_temperature'
    ]

    DEFAULT_VARS = [
        "z_50", "q_50", "t_50", "u_50", "v_50", "z_100", "q_100", "t_100", "u_100", "v_100", 
        "z_150", "q_150", "t_150", "u_150", "v_150", "z_200", "q_200", "t_200", "u_200", "v_200", 
        "z_250", "q_250", "t_250", "u_250", "v_250", "z_300", "q_300", "t_300", "u_300", "v_300", 
        "z_400", "q_400", "t_400", "u_400", "v_400", "z_500", "q_500", "t_500", "u_500", "v_500", 
        "z_600", "q_600", "t_600", "u_600", "v_600", "z_700", "q_700", "t_700", "u_700", "v_700", 
        "z_850", "q_850", "t_850", "u_850", "v_850", "z_925", "q_925", "t_925", "u_925", "v_925", 
        "z_1000", "q_1000", "t_1000", "u_1000", "v_1000", "u10", "v10", "t2m", "msl", 
        'u100', 'v100', 'd2m', 'sp', 'tcc', 'lcc', 'avg_sdswrf', 'avg_sdirswrf',
        'tcw', 'tcwv', 'tp1h', 'tp6h', 'sst'
    ]
    DEFAULT_MAP = {}
    for i, v in enumerate(DEFAULT_VARS):
        DEFAULT_MAP[v] = i

    DEFAULT_LEVELS = [
        50, 100, 150, 200, 250, 300, 400, 
        500, 600, 700, 850, 925, 1000
    ]
    DEFAULT_UPPER_VARS = ['z', 'q', 't', 'u', 'v']
    DEFAULT_SURFACE_VARS = [
        'u10', 'v10', 't2m', 'msl', 
        'u100', 'v100', 'd2m', 'sp', 'tcc', 'lcc', 'avg_sdswrf', 'avg_sdirswrf',
        'tcw', 'tcwv', 'tp1h', 'tp6h', 'sst'
    ]
    # DEFAULT_SURFACE_VARS = ['u10', 'v10', 't2m', 'msl']
    DEFAULT_CONSTANT_VARS = [
        'angle_of_sub_gridscale_orography', 'anisotropy_of_sub_gridscale_orography',
        'geopotential_at_surface', 'lake_cover',
        'land_sea_mask', 'soil_type',
    ]
    DATES = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def __init__(
        self, 
        root: str, 
        # variable string list
        in_upper_vars: list = DEFAULT_UPPER_VARS, 
        in_upper_levels: list = DEFAULT_LEVELS,
        in_surface_vars: list = DEFAULT_SURFACE_VARS,
        out_upper_vars: list = DEFAULT_UPPER_VARS,
        out_upper_levels: list = DEFAULT_LEVELS,
        out_surface_vars: list = DEFAULT_SURFACE_VARS,
        constant_vars: list = DEFAULT_CONSTANT_VARS,
        new_vars: list = NEW_VARS,
        # normalization path
        norm_mean_file: str = 'normalize_mean.npz', # npz file
        norm_std_file: str = 'normalize_std.npz', # npz file
        # shard information
        n_shard: int = 60,
        data_per_file: int = 146,
        predict_range: int = 6,
        flag: str = 'train', # ['train', 'val', 'test']
        data_gap: int = 6,
        year_st_ed: dict = {
            'train': [1979, 2021], 
            'val': [2022, 2022], 
            'test': [2022, 2022]
        },
        n_steps: int = 100000,
        st_step: int = 1,
        val_step: int = 12,
    ):
        super().__init__()

        self.root = root
        self.flag = flag
        
        self.constant_vars = constant_vars

        self.in_surface_vars = in_surface_vars
        self.in_upper_vars = in_upper_vars
        self.in_upper_levels = in_upper_levels

        self.out_surface_vars = out_surface_vars
        self.out_upper_vars = out_upper_vars
        self.out_upper_levels = out_upper_levels

        self.new_vars = new_vars

        self.var_map = self.DEFAULT_MAP
        self.in_surface_ids, self.in_upper_ids, \
            self.out_surface_ids, self.out_upper_ids, \
                self.out_upper_vars_all = self.get_var_lst()

        self.predict_range = predict_range
        self.data_gap = data_gap
        if flag != 'train':
            self.data_gap = 12

        self.norm_mean = np.load(os.path.join(root, norm_mean_file))
        self.norm_std = np.load(os.path.join(root, norm_std_file))

        self.constants = self.get_constant_vars(constant_vars)

        self.in_surface_transforms = self.get_transforms(self.in_surface_vars)
        self.in_upper_transforms = self.get_transforms(self.in_upper_vars, self.in_upper_levels)
        self.out_surface_transforms = self.get_transforms(self.out_surface_vars)
        self.out_upper_transforms = self.get_transforms(self.out_upper_vars, self.out_upper_levels)
        self.constant_transforms = self.get_transforms(constant_vars)

        self.year_st = year_st_ed[flag][0]
        self.n_shard = n_shard
        self.data_per_file = data_per_file
        self.n_data_per_year = data_per_file * n_shard // self.data_gap
        self.n_data_tot = \
            (year_st_ed[flag][1] - year_st_ed[flag][0] + 1) * self.n_data_per_year - 2 - val_step * (predict_range // 6) # 1460 * 39 - 2 - 12

        self.cnt = 0
        self.n_steps = n_steps
        self.curr_step = st_step if self.flag == 'train' else val_step

    def get_transforms(self, variables, levels=None):
        mean_lst, std_lst = [], []

        if levels is not None:
            for v in variables:
                for l in levels:
                    var = f"{v}_{l}"
                    mean_lst.append(self.norm_mean[var])
                    std_lst.append(self.norm_std[var])
        else:
            for v in variables:
                mean_lst.append(self.norm_mean[v])
                std_lst.append(self.norm_std[v])

        normalize_mean = np.concatenate(mean_lst)
        normalize_std = np.concatenate(std_lst)
        return transforms.Normalize(normalize_mean, normalize_std)

    def get_var_lst(self):
        in_surface_ids, in_upper_ids = [], []

        for v in self.in_surface_vars:
            in_surface_ids.append(self.var_map[v])
        for v in self.in_upper_vars:
            for l in self.in_upper_levels:
                in_upper_ids.append(self.var_map[f'{v}_{l}'])

        out_surface_ids, out_upper_ids, out_upper_vars = [], [], []
        for v in self.out_surface_vars:
            out_surface_ids.append(self.var_map[v])
        for v in self.out_upper_vars:
            for l in self.out_upper_levels:
                out_upper_vars.append(f'{v}_{l}')
                out_upper_ids.append(self.var_map[f'{v}_{l}'])
        
        return \
            in_surface_ids, in_upper_ids, \
            out_surface_ids, out_upper_ids, out_upper_vars

    def get_constant_vars(self, constant_vars):
        constant_data_list = []
        for var in constant_vars:
            path = os.path.join(self.root, 'constant', f'{var}.npy')
            constant_data_list.append(np.load(path)[0, :, :].astype(np.float32)) # [721, 1440]
        constant_data = np.vstack(constant_data_list)
        return constant_data

    def __getitem__(self, index):

        idx = index * self.data_gap
        
        def get_hour_date_month(i):
            # i = i % (self.n_data_per_year)
            hour = i % 24
            date, month, res_idx = 0, 1, (i // 24) % 365
            for d in self.DATES:
                if res_idx >= d:
                    res_idx = res_idx - d
                    month = month + 1
                else:
                    date = res_idx + 1
                    break
            date = (i // 24) % 365
            hour = torch.tensor([hour]).long()
            date = torch.tensor([date]).long()
            month = torch.tensor([month]).long()
            return hour, date, month

        def get_year_shard_idx(i):
            year = i // (self.data_per_file * self.n_shard) + self.year_st
            cur_idx = i % (self.data_per_file * self.n_shard)
            shard = cur_idx // self.data_per_file
            in_shard_idx = cur_idx % self.data_per_file
            return year, shard, in_shard_idx

        def read_data(year, shard, in_shard_idx):
            try:
                path = os.path.join(
                    '/data/data_025/train', 
                    f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                )
                return np.load(path).astype(np.float32)
            except:

                if year == 2019:
                    flag = 'val'
                elif year in [2018, 2020, 2021]:
                    flag = 'test'
                else:
                    flag = 'train'

                try:
                    path = os.path.join(
                        self.root, flag, 
                        f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                    )
                    return np.load(path).astype(np.float32)
                except:
                    try: 
                        path = os.path.join(
                            '/jupyter/data_025', flag, 
                            f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                        )
                        return np.load(path).astype(np.float32)
                    except:
                        path_1 = os.path.join(
                            '/jupyter/ea120/data', flag, 
                            f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                        )
                        path_2 = os.path.join(
                            '/jupyter/data_025', flag, 
                            f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                        )

                        dir_path_2 = os.path.join(
                            '/jupyter/data_025', flag, 
                            f'{year}_{shard}/'
                        )
                        if not os.path.exists(dir_path_2):
                            os.system(f'mkdir {dir_path_2}')
                        os.system(f'sudo cp {path_1} {path_2}')

                        path = os.path.join(
                            '/jupyter/ea120/data', flag, 
                            f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                        )
                        return np.load(path).astype(np.float32)

        def read_new_data(year, shard, in_shard_idx):
            data_lst = []
            for v in self.new_vars:
                try:
                    path = os.path.join(
                        '/data/data_025/new_data', v,
                        f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                    )
                    data = np.load(path).astype(np.float32) # 1 h w
                except:
                    path = os.path.join(
                        '/jupyter/data_v2/processed_data', v,
                        f'{year}_{shard}/{year}_{shard}_{in_shard_idx}.npy'
                    )
                    data = np.load(path).astype(np.float32) # 1 h w
                if v == 'sea_surface_temperature':
                    data[np.isnan(data)] = self.norm_mean['sst']
                data_lst.append(data[0])
                
            return np.stack(data_lst, axis=0)

        surface_lst, upper_lst = [], []
        hours, dates, months = [], [], []

        # worker_info = torch.utils.data.get_worker_info()
        # rank = torch.distributed.get_rank()
        # world_size = torch.distributed.get_world_size()
        # num_workers_per_ddp = worker_info.num_workers
        # worker_id = rank * num_workers_per_ddp + worker_info.id

        ids_lst = [idx + i * self.predict_range for i in range(self.curr_step + 1)]
        for tt, _id in enumerate(ids_lst):
            year, shard, in_shard_idx = get_year_shard_idx(_id)
            data = read_data(year, shard, in_shard_idx)

            # if worker_id == 0:
            #     print(tt, year, shard, in_shard_idx)

            if len(self.new_vars) > 0:
                new_data = read_new_data(year, shard, in_shard_idx)
                data = np.concatenate((data, new_data), axis=0)
            
            data_surface, data_upper = data[self.in_surface_ids], data[self.in_upper_ids]

            surface_lst.append(self.in_surface_transforms(torch.from_numpy(data_surface)))
            upper_lst.append(self.in_upper_transforms(torch.from_numpy(data_upper)))

            hour, date, month = get_hour_date_month(_id)
            hours.append(hour)
            dates.append(date)
            months.append(month)

        surface_lst = torch.stack(surface_lst, dim=0) # t, c, h, w
        upper_lst = torch.stack(upper_lst,dim=0) # t, c, h, w
        hours = torch.stack(hours, dim=0) # t, 1
        dates = torch.stack(dates, dim=0) # t, 1
        months = torch.stack(months, dim=0) # t, 1
        
        lead_time = self.predict_range
        in_constant = self.constant_transforms(torch.from_numpy(self.constants))
        lead_time = torch.tensor([lead_time]).to(surface_lst.dtype)

        self.cnt += 1
        if self.cnt % self.n_steps == 0 and self.flag == 'train':
            self.curr_step += 1
            self.curr_step = min(self.curr_step, 12)

        return surface_lst, upper_lst, in_constant, lead_time, hours, dates, months

    def __len__(self):
        if self.flag != 'train':
            return 80
        else:
            return self.n_data_tot

if __name__ == '__main__':
    dataset = ERA5Datasets(root='/jupyter/ea120/data')
