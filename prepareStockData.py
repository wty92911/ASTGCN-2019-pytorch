import os
import csv
import json
import torch
import pandas as pd
from datetime import datetime
import numpy as np


def split_minute_data(minute_data, start_minute, seq_len = 12, gap_len = 16, pre_len = 12, select_nums=30):
    daily_minutes = 240 
    select_nums = np.random.randint(1, 2*select_nums, size=1)[0]
    num_days = minute_data.shape[1] // daily_minutes
    X = []
    Y = []
    Z = []
    for i in range(num_days):
        start_index = i * daily_minutes
        end_index = (i + 1) * daily_minutes
        for _ in range(select_nums):
            start = np.random.randint(start_index, end_index - seq_len - gap_len - pre_len + 1)
            X.append(minute_data[:, start: start + seq_len])
            Y.append(minute_data[:, start + seq_len + gap_len: start + seq_len + gap_len + pre_len])
            Z.append(start + start_minute)
    return np.array(X), np.array(Y), np.array(Z)

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray (B,N,T)
    Returns
    ----------
    stats: dict, two keys: mean and std
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,2), keepdims=True)
    std = train.std(axis=(0,1,2), keepdims=True)
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm

np.random.seed(320427)
data_dir = '/home/undergrad2023/tywang/SPGCL/SPGCL-main/datasets'
data_name = 'CSI500'
date_format = r"%Y-%m-%d %H:%M:%S"
data_format = np.float32
with open(os.path.join(data_dir, data_name, "features_col.json"), 'r') as f:
    features_col = json.load(f)
with open(os.path.join(data_dir, data_name, "minute_features_col.json"), 'r') as f:
    minute_features_col = json.load(f)
with open(os.path.join(data_dir, data_name, "date.csv"), 'r') as f:
    reader = csv.reader(f)
    daily_date = [datetime.strptime(x[0], date_format) for x in reader] #[DT]
with open(os.path.join(data_dir, data_name, "minute_datetime.csv"), 'r') as f:
    reader = csv.reader(f)
    minute_datetime = [datetime.strptime(x[0], date_format) for x in reader] #[MT]

daily_features = torch.load(os.path.join(data_dir, data_name, "features.pth")).numpy().astype(data_format) #[N, DT, DF]
minute_features = torch.load(os.path.join(data_dir, data_name, "minute_features.pth")).numpy().astype(data_format) #[N, MT, MF]
reg_cap = pd.read_csv(os.path.join(data_dir, data_name, "reg_capital.csv")).values[:, 1].astype(data_format) #[N]
num_nodes = daily_features.shape[0] # = minute_features.shape[0] = 500

industry = daily_features[:, :, features_col.index('industry')] #[N, DT] == [N, repeat]
industry = industry[:, 0] #[N]
turnover_rate = daily_features[:, :, features_col.index('turnover_rate')] #[N, DT]
turnover_rate = np.mean(turnover_rate, axis=1).astype(data_format) + 1e-3 #[N]
minute_close = minute_features[:, :, minute_features_col.index('close')] #[N, MT] as label

reg_cap += 1e-3
daily_industry_dist = reg_cap[np.newaxis, :] / reg_cap[:, np.newaxis] + turnover_rate[np.newaxis, :] / turnover_rate[:, np.newaxis]
daily_industry_dist[industry[:, np.newaxis] != industry[np.newaxis, :]] = 0
# save daily industry distance as CSI500.csv with title from to dis
np.savetxt(os.path.join(data_dir, data_name, "CSI500_dis.csv"), daily_industry_dist, delimiter=',', header='from,to,dis', comments='')

minute_per_day = 240
num_days = int(minute_close.shape[1] / minute_per_day)

# split minute_close into train, val, test with 3:1:1
split_line1 = int(num_days * 0.6) * minute_per_day
split_line2 = int(num_days * 0.8) * minute_per_day

train_set = minute_close[:, :split_line1]
val_set = minute_close[:, split_line1: split_line2]
test_set = minute_close[:, split_line2:]

train_x, train_target, train_timestamp = split_minute_data(train_set, 0)
val_x, val_target, val_timestamp  = split_minute_data(val_set, split_line1)
test_x, test_target, test_timestamp = split_minute_data(test_set, split_line2)

(stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

all_data = {
    'train': {
        'x': train_x_norm,
        'target': train_target,
        'timestamp': train_timestamp,
    },
    'val': {
        'x': val_x_norm,
        'target': val_target,
        'timestamp': val_timestamp,
    },
    'test': {
        'x': test_x_norm,
        'target': test_target,
        'timestamp': test_timestamp,
    },
    'stats': {
        '_mean': stats['_mean'],
        '_std': stats['_std'],
    }
}


print('train x:', all_data['train']['x'].shape)
print('train target:', all_data['train']['target'].shape)
print('train timestamp:', all_data['train']['timestamp'].shape)
print()
print('val x:', all_data['val']['x'].shape)
print('val target:', all_data['val']['target'].shape)
print('val timestamp:', all_data['val']['timestamp'].shape)
print()
print('test x:', all_data['test']['x'].shape)
print('test target:', all_data['test']['target'].shape)
print('test timestamp:', all_data['test']['timestamp'].shape)
print()
print('train data _mean :', stats['_mean'].shape, stats['_mean'])
print('train data _std :', stats['_std'].shape, stats['_std'])

np.savez_compressed(
    os.path.join(data_dir, data_name, "CSI500_sampled.npz"),
    train_x=all_data['train']['x'], train_target=all_data['train']['target'],
    train_timestamp=all_data['train']['timestamp'],
    val_x=all_data['val']['x'], val_target=all_data['val']['target'],
    val_timestamp=all_data['val']['timestamp'],
    test_x=all_data['test']['x'], test_target=all_data['test']['target'],
    test_timestamp=all_data['test']['timestamp'],
    mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
)