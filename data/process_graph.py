#####################################最终版本##################################
import os.path
import random
import torch
import pandas as pd
import numpy as np
import pickle as pkl
from math import cos, asin, sqrt, pi
from os.path import join
import math


def distance(lat1, lon1, lat2, lon2):
    r = 6371
    p = pi / 180
    a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 2 * r * asin(sqrt(a))

def distance_mat_form(lat_vec: np.ndarray, lon_vec: np.ndarray):
    # Shape of lat_vec & lon_vec: [n_poi, 1]
    r = 6371
    p = pi / 180
    lat_mat = np.repeat(lat_vec, lat_vec.shape[0], axis=-1)
    lon_mat = np.repeat(lon_vec, lon_vec.shape[0], axis=-1)
    a_mat = 0.5 - np.cos((lat_mat.T - lat_mat) * p) / 2 \
            + np.matmul(np.cos(lat_vec * p), np.cos(lat_vec * p).T) * (1 - np.cos((lon_mat.T - lon_mat) * p)) / 2
    return 2 * r * np.arcsin(np.sqrt(a_mat))

def gen_nei_graph(df: pd.DataFrame, df_time: pd.DataFrame, n_users, n_pois, train=False):
    nei_dict = {idx: [] for idx in range(n_users)}
    edges = [[], []]

    for uid, _item in df.groupby('uid'):
        time_dict = df_time.set_index('uid')[['val_piv', 'test_piv']].to_dict('index')
        if uid in time_dict:
            _val_piv = time_dict[uid]['val_piv']
            _test_piv = time_dict[uid]['test_piv']
        poi_list = _item['poi'].tolist()
        if train:
            poi_list = poi_list[: _val_piv]

        nei_dict[uid] = poi_list
        edges[0] += [uid for _ in poi_list]
        edges[1] += poi_list

    return nei_dict, torch.LongTensor(edges)

def gen_loc_graph(poi_loc, n_pois, thre, _dist_mat=None):
    if _dist_mat is None:
        lat_vec = np.array([poi_loc[poi][0] for poi in range(n_pois)], dtype=np.float64).reshape(-1, 1)

        lon_vec = np.array([poi_loc[poi][1] for poi in range(n_pois)], dtype=np.float64).reshape(-1, 1)

        _dist_mat = distance_mat_form(lat_vec, lon_vec)

    adj_mat = np.triu(_dist_mat <= thre, k=1)
    num_edges = adj_mat.sum()
    print(f'Edges on dist_graph: {num_edges}, avg degree: {num_edges / n_pois}')

    idx_mat = np.arange(n_pois).reshape(-1, 1).repeat(n_pois, -1)
    row_idx = idx_mat[adj_mat]
    col_idx = idx_mat.T[adj_mat]
    edges = np.stack((row_idx, col_idx))

    nei_dict = {poi: [] for poi in range(n_pois)}
    for e_idx in range(edges.shape[1]):
        src, dst = edges[:, e_idx]
        nei_dict[src].append(dst)
        nei_dict[dst].append(src)
    return _dist_mat, edges, nei_dict

def remap(df: pd.DataFrame, n_users, n_pois):
    uid_dict = dict(zip(pd.unique(df['uid']), range(n_users)))
    poi_dict = dict(zip(pd.unique(df['poi']), range(n_pois)))
    df['uid'] = df['uid'].map(uid_dict)
    df['poi'] = df['poi'].map(poi_dict)
    return df, uid_dict, poi_dict

random.seed(1234)
target_dataset = 'TKY'
source_pth = f'./raw/{target_dataset}'
time = 'time_tky.csv'

dist_pth = f'./processed/{target_dataset.lower()}_v0.3_r0.1'
columns_to_load_train = ['UserId','PoiId','Latitude', 'Longitude','UTCTimeOffset']
columns_to_load = ['uid', 'poi', 'lat', 'lon', 'time']
review_pth = join(dist_pth, 'all_data.pkl')

if not os.path.exists(review_pth) or True:
    print(f'Load from {source_pth}\nData preprocessing...')
    trn_df = pd.read_csv(join(source_pth, '18_train_final_v0.3_r0.1_seed42_time10m.csv'),usecols=columns_to_load_train,encoding='unicode_escape')
    val_df = pd.read_csv(join(source_pth, 'val_data_media.csv'), usecols=columns_to_load, encoding='unicode_escape')
    tst_df = pd.read_csv(join(source_pth, 'tst_data_media.csv'), usecols=columns_to_load, encoding='unicode_escape')
    time_df = pd.read_csv(join(source_pth, time ), encoding='unicode_escape')

    column_rename_map = {
        'UserId': 'uid',
        'PoiId': 'poi',
        'Latitude': 'lat',
        'Longitude': 'lon',
        'UTCTimeOffset': 'time'}
    trn_df.rename(columns=column_rename_map, inplace=True)
    trn_df = trn_df[['uid', 'poi', 'lat', 'lon', 'time']]
    val_df = val_df[['uid', 'poi', 'lat', 'lon', 'time']]
    tst_df = tst_df[['uid', 'poi', 'lat', 'lon', 'time']]

    review_df = pd.concat((trn_df, val_df, tst_df))
    n_user, n_poi = pd.unique(review_df['uid']).shape[0], pd.unique(review_df['poi']).shape[0]


    trn_set, val_set, tst_set = [], [], []
    time_dict = time_df.set_index('uid')[['val_piv', 'test_piv']].to_dict('index')

    for uid, line in review_df.groupby('uid'):
        if uid in time_dict:
            val_piv = time_dict[uid]['val_piv']
            test_piv = time_dict[uid]['test_piv']

        pos_list, time_list = line['poi'].tolist(), line['time'].tolist()
        for i in range(1, len(pos_list)):
            location = (line['lat'].iloc[i], line['lon'].iloc[i])
            if time_list[i] < val_piv:
                trn_set.append((uid, pos_list[i], pos_list[:i], time_list[:i], time_list[i]))
            elif val_piv <= time_list[i]< test_piv:
                val_set.append((uid, pos_list[i], pos_list[:i], time_list[:i], time_list[i]))
            else:
                tst_set.append((uid, pos_list[i], pos_list[:i], time_list[:i], time_list[i]))
    # get loc_dict
    loc_dict = {poi: None for poi in range(1,n_poi+1)}
    for poi, item in review_df.groupby('poi'):
        lat, lon = item['lat'].iloc[0], item['lon'].iloc[0]
        loc_dict[poi] = (lat, lon)

    print('trn_set size:', len(trn_set))
    print('val_set size:', len(val_set))
    print('tst_set size:', len(tst_set))
    print('trn_df size:', len(trn_df))
    print('val_df size:', len(val_df))
    print('tst_df size:', len(tst_df))
    print('review_df size:', len(review_df))

    if not os.path.exists(dist_pth):
        os.mkdir(dist_pth)
    with open(review_pth, 'wb') as f:
        pkl.dump((n_user, n_poi), f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(review_df, f, pkl.HIGHEST_PROTOCOL)
        pkl.dump((trn_set, val_set, tst_set), f, pkl.HIGHEST_PROTOCOL)
        pkl.dump((trn_df, val_df, tst_df), f, pkl.HIGHEST_PROTOCOL)
        pkl.dump(loc_dict, f, pkl.HIGHEST_PROTOCOL)
    print(f'Process Done\n')

with open(review_pth, 'rb') as f:
    n_user, n_poi = pkl.load(f)
    review_df = pkl.load(f)
    trn_set, val_set, tst_set = pkl.load(f)
    trn_df, val_df, tst_df = pkl.load(f)
    loc_dict = pkl.load(f)

print(f'Remapped data loaded from {review_pth}')
print(f'#Interaction {len(review_df)}, #User: {n_user}, #POI: {n_poi}')
print(f'Avg.#visit: {len(review_df) / n_user}, density: {len(review_df) / n_user / n_poi}')
print(f'Full data size: {review_df.shape[0]}, #User: {n_user}, #POI: {n_poi}')
print(f'Train size: {trn_df.shape[0]}, Val size: {val_df.shape[0]}, Test size: {tst_df.shape[0]}')


print('Generating UI graph...')
time_df = pd.read_csv(join(source_pth, time ), encoding='unicode_escape')
ui_nei_dict, ui_edges = gen_nei_graph(review_df,time_df, n_user, n_poi)
with open(join(dist_pth, 'ui_graph.pkl'), 'wb') as f:
    pkl.dump(ui_nei_dict, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(ui_edges, f, pkl.HIGHEST_PROTOCOL)

dist_threshold = 1.
print(f'UI graph dumped, generating location graph with delta d: {dist_threshold}km...\n')

dist_mat = None

dist_mat, dist_edges, dist_dict = gen_loc_graph(loc_dict, n_poi, dist_threshold, dist_mat)
with open(join(dist_pth, 'dist_graph.pkl'), 'wb') as f:
    pkl.dump(dist_edges, f, pkl.HIGHEST_PROTOCOL)
    pkl.dump(dist_dict, f, pkl.HIGHEST_PROTOCOL)
np.save(join(dist_pth, 'dist_mat.npy'), dist_mat)

dist_on_graph = dist_mat[dist_edges[0], dist_edges[1]]
np.save(join(dist_pth, 'dist_on_graph.npy'), dist_on_graph)
print('Distance graph dumped, process done.')

