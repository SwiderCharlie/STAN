import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import os


def get_adjacency_matrix(distance_df, sensor_ids):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[str(sensor_id)] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind or row[0] == row[1]:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    max_val = max(distances)
    adj_mx = np.exp(-np.square(dist_mx / std))
    # adj_mx[adj_mx < normalized_k] = 0
    adj_mx[adj_mx > 0] = 1
    adj_mx = adj_mx.astype(np.int32)

    # adj_dist_mx = dist_mx / max_val
    # adj_dist_mx[adj_mx == 0] = np.inf
    # for i in range(adj_dist_mx.shape[0]):
    #     for j in range(adj_dist_mx.shape[1]):
    #         if i == j:
    #             adj_mx[i, j] = 1
    #             adj_dist_mx[i, j] = 0

    return sensor_ids, sensor_id_to_ind, adj_mx


def get_graph_data(adj):
    adj = torch.from_numpy(adj)

    # in_degree: (n_node) 节点入度
    in_degree = adj.sum(dim=1).view(-1)
    # out_degree: (n_node) 节点出度
    out_degree = adj.sum(dim=0).view(-1)

    return in_degree, out_degree


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    # 数据集
    parser.add_argument('--dataset', type=str, default='PEMS08', help="data file: ['PEMS-BAY', 'PEMS08']")
    # 读入和导出文件名称
    parser.add_argument('--distances_filename', type=str, default='PEMS08.csv',
                        help="CSV file containing sensor distances with three columns: [from, to, distance].")
    parser.add_argument('--sensor_ids_filename', type=str, default='graph_sensor_ids_bay.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--output_pkl_filename', type=str, default='graph_data.pkl',
                        help='Path of the output file.')
    # 节点个数（只有PEMS08数据集会用到）
    parser.add_argument('--num_nodes', type=int, default=170)
    args = parser.parse_args()

    print("Generating graph data of dataset:", args.dataset)

    if args.dataset == 'PEMS-BAY':
        args.distances_filename = 'distances_bay_2017.csv'
        with open(os.path.join(args.dataset, 'raw_data', args.sensor_ids_filename)) as f:
            sensor_ids = f.read().strip().split(',')
    else:
        sensor_ids = list(range(args.num_nodes))

    dis_filename = os.path.join(args.dataset, 'raw_data', args.distances_filename)
    distance_df = pd.read_csv(dis_filename, dtype={'from': 'str', 'to': 'str'})

    _, sensor_id_to_ind, adj = get_adjacency_matrix(distance_df, sensor_ids)
    in_degree, out_degree = get_graph_data(adj)

    # Save to pickle file.
    with open(os.path.join(args.dataset, args.output_pkl_filename), 'wb') as f:
        pickle.dump([adj, in_degree, out_degree], f, protocol=2)
