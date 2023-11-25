import argparse
import os
import time
import numpy as np
import pandas as pd


# 将原始数据集整理成多个样本数据
def generate_graph_seq2seq_io_data(data, x_offsets, y_offsets):
    num_samples = data.shape[0]
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


# 生成PEMS08数据集拼接的时间戳信息
def timestamp_pems08(dim, add_time_of_day, add_day_of_week):
    time_of_day, day_of_week = None, None

    # time_of_day
    if add_time_of_day:
        if dim == 295:  # time_of_dim dimension: 288
            time_of_day = np.tile(np.eye(288, 288), [62, 1])
            time_of_day = np.tile(time_of_day, [170, 1, 1]).transpose((1, 0, 2))

        else:    # time_of_dim dimension: 1
            time = np.arange(288) / 288
            time_of_day = []
            for t in time:
                time_of_day.append(np.ones((170, 1)) * t)
            time_of_day = np.stack(time_of_day, axis=0)
            time_of_day = np.tile(time_of_day, [62, 1, 1])

    # day_of_week
    if add_day_of_week:
        date = pd.date_range("7/1/2016 00:00:00", "8/31/2016 23:55:00", periods=288 * 62)
        if dim == 2:  # day_of_week dimension: 1
            day_of_week = np.array(date.day_of_week)
            day_of_week = np.expand_dims(day_of_week, 1).repeat(170, 1)
            day_of_week = np.expand_dims(day_of_week, 2)

        else:  # day_of_week dimension: 7
            day_of_week = np.zeros(shape=(288 * 62, 170, 7))
            day_of_week[np.arange(288 * 62), :, date.dayofweek] = 1

    return time_of_day,  day_of_week


# 生成PEMS-BAY数据集拼接的时间戳信息
def timestamp_bay(date, dim, add_time_of_day, add_day_of_week):
    time_of_day, day_of_week = None, None

    # time_of_day
    if add_time_of_day:
        if dim == 295:  # time_of_dim dimension: 288
            time_of_day = np.tile(np.expand_dims(np.array(list(range(288))), axis=-1), [len(date) // 288 + 1, 1])
            time_of_day = np.tile(time_of_day[:len(date)], [325, 1, 1]).transpose((1, 0, 2))

        else:  # time_of_dim dimension: 1
            time_of_day = (date.values - date.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_of_day = np.tile(time_of_day, [1, 325, 1]).transpose((2, 1, 0))

    # day_of_week
    if add_day_of_week:
        if dim == 2:  # day_of_week dimension: 1
            day_of_week = np.array(date.dayofweek)
            day_of_week = np.expand_dims(day_of_week, 1).repeat(325, 1)
            day_of_week = np.expand_dims(day_of_week, 2)

        else:  # day_of_week dimension: 7
            day_of_week = np.zeros(shape=(len(date), 325, 7))
            day_of_week[np.arange(len(date)), :, date.dayofweek] = 1

    return time_of_day, day_of_week


def generate_train_val_test(args):
    pred_len = args.pred_len

    if args.data == 'PEMS08':  # PEMS08 数据集
        data = np.load(args.traffic_filename)['data'][:, :, 0:1]  # ndarray, (T, N, D)
        data_list = [data.astype(np.float32)]
        del data
        # time_of_day: (T, N, D_time)
        # day_of_week: (T, N, D_day)
        time_of_day, day_of_week = timestamp_pems08(args.timestamp_dim, args.add_time_of_day, args.add_day_of_week)

    else:  # PEMS-BAY 数据集
        df = pd.read_hdf(args.traffic_filename)  # DataFrame, (T, N)
        num_samples, _ = df.shape
        data = np.expand_dims(df.values, axis=-1)  # ndarray, (T, N, 1)
        data_list = [data.astype(np.float32)]
        del data
        time_of_day, day_of_week = timestamp_bay(df.index, args.timestamp_dim, args.add_time_of_day, args.add_day_of_week)

    if time_of_day is not None:
        data_list.append(time_of_day.astype(np.float32))
        del time_of_day
    if day_of_week is not None:
        data_list.append(day_of_week.astype(np.float32))
        del day_of_week
    data = np.concatenate(data_list, axis=-1)
    del data_list

    x_offsets = np.sort(np.concatenate((np.arange(-pred_len + 1, 1, 1),)))
    y_offsets = np.sort(np.arange(1, pred_len + 1, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(data, x_offsets=x_offsets, y_offsets=y_offsets)
    del data
    y = y[:, :, :, 0, None]
    print("x shape: ", x.shape, ", y shape: ", y.shape)

    # 划分训练集、验证集、测试集（7:1:2）
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # 训练集
    x_train, y_train = x[:num_train], y[:num_train]
    # 验证集
    x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
    # 测试集
    x_test, y_test = x[-num_test:], y[-num_test:]
    del x, y

    # 保存数据
    print("saving data ...")
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.data, "%s.npz" % cat), x=_x, y=_y)


if __name__ == "__main__":
    # 设置参数
    parser = argparse.ArgumentParser()
    # 选择数据集对应历史交通数据文件
    parser.add_argument('--traffic_filename', type=str,
                        default='PEMS08/raw_data/PEMS08.npz',
                        help="Raw traffic readings: ['PEMS-BAY/raw_data/pems-bay.h5', 'PEMS08/raw_data/PEMS08.npz']")
    # 历史和预测时间序列长度
    parser.add_argument('--pred_len', type=int, default=12)
    # 是否添加time-of-day时间戳信息
    parser.add_argument('--add_time_of_day', type=bool, default=True)
    # 是否添加day-of-week时间戳信息
    parser.add_argument('--add_day_of_week', type=bool, default=True)
    # 拼接的时间戳信息维度
    parser.add_argument('--timestamp_dim', type=int, default=8, help="[1+1=2, 1+7=8, 288+7=295]")
    args = parser.parse_args()
    args.data = args.traffic_filename.split('/')[0]

    print("Generating training data of dataset:", args.data)
    start_time = time.time()
    generate_train_val_test(args)
    tol_time = int(time.time() - start_time)
    print("Total Time：{}:{:0>2d}".format(tol_time // 60, tol_time % 60))