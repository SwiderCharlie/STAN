import numpy as np
import os
import pickle


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0

        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            del x_padding, y_padding

        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)

        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys, = xs[permutation], ys[permutation]

        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield x_i, y_i
                self.current_ind += 1

        return _wrapper()


def load_dataset(dataset_dir, batch_size, flag, scale=True):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())

    # 标准化数据
    if scale:
        data['x_' + flag][..., 0] = scaler.transform(data['x_' + flag][..., 0])
        data['y_' + flag][..., 0] = scaler.transform(data['y_' + flag][..., 0])

    if flag == 'train':
        dataloader = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    elif flag == 'val':
        dataloader = DataLoader(data['x_val'], data['y_val'], batch_size, shuffle=False)
    else:
        dataloader = DataLoader(data['x_test'], data['y_test'], 1, shuffle=False)

    return dataloader, scaler


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph_data(pkl_filename):
    adj, in_degree, out_degree = load_pickle(pkl_filename)
    return adj, in_degree, out_degree
