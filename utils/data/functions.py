import numpy as np
import pandas as pd
import torch
import os

def load_features(feat_path, dtype=np.float32):
    feat_df = pd.read_csv(feat_path, header=None)
    feat = np.array(feat_df, dtype=dtype)
    return feat


def load_adjacency_matrix(adj_path, dtype=np.float32):
    adj_df = pd.read_csv(adj_path, header=None)
    adj = np.array(adj_df, dtype=dtype)
    return adj


def min_max_normalize(data, dataName): # 归一化
    if dataName == 'jphfmd':
        min_val = np.min(data)
        max_val = np.max(data)
    else:
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)


# 反归一化
def min_max_denormalize(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min


def combine_data(case, weather):  # 将对应城市的病例与天气组合
    # 初始化一个张量来存储结果 (batch_size, num_node, feat_num)
    result = None
    case_flatten = case.reshape(-1)
    weather_flatten = weather.reshape(-1)
    weather_f_num = int(weather.shape[1] / case.shape[1])
    # 遍历 case 和 weather 来构建新的张量
    i = 0
    j = 0
    while i < case_flatten.shape[0] and j < weather_flatten.shape[0]:
        # 从 case 中取一个值
        case_value = case_flatten[i]
        # 从 weather 中连续取weather_f_num个值
        weather_values = weather_flatten[j:j + weather_f_num]
        if case_value.ndim == 0:
            case_value = case_value.reshape(1)
        if weather_values.ndim == 0:
            weather_values = weather_values.reshape(1)

        # 将 case_value 和 weather_values 组合
        combined_values = np.concatenate((case_value, weather_values))
        # 将组合后的序列存储到 result 中
        if result is None:
            result = combined_values
        else:
            result = np.concatenate((result, combined_values))
        i += 1
        j += 5

    result = result.reshape(case.shape[0], case.shape[1], -1)
    return result


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.7, normalize=True, dataName='jphfmd'
):
    """
    :param data: feature matrix with hfmd cases
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]

    train_size = int(time_len * split_ratio)

    train_X, train_Y, val_X, val_Y, test_X, test_Y = list(), list(), list(), list(), list(), list()
    if dataName == 'jphfmd':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(script_dir, r"..\..\data", "weather.csv")
        _weather = pd.read_csv(relative_path, header=None)
        relative_path = os.path.join(script_dir, r"..\..\data", "weekend.csv")
        weekend = pd.read_csv(relative_path, header=None)
        _weather = np.array(_weather, dtype=np.float32)
        if normalize:
            data = min_max_normalize(data, dataName)
            weekend = weekend / 52
            _weather = min_max_normalize(_weather, dataName)
        train_data = data[:train_size]
        test_data = data[train_size:time_len]
        _weather_train, _weather_val, _weather_test = list(), list(), list()
        combine_train, combine_val, combine_test, = list(), list(), list()
        for i in range(0, len(train_data) - seq_len - pre_len):
            x = np.array(train_data[i: i + seq_len])
            w = np.array(_weather[i:i + seq_len])
            train_X.append(x)
            _weather_train.append(w)

            com = combine_data(x, w)
            week = np.array(weekend[i:i + seq_len])
            week = week.reshape(week.shape[0], week.shape[1], 1)
            com = np.concatenate((com, week), axis=2)
            combine_train.append(com)
            train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len]))

        for i in range(len(test_data) - seq_len - pre_len):
            x = np.array(test_data[i: i + seq_len])
            w = np.array(_weather[i: i + seq_len])

            test_X.append(x)
            _weather_test.append(w)

            com = combine_data(x, w)
            week = np.array(weekend[i:i + seq_len])
            week = week.reshape(week.shape[0], week.shape[1], 1)
            com = np.concatenate((com, week), axis=2)
            combine_test.append(com)

            test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len]))

            combine_train = np.array(combine_train)
            combine_test = np.array(combine_test)
            train_Y = np.array(train_Y)
            test_Y = np.array(test_Y)
            _train_X = combine_train
            _test_X = combine_test
            return (_train_X, train_Y,
                    _test_X, test_Y)
    else:
        if normalize:
            data = min_max_normalize(data, dataName)
        train_data = data[:train_size]
        test_data = data[train_size:time_len]
        for i in range(len(train_data) - seq_len - pre_len):
            x = np.array(train_data[i: i + seq_len])
            train_X.append(x)
            train_Y.append(np.array(train_data[i + seq_len: i + seq_len + pre_len, -1]))
        for i in range(len(test_data) - seq_len - pre_len):
            x = np.array(test_data[i: i + seq_len])
            test_X.append(x)
            test_Y.append(np.array(test_data[i + seq_len: i + seq_len + pre_len, -1]))

        train_Y = np.array(train_Y)
        test_Y = np.array(test_Y)

        return train_X, train_Y, test_X, test_Y


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True, dataName='jphfmd'
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
        dataName=dataName
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
