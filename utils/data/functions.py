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


def min_max_normalize(data): # 归一化
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


# 反归一化
def min_max_denormalize(normalized_data, original_min, original_max):
    return normalized_data * (original_max - original_min) + original_min

def wmin_max_normalize(data):
    # 确保data是一个二维数组
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array")

    # 初始化归一化后的数组
    normalized_data = np.zeros_like(data, dtype=np.float32)

    # 对每一列进行归一化
    for i in range(data.shape[1]):
        col_min = np.min(data[:, i])
        col_max = np.max(data[:, i])
        normalized_data[:, i] = (data[:, i] - col_min) / (col_max - col_min)

    return normalized_data
def combine_data(case, weather):  # 将对应城市的病例与天气组合
    # 初始化一个张量来存储结果 (batch_size, num_node, feat_num)
    result = None
    case_flatten = case.reshape(-1)
    weather_flatten = weather.reshape(-1)
    # 遍历 case 和 weather 来构建新的张量
    i = 0
    j = 0
    while i < case_flatten.shape[0] and j < weather_flatten.shape[0]:
        # 从 case 中取一个值
        case_value = case_flatten[i]
        # 从 weather 中连续取5个值
        weather_values = weather_flatten[j:j + 5]
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


weather = None
weather_train = None
weather_val = None
weather_test = None
def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.7, normalize=True,dataName=""
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _weather = pd.read_csv(os.path.join(script_dir, "..", "..", "data", "weather.csv"), header=None)
    weekend = pd.read_csv(os.path.join(script_dir, "..", "..", "data", "weekend.csv"), header=None)
    _weather = np.array(_weather, dtype=np.float32)

    weather_classes = [np.empty((0, _weather.shape[0]), dtype=np.float32) for _ in range(5)]

    for i in range(0, _weather.shape[1], 1):
        # 拼接当前类别的所有列
        weather_classes[i % 5] = np.vstack([weather_classes[i % 5], _weather[:, i]])


    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        data = min_max_normalize(data)
        weekend = weekend / 52
        _weather = min_max_normalize(_weather)

    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    val_size = int(time_len * 0)
    test_data = data[train_size + val_size:time_len]
    # 病例
    train_X, train_Y, val_X, val_Y ,test_X, test_Y = list(), list(), list(), list(), list(), list()
    # 天气
    _weather_train, _weather_val, _weather_test = list(), list(), list()
    # 组合病例与对应天气
    combine_train,  combine_val, combine_test, = list(), list(), list()
    print("data: train process....")
    for i in range(len(train_data) - seq_len - pre_len):
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

    print("data: test process....")
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
    train_X = np.array(train_X).reshape(len(train_X), seq_len, 47, 1)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X).reshape(len(test_X), seq_len, 47, 1)
    test_Y = np.array(test_Y)
    _train_X = np.concatenate((train_X, combine_train), axis=3)
    _test_X = np.concatenate((test_X, combine_test), axis=3)
    return (_train_X, train_Y,
            _test_X, test_Y)


def generate_torch_datasets(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True,dataName =""
):
    train_X, train_Y, test_X, test_Y = generate_dataset(
        data,
        seq_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )

    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
    )
    return train_dataset, test_dataset
