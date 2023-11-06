import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
import yaml
import os

with open('config.yaml', 'r', encoding='utf-8') as F:
    config_yaml = yaml.safe_load(F)
station = config_yaml['Station']['station']


def multi_station(data, train_ratio, valid_ratio):
    T = data.shape[0]
    cols_to_select = [6, 10, 11, 18, 20, 22, 21, 106, 107, 108, 109, 110, 112, 113]  # 沿海城市
    # cols_to_select = [56, 57, 101, 102, 103, 104, 105]
    # cols_to_select = [47, 99, 207, 208, 209, 210]
    data = data[:, cols_to_select, :]
    # data = data[:, 2:10, :]
    train_data = data[0:int(T * train_ratio), :, :]
    train_data = train_data.reshape(-1, data.shape[2])
    valid_data = data[int(T * train_ratio):int(T * (train_ratio + valid_ratio)), station, :]
    valid_data = valid_data.reshape(-1, data.shape[2])
    data_2D = np.concatenate((train_data, valid_data), axis=0)
    data_3d = np.expand_dims(data_2D, axis=1)
    return data_3d


def get_forward_backward_mutation_data(is_model):
    all_dataset = np.load("Original_data/city_data_wind direction.npy")
    # all_dataset = np.flip(all_dataset, 0)
    min_values = all_dataset.min(axis=(0, 1), keepdims=True)
    max_values = all_dataset.max(axis=(0, 1), keepdims=True)

    # 执行最大最小归一化
    all_dataset = (all_dataset - min_values) / (max_values - min_values)
    T = all_dataset.shape[0]
    all_data = all_dataset[:, station, -1]
    # ratio for train or valid
    train_ratio = config_yaml['ratio']['train_ratio']
    valid_ratio = config_yaml['ratio']['valid_ratio']
    flatten_data = multi_station(all_dataset, train_ratio, valid_ratio)
    all_dataset = flatten_data
    dataset = all_dataset[0:int(all_dataset.shape[0] * train_ratio), :, :]
    train_data = all_dataset[0:int(all_dataset.shape[0] * train_ratio), station, -1]  # 训练数据
    valid_dataset = all_dataset[
                    int(all_dataset.shape[0] * train_ratio):int(all_dataset.shape[0] * (train_ratio + valid_ratio)),
                    station, :]  # 验证数据
    # 定义阈值
    threshold = config_yaml['AQI_mutation']['AQI_threshold']
    min_threshold = config_yaml['AQI_mutation']['min_threshold']
    # 滑动窗口的大小，可以根据实际情况调整
    window_size = config_yaml['AQI_mutation']['AQI_window_size']
    # 想要在剧烈变化前后获取的时间序列的长度
    lookback = config_yaml['AQI_mutation']['lookback']
    lookforward = config_yaml['AQI_mutation']['lookforward']
    # 使用滑动窗口计算每个窗口的最大值和最小值
    max_filter = maximum_filter1d(train_data, size=window_size, mode='constant', origin=-2)
    min_filter = minimum_filter1d(train_data, size=window_size, mode='constant', origin=-2)

    # 计算窗口的第一个值与最大值和最小值的差值
    diff_max = train_data[:-window_size + 1] - max_filter[:-window_size + 1]
    diff_min = train_data[:-window_size + 1] - min_filter[:-window_size + 1]

    big_change_data = train_data[:-window_size + 1][
        (np.abs(diff_max) >= threshold / 500) | (np.abs(diff_min) >= threshold / 500)]
    min_val = np.min(big_change_data)
    max_val = np.max(big_change_data)
    interval = (max_val - min_val) / 5
    labels = np.digitize(big_change_data, bins=[min_val + i * interval for i in range(1, 5)])
    counts = np.bincount(labels)

    # 找出剧烈变化点的索引
    anchor_indices = np.where((np.abs(diff_max) >= threshold / 500) & (np.abs(diff_max) <= min_val + 1 * interval) | (
            np.abs(diff_min) >= threshold / 500) & (np.abs(diff_min) <= min_val + 1 * interval))[0]
    change_indices = np.where((np.abs(diff_max) >= threshold / 500) | (np.abs(diff_min) >= threshold / 500))[0]
    unchange_indices = np.where((np.abs(diff_max) < min_threshold / 500) & (np.abs(diff_min) < min_threshold / 500))[
        0]

    # 遍历所有的锚点
    anchor_result = []
    for i in anchor_indices:
        # 如果前后的时间序列长度不足，则跳过
        if i < lookback or i + lookforward >= len(train_data):
            continue
        anchor = dataset[i - lookback + 1:i + 1, station, :]  # 当前突变值与下一个值
        anchor_result.append(anchor)

    # 遍历所有的剧烈变化的点
    change_result = []
    for i in change_indices:
        # 如果前后的时间序列长度不足，则跳过
        if i < lookback or i + lookforward >= len(train_data):
            continue
        change = dataset[i - lookback + 1:i + 1, station, :]  # 当前突变值与下一个值
        change_result.append(change)

    # 遍历所有的非剧烈变化的点
    unchange_result = []
    for i in unchange_indices:
        if i < lookback or i + lookforward >= len(train_data):
            continue
        unchange = dataset[i - lookback + 1:i + 1, station, :]
        unchange_result.append(unchange)

    # 验证阶段的非突变值

    # 转换结果为二维numpy数组
    anchor_result = np.array(anchor_result)
    change_result = np.array(change_result)
    unchange_result = np.array(unchange_result)
    if is_model == "model":
        diff_anchor_result = np.diff(anchor_result, axis=1)
        diff_change_result = np.diff(change_result, axis=1)
        diff_unchange_result = np.diff(unchange_result, axis=1)

        min_indice, min_distance = find_min_distances_indices_cos(diff_anchor_result, diff_change_result)
        # min_indice, min_distance = find_min_distance_indices(diff_anchor_result, diff_change_result)
        # min_indice = np.unique(min_indice.flatten())
        new_diff_change_result = diff_change_result[min_indice.flatten()]
        # max_indice, max_distance = find_max_distances_indices(diff_anchor_result, diff_unchange_result)
        max_indice, max_distance = find_max_distances_indices_cos(diff_anchor_result, diff_unchange_result)
        # max_indice = np.unique(max_indice.flatten())
        new_diff_unchange_result = diff_unchange_result[max_indice.flatten()]
        diff_unchange_result = new_diff_unchange_result
        diff_change_result = new_diff_change_result[:diff_unchange_result.shape[0], :, :]
        diff_anchor_result = diff_anchor_result[:diff_unchange_result.shape[0], :, :]
        is_noise = False
        if is_noise:
            mean = 0
            std = 1
            start_row, end_row = 1000, 2000
            noise = np.random.normal(mean, std, (end_row - start_row, diff_anchor_result.shape[1]))
            diff_anchor_result[start_row:end_row, :, -1] += noise
            diff_change_result[start_row:end_row, :, :] = diff_anchor_result[start_row:end_row, :, :]

    elif is_model == "load_model":
        diff_anchor_result = np.diff(anchor_result, axis=1)
        diff_change_result = np.diff(change_result, axis=1)
        diff_unchange_result = np.diff(unchange_result, axis=1)
    valid_diff_before_result, new_flag_result = get_forward_backward_valid_test_data(
        valid_dataset, window_size, threshold, lookback, lookforward)
    # 差分之后的最后一个值作为预测值
    return diff_anchor_result, diff_change_result, \
        diff_unchange_result, valid_diff_before_result, new_flag_result


def get_forward_backward_valid_test_data(data_multi_feature, window_size, threshold, lookback, lookforward):
    # data_multi_feature = np.flip(data_multi_feature, 0)
    data = data_multi_feature[:, -1]

    # 初始化一个数组来存放标记
    change_mask = np.zeros(data.shape)

    # 对每一个时间点进行操作
    for i in range(data.shape[0] - window_size):
        window_data = data[i:i + window_size]  # 选择后面一段时间内的数据
        current_value = data[i]

        # 防止窗口内所有值为0
        # if np.all(window_data == 0):
        #     continue

        min_value = np.min(window_data)
        max_value = np.max(window_data)

        # 检查是否发生剧烈变化
        if np.abs(current_value - min_value) >= threshold / 500 or np.abs(current_value - max_value) >= threshold / 500:
            change_mask[i] = 1
        sum = change_mask.sum()
    total_before_len = []
    total_after_len = []
    total_flag_len = []
    segment_length = (config_yaml['Batch']['batch_size']) * 3
    data = data[:len(data) // window_size * window_size]
    for i in range(0, len(data) - segment_length + 1, 10):
        # 划分段
        segment = data_multi_feature[i:i + segment_length]
        flag = change_mask[i:i + segment_length]
        # segment = np.flip(segment, 0)
        # flag = np.flip(flag, 0)
        before_result = []
        after_result = []
        flag_result = []
        for j in range(segment.shape[0]):
            if j < (lookback - 1) or j + lookforward > len(segment):
                continue
            before = segment[j - lookback + 1:j + 1, :]  # !
            after = segment[j - 2:j + lookforward - 2, :]
            segment_flag = flag[j]
            flag_result.append(segment_flag)
            before_result.append(before)
            after_result.append(after)
        before_result = np.array(before_result)
        after_result = np.array(after_result)
        flag_result = np.array(flag_result)
        total_before_len.append(before_result)
        total_after_len.append(after_result)
        total_flag_len.append(flag_result)
    total_before_len = np.array(total_before_len)  # 已经构成新的完整数组
    total_after_len = np.array(total_after_len)
    total_flag_len = np.array(total_flag_len)
    ratio = 0.1
    rg = np.random.default_rng(1220)
    indices = rg.choice(total_before_len.shape[0], size=int(ratio * total_before_len.shape[0]), replace=False)

    index = np.zeros(total_before_len.shape[0], dtype=bool)
    index[indices] = True
    total_before_len = total_before_len[index, :]
    total_after_len = total_after_len[index, :]
    total_flag_len = total_flag_len[index, :]
    total_before_len = total_before_len.reshape(total_before_len.shape[0] * total_before_len.shape[1],
                                                total_before_len.shape[2], total_before_len.shape[3])
    total_after_len = total_after_len.reshape(total_after_len.shape[0] * total_after_len.shape[1],
                                              total_after_len.shape[2], total_after_len.shape[3])
    total_flag_len = total_flag_len.reshape(total_flag_len.shape[0] * total_flag_len.shape[1])
    s = total_flag_len.sum()

    # # 计算要选择的元素的数量
    # num_elements = int(ratio * before_result.shape[0])
    #
    # # 随机选择元素的索引
    # indices = np.random.choice(before_result.shape[0], num_elements, replace=False)
    #
    # # 提取选择的元素
    # new_before_result = before_result[indices]
    # new_after_result = after_result[indices]
    # new_flag_result = flag_result[indices]

    diff_before_result = np.diff(total_before_len, axis=1)
    diff_after_result = np.diff(total_after_len, axis=1)
    diff_flip_after_result = np.flip(diff_after_result, axis=1)
    # a = new_flag_result.sum()
    # b = new_flag_result.shape[0]
    # c = a / b
    return diff_before_result, total_flag_len


def find_min_distance_indices(A, B):
    A = A[:, :, -1]
    B = B[:, :, -1]

    top_five_indices = np.zeros((A.shape[0], 1), dtype=int)
    top_five_distances = np.zeros((A.shape[0], 1))

    for i, row_a in enumerate(A):
        distances = np.linalg.norm(B - row_a, axis=1)
        sorted_indices = np.argsort(distances)[1:2]
        top_five_indices[i] = sorted_indices
        top_five_distances[i] = distances[sorted_indices]

    return top_five_indices, top_five_distances


def find_max_distances_indices(A, B):
    A = A[:, :, -1]
    B = B[:, :, -1]
    top_five_indices = np.zeros((A.shape[0], 1), dtype=int)
    top_five_distances = np.zeros((A.shape[0], 1))

    for i, row_a in enumerate(A):
        distances = np.linalg.norm(B - row_a, axis=1)
        sorted_indices = np.argsort(distances)[-1:]
        top_five_indices[i] = sorted_indices
        top_five_distances[i] = distances[sorted_indices][::-1]
    return top_five_indices, top_five_distances


def find_max_distances_indices_cos(A, B):
    A = A[:, :, -1]
    B = B[:, :, -1]
    top_five_indices = np.zeros((A.shape[0], 1), dtype=int)
    top_five_distances = np.zeros((A.shape[0], 1))

    for i, row_a in enumerate(A):
        # 计算A和B之间的余弦相似度
        dot_product = B @ row_a
        norm_a = np.linalg.norm(row_a)
        norm_b = np.linalg.norm(B, axis=1)
        cosine_similarity = dot_product / (norm_a * norm_b)

        # 计算余弦距离
        distances = 1 - cosine_similarity

        sorted_indices = np.argsort(distances)[-1:]
        top_five_indices[i] = sorted_indices
        top_five_distances[i] = distances[sorted_indices]
    return top_five_indices, top_five_distances


def find_min_distances_indices_cos(A, B):
    A = A[:, :, -1]
    B = B[:, :, -1]
    top_five_indices = np.zeros((A.shape[0], 1), dtype=int)
    top_five_distances = np.zeros((A.shape[0], 1))

    for i, row_a in enumerate(A):
        # 计算A和B之间的余弦相似度
        dot_product = B @ row_a
        norm_a = np.linalg.norm(row_a)
        norm_b = np.linalg.norm(B, axis=1)
        cosine_similarity = dot_product / (norm_a * norm_b)

        # 计算余弦距离
        distances = 1 - cosine_similarity

        sorted_indices = np.argsort(distances)[1:2]
        top_five_indices[i] = sorted_indices
        top_five_distances[i] = distances[sorted_indices]
    return top_five_indices, top_five_distances
