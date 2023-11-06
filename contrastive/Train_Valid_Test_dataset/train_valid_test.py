def train_valid_test(diff_before_result, diff_flip_after_result,
                     train_ratio,
                     valid_ratio):
    N, L, C = diff_before_result.shape[0], diff_before_result.shape[1], diff_before_result.shape[2]
    # 训练数据
    train_before_data_dataset = diff_before_result[0:int(N * train_ratio), :, :]
    train_after_data_dataset = diff_flip_after_result[0:int(N * train_ratio), :, :]
    # 验证数据
    valid_before_data_dataset = diff_before_result[int(N * train_ratio):int(N * (train_ratio + valid_ratio)), :, :]
    valid_after_data_dataset = diff_flip_after_result[int(N * train_ratio):int(N * (train_ratio + valid_ratio)), :, :]
    # 测试数据
    test_before_data_dataset = diff_before_result[int(N * (train_ratio + valid_ratio)):, :, :]
    test_after_data_dataset = diff_flip_after_result[int(N * (train_ratio + valid_ratio)):, :, :]
    return train_before_data_dataset, train_after_data_dataset, \
        valid_before_data_dataset, valid_after_data_dataset, \
        test_before_data_dataset, test_after_data_dataset
