import pandas as pd
import numpy as np

# 数据文件路径
DATA_DIR = '..\\data set\\'
PROCESSED_DIR = '..\\processed data\\'

# 读取数据
def load_data():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR + 'train.csv')
    test = pd.read_csv(DATA_DIR + 'test.csv')
    event_type = pd.read_csv(DATA_DIR + 'event_type.csv')
    log_feature = pd.read_csv(DATA_DIR + 'log_feature.csv')
    resource_type = pd.read_csv(DATA_DIR + 'resource_type.csv')
    severity_type = pd.read_csv(DATA_DIR + 'severity_type.csv')
    
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    print(f"Event type data shape: {event_type.shape}")
    print(f"Log feature data shape: {log_feature.shape}")
    print(f"Resource type data shape: {resource_type.shape}")
    print(f"Severity type data shape: {severity_type.shape}")
    
    return train, test, event_type, log_feature, resource_type, severity_type

# 处理事件类型数据
def process_event_type(event_type):
    # 统计每个id的事件类型数量
    event_count = event_type.groupby('id').size().reset_index(name='event_count')
    
    # 事件类型编码
    event_dummies = pd.get_dummies(event_type, columns=['event_type'])
    event_features = event_dummies.groupby('id').sum().reset_index()
    
    return event_count, event_features

# 处理日志特征数据
def process_log_feature(log_feature):
    # 统计每个id的日志特征数量
    log_count = log_feature.groupby('id').size().reset_index(name='log_count')
    
    # 统计每个id的总volume
    log_volume = log_feature.groupby('id')['volume'].sum().reset_index(name='total_volume')
    
    # 日志特征编码
    log_dummies = pd.get_dummies(log_feature, columns=['log_feature'])
    log_features = log_dummies.groupby('id').sum().reset_index()
    
    return log_count, log_volume, log_features

# 处理资源类型数据
def process_resource_type(resource_type):
    # 资源类型编码
    resource_dummies = pd.get_dummies(resource_type, columns=['resource_type'])
    resource_features = resource_dummies.groupby('id').sum().reset_index()
    
    return resource_features

# 处理严重程度数据
def process_severity_type(severity_type):
    # 严重程度编码
    severity_dummies = pd.get_dummies(severity_type, columns=['severity_type'])
    return severity_dummies

# 处理位置数据
def process_location(data):
    # 提取位置编号
    data['location_id'] = data['location'].apply(lambda x: int(x.split(' ')[1]))
    return data

# 合并所有数据
def merge_data(train, test, event_count, event_features, log_count, log_volume, log_features, resource_features, severity_dummies):
    print("Merging data...")
    
    # 合并训练数据
    train_merged = train.copy()
    train_merged = pd.merge(train_merged, event_count, on='id', how='left')
    train_merged = pd.merge(train_merged, event_features, on='id', how='left')
    train_merged = pd.merge(train_merged, log_count, on='id', how='left')
    train_merged = pd.merge(train_merged, log_volume, on='id', how='left')
    train_merged = pd.merge(train_merged, log_features, on='id', how='left')
    train_merged = pd.merge(train_merged, resource_features, on='id', how='left')
    train_merged = pd.merge(train_merged, severity_dummies, on='id', how='left')
    train_merged = process_location(train_merged)
    
    # 合并测试数据
    test_merged = test.copy()
    test_merged = pd.merge(test_merged, event_count, on='id', how='left')
    test_merged = pd.merge(test_merged, event_features, on='id', how='left')
    test_merged = pd.merge(test_merged, log_count, on='id', how='left')
    test_merged = pd.merge(test_merged, log_volume, on='id', how='left')
    test_merged = pd.merge(test_merged, log_features, on='id', how='left')
    test_merged = pd.merge(test_merged, resource_features, on='id', how='left')
    test_merged = pd.merge(test_merged, severity_dummies, on='id', how='left')
    test_merged = process_location(test_merged)
    
    # 填充缺失值
    train_merged = train_merged.fillna(0)
    test_merged = test_merged.fillna(0)
    
    print(f"Merged training data shape: {train_merged.shape}")
    print(f"Merged test data shape: {test_merged.shape}")
    
    return train_merged, test_merged

# 保存处理后的数据
def save_processed_data(train_merged, test_merged):
    output_dir = PROCESSED_DIR
    train_merged.to_csv(output_dir + 'train_processed.csv', index=False)
    test_merged.to_csv(output_dir + 'test_processed.csv', index=False)
    print("Processed data saved")

# 主函数
def main():
    # 加载数据
    train, test, event_type, log_feature, resource_type, severity_type = load_data()
    
    # 处理各类型数据
    event_count, event_features = process_event_type(event_type)
    log_count, log_volume, log_features = process_log_feature(log_feature)
    resource_features = process_resource_type(resource_type)
    severity_dummies = process_severity_type(severity_type)
    
    # 合并数据
    train_merged, test_merged = merge_data(
        train, test, event_count, event_features, log_count, log_volume, log_features, resource_features, severity_dummies
    )
    
    # 保存处理后的数据
    save_processed_data(train_merged, test_merged)
    
    print("Data preparation completed!")

if __name__ == '__main__':
    main()