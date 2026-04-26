# feature_engineering_v2.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split

DATA_DIR = '../data set/'
PROCESSED_DIR = '../processed data/'

def load_raw_data():
    train = pd.read_csv(DATA_DIR + 'train.csv')
    test = pd.read_csv(DATA_DIR + 'test.csv')
    event_type = pd.read_csv(DATA_DIR + 'event_type.csv')
    log_feature = pd.read_csv(DATA_DIR + 'log_feature.csv')
    resource_type = pd.read_csv(DATA_DIR + 'resource_type.csv')
    severity_type = pd.read_csv(DATA_DIR + 'severity_type.csv')
    
    return train, test, event_type, log_feature, resource_type, severity_type

def create_interaction_features(df):
    """创建交互特征"""
    # 事件和日志的交互
    if 'event_count' in df.columns and 'log_count' in df.columns:
        df['event_log_ratio'] = df['event_count'] / (df['log_count'] + 1)
        df['event_log_product'] = df['event_count'] * df['log_count']
    
    # 位置聚类
    if 'location_id' in df.columns:
        df['location_cluster'] = df['location_id'] // 100
    
    return df

def create_statistical_features(df, log_feature):
    """创建统计特征"""
    # 日志特征统计
    log_stats = log_feature.groupby('id')['volume'].agg(['mean', 'std', 'sum', 'max', 'min']).reset_index()
    log_stats.columns = ['id', 'volume_mean', 'volume_std', 'volume_total', 'volume_max', 'volume_min']
    df = df.merge(log_stats, on='id', how='left')
    
    # 事件类型多样性
    event_diversity = log_feature.groupby('id')['log_feature'].nunique().reset_index(name='log_diversity')
    df = df.merge(event_diversity, on='id', how='left')
    
    return df

def main():
    # 加载数据
    train, test, event_type, log_feature, resource_type, severity_type = load_raw_data()
    
    # 加载已处理的数据
    train_processed = pd.read_csv(PROCESSED_DIR + 'train_processed.csv')
    test_processed = pd.read_csv(PROCESSED_DIR + 'test_processed.csv')
    
    # 添加统计特征
    train_processed = create_statistical_features(train_processed, log_feature)
    test_processed = create_statistical_features(test_processed, log_feature)
    
    # 添加交互特征
    train_processed = create_interaction_features(train_processed)
    test_processed = create_interaction_features(test_processed)
    
    # 填充缺失值
    train_processed = train_processed.fillna(0)
    test_processed = test_processed.fillna(0)
    
    # 保存
    train_processed.to_csv(PROCESSED_DIR + 'train_v2.csv', index=False)
    test_processed.to_csv(PROCESSED_DIR + 'test_v2.csv', index=False)
    
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    print("Feature engineering v2 completed!")

if __name__ == '__main__':
    main()