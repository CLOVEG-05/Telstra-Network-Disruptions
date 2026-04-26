# advanced_features.py
import pandas as pd
import numpy as np

PROCESSED_DIR = '../processed data/'

def create_advanced_features(df):
    """
    创建更多高级特征交叉
    """
    print("Creating advanced features...")
    
    # 体积相关交互
    if 'total_volume' in df.columns and 'event_count' in df.columns:
        df['volume_per_event'] = df['total_volume'] / (df['event_count'] + 1)
        df['volume_event_product'] = df['total_volume'] * df['event_count']
    
    # 多样性与强度的交互
    if 'log_diversity' in df.columns and 'total_volume' in df.columns:
        df['diversity_volume_ratio'] = df['log_diversity'] / (df['total_volume'] + 1)
        df['diversity_volume_product'] = df['log_diversity'] * df['total_volume']
    
    # 位置与事件的交互
    if 'location_cluster' in df.columns and 'event_count' in df.columns:
        df['location_event'] = df['location_cluster'] * df['event_count']
    
    # 日志特征相关交互
    if 'log_count' in df.columns and 'event_count' in df.columns:
        df['log_event_ratio'] = df['log_count'] / (df['event_count'] + 1)
        df['log_event_sum'] = df['log_count'] + df['event_count']
    
    # 资源类型与严重程度的交互
    if 'total_volume' in df.columns and 'log_count' in df.columns:
        df['volume_log_ratio'] = df['total_volume'] / (df['log_count'] + 1)
    
    # 事件类型多样性
    if 'event_count' in df.columns and 'log_diversity' in df.columns:
        df['event_log_diversity'] = df['event_count'] * df['log_diversity']
    
    print(f"Created {len([c for c in df.columns if c not in ['id', 'location', 'fault_severity']])} features")
    
    return df

def main():
    print("Loading data...")
    train = pd.read_csv(PROCESSED_DIR + 'train_target_encoded.csv')
    test = pd.read_csv(PROCESSED_DIR + 'test_target_encoded.csv')
    
    print(f"Train shape before: {train.shape}")
    print(f"Test shape before: {test.shape}")
    
    # 创建高级特征
    train_processed = create_advanced_features(train)
    test_processed = create_advanced_features(test)
    
    print(f"Train shape after: {train_processed.shape}")
    print(f"Test shape after: {test_processed.shape}")
    
    # 填充缺失值
    train_processed = train_processed.fillna(0)
    test_processed = test_processed.fillna(0)
    
    # 保存
    train_processed.to_csv(PROCESSED_DIR + 'train_advanced.csv', index=False)
    test_processed.to_csv(PROCESSED_DIR + 'test_advanced.csv', index=False)
    
    print("Advanced feature engineering completed!")

if __name__ == '__main__':
    main()