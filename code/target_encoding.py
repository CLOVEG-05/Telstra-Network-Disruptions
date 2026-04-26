# target_encoding.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

DATA_DIR = '../data set/'
PROCESSED_DIR = '../processed data/'

def target_encode(df, target, categorical_cols, cv=5):
    """
    对类别特征进行目标编码，使用交叉验证避免数据泄露
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    for col in categorical_cols:
        df[f'{col}_target_encoded'] = 0.0
        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]
            
            means = train_df.groupby(col)[target].mean()
            df.loc[val_df.index, f'{col}_target_encoded'] = val_df[col].map(means).fillna(train_df[target].mean())
    
    return df

def main():
    print("Loading data...")
    train = pd.read_csv(PROCESSED_DIR + 'train_v2.csv')
    test = pd.read_csv(PROCESSED_DIR + 'test_v2.csv')
    
    print(f"Train shape before: {train.shape}")
    print(f"Test shape before: {test.shape}")
    
    # 对location进行目标编码
    print("Applying target encoding to location_id...")
    train_processed = target_encode(train.copy(), 'fault_severity', ['location_id'])
    
    # 对测试集使用训练集的均值
    location_means = train.groupby('location_id')['fault_severity'].mean()
    test_processed = test.copy()
    test_processed['location_id_target_encoded'] = test_processed['location_id'].map(location_means).fillna(train['fault_severity'].mean())
    
    print(f"Train shape after: {train_processed.shape}")
    print(f"Test shape after: {test_processed.shape}")
    
    # 保存
    train_processed.to_csv(PROCESSED_DIR + 'train_target_encoded.csv', index=False)
    test_processed.to_csv(PROCESSED_DIR + 'test_target_encoded.csv', index=False)
    
    print("Target encoding completed!")
    print(f"New features added: location_id_target_encoded")

if __name__ == '__main__':
    main()