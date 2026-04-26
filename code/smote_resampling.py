# smote_resampling.py
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

PROCESSED_DIR = '../processed data/'

def apply_smote(X_train, y_train):
    """应用SMOTE数据增强"""
    print(f"原始类别分布: {np.bincount(y_train.astype(int))}")
    
    # 组合采样策略 - SMOTE只用于过采样少数类
    over = SMOTE(sampling_strategy={2: 1500}, random_state=42)  # 过采样类别2到1500
    under = RandomUnderSampler(sampling_strategy={0: 4000}, random_state=42)  # 欠采样类别0到4000
    
    steps = [('over', over), ('under', under)]
    pipeline = Pipeline(steps=steps)
    
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    
    print(f"重采样后分布: {np.bincount(y_resampled.astype(int))}")
    
    return X_resampled, y_resampled

def main():
    # 加载数据
    train = pd.read_csv(PROCESSED_DIR + 'train_v2.csv')

    X = train.drop(['id', 'location', 'fault_severity'], axis=1).values
    y = train['fault_severity'].values
    
    # 应用SMOTE
    X_resampled, y_resampled = apply_smote(X, y)
    
    # 保存
    resampled_df = pd.DataFrame(X_resampled, columns=train.drop(['id', 'location', 'fault_severity'], axis=1).columns)
    resampled_df['fault_severity'] = y_resampled
    resampled_df.to_csv(PROCESSED_DIR + 'train_smote.csv', index=False)
    
    print("SMOTE completed!")

if __name__ == '__main__':
    main()