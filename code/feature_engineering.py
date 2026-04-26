import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# 数据文件路径
DATA_DIR = '..\\data set\\'
PROCESSED_DIR = '..\\processed data\\'

# 加载处理后的数据
def load_processed_data():
    print("Loading processed data...")
    train = pd.read_csv(PROCESSED_DIR + 'train_processed.csv')
    test = pd.read_csv(PROCESSED_DIR + 'test_processed.csv')
    
    print(f"Training data shape: {train.shape}")
    print(f"Test data shape: {test.shape}")
    
    return train, test

# 特征选择
def feature_selection(X_train, y_train, k=100):
    print(f"Selecting top {k} most relevant features...")
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    
    # 获取选中的特征名称
    selected_features = X_train.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} features")
    print(f"Top 10 selected features: {list(selected_features[:10])}")
    
    return X_train_selected, selected_features, selector

# 数据预处理
def preprocess_data():
    # 加载数据
    train, test = load_processed_data()
    
    # 分离特征和目标变量
    X_train = train.drop(['id', 'location', 'fault_severity'], axis=1)
    y_train = train['fault_severity']
    X_test = test.drop(['id', 'location'], axis=1)
    test_ids = test['id']
    
    print(f"Original features: {X_train.shape[1]}")
    
    # 特征选择
    X_train_selected, selected_features, selector = feature_selection(X_train, y_train, k=100)
    
    # 应用特征选择到测试集
    X_test_selected = selector.transform(X_test)
    
    # 特征标准化
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # 数据分割
    print("Splitting train and validation sets...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set shape: {X_train_final.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return (
        X_train_final, y_train_final, X_val, y_val, 
        X_test_scaled, test_ids, selected_features, scaler
    )

# 保存预处理后的数据
def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, test_ids, selected_features):
    output_dir = PROCESSED_DIR
    
    # 保存训练数据
    train_data = pd.DataFrame(X_train, columns=selected_features)
    train_data['fault_severity'] = y_train.reset_index(drop=True)
    train_data.to_csv(output_dir + 'train_preprocessed.csv', index=False)
    
    # 保存验证数据
    val_data = pd.DataFrame(X_val, columns=selected_features)
    val_data['fault_severity'] = y_val.reset_index(drop=True)
    val_data.to_csv(output_dir + 'val_preprocessed.csv', index=False)
    
    # 保存测试数据
    test_data = pd.DataFrame(X_test, columns=selected_features)
    test_data['id'] = test_ids.reset_index(drop=True)
    test_data.to_csv(output_dir + 'test_preprocessed.csv', index=False)
    
    print("Preprocessed data saved")

# 主函数
def main():
    # 预处理数据
    X_train, y_train, X_val, y_val, X_test, test_ids, selected_features, scaler = preprocess_data()
    
    # 保存预处理后的数据
    save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, test_ids, selected_features)
    
    print("Data preprocessing and feature engineering completed!")

if __name__ == '__main__':
    main()