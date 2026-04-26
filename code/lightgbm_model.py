# lightgbm_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 数据路径
PROCESSED_DIR = '../processed data/'
MODEL_DIR = '../model/'
VIS_DIR = '../visualization/'

def load_data():
    train = pd.read_csv(PROCESSED_DIR + 'train_processed.csv')
    test = pd.read_csv(PROCESSED_DIR + 'test_processed.csv')
    
    X = train.drop(['id', 'location', 'fault_severity'], axis=1)
    y = train['fault_severity']
    X_test = test.drop(['id', 'location'], axis=1)
    test_ids = test['id']
    
    return X, y, X_test, test_ids

def train_lightgbm(X, y, X_test, test_ids):
    # GPU加速配置
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'device': 'gpu',              # GPU加速
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'learning_rate': 0.05,
        'max_depth': 7,
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'class_weight': 'balanced'    # 处理类别不平衡
    }
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold+1}/5')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # 训练模型
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # 预测
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
        
        # 评估
        val_pred_class = val_pred.argmax(axis=1)
        acc = accuracy_score(y_val, val_pred_class)
        scores.append(acc)
        print(f'Fold {fold+1} Accuracy: {acc:.4f}')
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
    
    print(f'\nMean CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    
    return test_preds, scores

def create_submission(test_ids, predictions):
    submission = pd.DataFrame({
        'id': test_ids,
        'predict_0': predictions[:, 0],
        'predict_1': predictions[:, 1],
        'predict_2': predictions[:, 2]
    })
    submission.to_csv(PROCESSED_DIR + 'submission_lgb.csv', index=False)
    print("\nSubmission saved!")
    print(submission.head(10))
    return submission

def main():
    print("Loading data...")
    X, y, X_test, test_ids = load_data()
    
    print("\nTraining LightGBM with GPU...")
    predictions, scores = train_lightgbm(X, y, X_test, test_ids)
    
    print("\nCreating submission...")
    create_submission(test_ids, predictions)
    
    print("\nDone!")

if __name__ == '__main__':
    main()