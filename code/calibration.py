# calibration.py
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import lightgbm as lgb

PROCESSED_DIR = '../processed data/'

def main():
    print("Loading data...")
    train = pd.read_csv(PROCESSED_DIR + 'train_advanced.csv')
    test = pd.read_csv(PROCESSED_DIR + 'test_advanced.csv')
    
    X = train.drop(['id', 'location', 'fault_severity'], axis=1)
    y = train['fault_severity']
    X_test = test.drop(['id', 'location'], axis=1)
    test_ids = test['id']
    
    print(f"Training data shape: {X.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # 定义LightGBM模型
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    
    print("\nTraining calibrated LightGBM...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'Fold {fold+1}/5')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 使用Platt缩放进行概率校准
        calibrated_model = CalibratedClassifierCV(
            lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=7,
                learning_rate=0.05,
                num_leaves=63,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            ),
            method='sigmoid',
            cv=3
        )
        
        calibrated_model.fit(X_train, y_train)
        
        # 预测
        val_pred = calibrated_model.predict_proba(X_val)
        test_pred = calibrated_model.predict_proba(X_test)
        
        # 计算Log Loss
        val_logloss = log_loss(y_val, val_pred)
        print(f'Fold {fold+1} Log Loss: {val_logloss:.4f}')
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
    
    # 计算总体Log Loss
    overall_logloss = log_loss(y, oof_preds)
    print(f'\nOverall CV Log Loss: {overall_logloss:.4f}')
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'predict_0': test_preds[:, 0],
        'predict_1': test_preds[:, 1],
        'predict_2': test_preds[:, 2]
    })
    submission.to_csv(PROCESSED_DIR + 'submission_calibrated.csv', index=False)
    print("\nCalibrated submission saved!")
    print(submission.head(10))
    
    return overall_logloss

if __name__ == '__main__':
    main()