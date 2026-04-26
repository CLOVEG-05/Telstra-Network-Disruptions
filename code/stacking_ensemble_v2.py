# stacking_ensemble_v2.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

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
    
    # 定义基模型
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )
    
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        tree_method='hist',
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    
    cat_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='MultiClass',
        classes_count=3,
        random_state=42,
        verbose=0
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 定义元学习器
    meta_learner = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    
    # 创建Stacking集成
    stacking_clf = StackingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('xgb', xgb_model),
            ('cat', cat_model),
            ('rf', rf_model)
        ],
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        passthrough=False,
        n_jobs=-1
    )
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 3))
    test_preds = np.zeros((len(X_test), 3))
    scores = []
    
    print("\nTraining Stacking Ensemble Model...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold+1}/5')
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练Stacking模型
        stacking_clf.fit(X_train, y_train)
        
        # 预测
        val_pred = stacking_clf.predict_proba(X_val)
        test_pred = stacking_clf.predict_proba(X_test)
        
        # 评估
        acc = (val_pred.argmax(axis=1) == y_val).mean()
        logloss = log_loss(y_val, val_pred)
        scores.append(logloss)
        print(f'Fold {fold+1} Accuracy: {acc:.4f}, Log Loss: {logloss:.4f}')
        
        oof_preds[val_idx] = val_pred
        test_preds += test_pred / 5
    
    # 计算总体Log Loss
    overall_logloss = log_loss(y, oof_preds)
    print(f'\nMean CV Log Loss: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
    print(f'Overall CV Log Loss: {overall_logloss:.4f}')
    
    # 全量训练
    print("\nTraining on full dataset...")
    stacking_clf.fit(X, y)
    final_test_preds = stacking_clf.predict_proba(X_test)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'predict_0': final_test_preds[:, 0],
        'predict_1': final_test_preds[:, 1],
        'predict_2': final_test_preds[:, 2]
    })
    submission.to_csv(PROCESSED_DIR + 'submission_stacking.csv', index=False)
    print("\nStacking submission saved!")
    print(submission.head(10))
    
    return overall_logloss

if __name__ == '__main__':
    main()