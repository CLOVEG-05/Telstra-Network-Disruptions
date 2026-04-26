# ensemble_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DIR = '../processed data/'

def load_data():
    train = pd.read_csv(PROCESSED_DIR + 'train_smote.csv')
    test = pd.read_csv(PROCESSED_DIR + 'test_v2.csv')

    X = train.drop(['fault_severity'], axis=1)
    y = train['fault_severity']
    X_test = test.drop(['id', 'location'], axis=1)
    test_ids = test['id']

    return X, y, X_test, test_ids

def train_ensemble(X, y, X_test, test_ids):
    # 定义模型
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

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 软投票集成
    ensemble = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('xgb', xgb_model),
            ('rf', rf_model)
        ],
        voting='soft',
        weights=[0.4, 0.4, 0.2]  # LGB和XGB权重更高
    )

    # 5折交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    test_preds = np.zeros((len(X_test), 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f'\nFold {fold+1}/5')

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 训练
        ensemble.fit(X_train, y_train)

        # 预测
        val_pred = ensemble.predict(X_val)
        test_pred = ensemble.predict_proba(X_test)

        # 评估
        acc = accuracy_score(y_val, val_pred)
        scores.append(acc)
        print(f'Fold {fold+1} Accuracy: {acc:.4f}')

        test_preds += test_pred / 5

    print(f'\nMean CV Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

    # 全量训练
    ensemble.fit(X, y)
    final_test_preds = ensemble.predict_proba(X_test)

    return final_test_preds, scores

def create_submission(test_ids, predictions):
    submission = pd.DataFrame({
        'id': test_ids,
        'predict_0': predictions[:, 0],
        'predict_1': predictions[:, 1],
        'predict_2': predictions[:, 2]
    })
    submission.to_csv(PROCESSED_DIR + 'submission_final.csv', index=False)
    print("\nFinal submission saved!")
    print(submission.head(10))
    return submission

def main():
    print("Loading data...")
    X, y, X_test, test_ids = load_data()

    print("\nTraining Ensemble Model...")
    predictions, scores = train_ensemble(X, y, X_test, test_ids)

    print("\nCreating submission...")
    create_submission(test_ids, predictions)

    print("\nDone!")

if __name__ == '__main__':
    main()