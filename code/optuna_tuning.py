# optuna_tuning.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import optuna

PROCESSED_DIR = '../processed data/'

def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        'verbose': -1,
        'class_weight': 'balanced'
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params, train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val)
        score = log_loss(y_val, val_pred)
        scores.append(score)
    
    return np.mean(scores)

def main():
    global X, y
    
    print("Loading data...")
    train = pd.read_csv(PROCESSED_DIR + 'train_advanced.csv')
    
    X = train.drop(['id', 'location', 'fault_severity'], axis=1)
    y = train['fault_severity']
    
    print(f"Training data shape: {X.shape}")
    
    # 运行Optuna优化
    print("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f"\nBest Log Loss: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    
    # 保存最佳参数
    best_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'verbose': -1,
        'class_weight': 'balanced',
        **study.best_params
    }
    
    import json
    with open(PROCESSED_DIR + 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print("\nBest parameters saved to best_params.json")

if __name__ == '__main__':
    main()