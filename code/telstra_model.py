import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 数据文件路径
DATA_DIR = '..\\processed data\\'
MODEL_DIR = '..\\model\\'
VIS_DIR = '..\\visualization\\'

# 确保目录存在
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

# 加载预处理后的数据
def load_preprocessed_data():
    print("Loading preprocessed data...")
    train = pd.read_csv(DATA_DIR + 'train_preprocessed.csv')
    val = pd.read_csv(DATA_DIR + 'val_preprocessed.csv')
    test = pd.read_csv(DATA_DIR + 'test_preprocessed.csv')
    
    # 分离特征和目标变量
    X_train = train.drop('fault_severity', axis=1).values
    y_train = train['fault_severity'].values
    X_val = val.drop('fault_severity', axis=1).values
    y_val = val['fault_severity'].values
    X_test = test.drop('id', axis=1).values
    test_ids = test['id'].values
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, test_ids

# 转换为PyTorch张量
def to_tensor(X, y=None):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor
    return X_tensor

# 定义模型架构
class TelstraModel(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super(TelstraModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
    
    # 保存最佳模型
    torch.save(best_model, MODEL_DIR + 'best_model.pth')
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Model saved to", MODEL_DIR + 'best_model.pth')
    
    return train_losses, val_losses, val_accuracies

# 评估模型
def evaluate_model(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return y_true, y_pred

# 生成预测
def generate_predictions(model, test_loader, test_ids):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 处理批次数据
            if isinstance(batch, list):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 获取概率
            probs = torch.softmax(outputs, dim=1)
            probabilities.extend(probs.cpu().numpy())
            
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    # 生成提交文件（Kaggle要求的格式）
    submission = pd.DataFrame({
        'id': test_ids,
        'predict_0': np.array(probabilities)[:, 0],
        'predict_1': np.array(probabilities)[:, 1],
        'predict_2': np.array(probabilities)[:, 2]
    })
    submission.to_csv(DATA_DIR + 'submission.csv', index=False)
    print("Predictions saved to", DATA_DIR + 'submission.csv')
    print("Submission file format: id, predict_0, predict_1, predict_2")
    
    return submission

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'training_curves.png')
    print("Training curves saved to", VIS_DIR + 'training_curves.png')

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['0', '1', '2'], 
                yticklabels=['0', '1', '2'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'confusion_matrix.png')
    print("Confusion matrix saved to", VIS_DIR + 'confusion_matrix.png')

# 主函数
def main():
    # 加载数据
    X_train, y_train, X_val, y_val, X_test, test_ids = load_preprocessed_data()
    
    # 创建数据加载器
    batch_size = 64
    
    train_dataset = torch.utils.data.TensorDataset(*to_tensor(X_train, y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(*to_tensor(X_val, y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_dataset = torch.utils.data.TensorDataset(to_tensor(X_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = TelstraModel(input_dim)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("Starting model training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=50
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load(MODEL_DIR + 'best_model.pth'))
    
    # 评估模型
    print("Evaluating model...")
    y_true, y_pred = evaluate_model(model, val_loader)
    
    # 打印分类报告
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    # 生成预测
    print("Generating test set predictions...")
    submission = generate_predictions(model, test_loader, test_ids)
    
    # 生成可视化
    print("Generating visualizations...")
    plot_training_curves(train_losses, val_losses, val_accuracies)
    plot_confusion_matrix(y_true, y_pred)
    
    print("Model training and evaluation completed!")

if __name__ == '__main__':
    main()