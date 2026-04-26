import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties

# 数据文件路径
DATA_DIR = '..\\data set\\'
VIS_DIR = '..\\visualization\\'

# 确保目录存在
os.makedirs(VIS_DIR, exist_ok=True)

# 设置中文字体
def set_chinese_font():
    # 尝试使用系统中的中文字体
    font_list = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'PingFang SC']
    for font_name in font_list:
        try:
            font = FontProperties(family=font_name, size=12)
            return font
        except:
            continue
    return None

font = set_chinese_font()

# 加载原始训练数据
def load_raw_data():
    print("Loading raw data...")
    train = pd.read_csv(DATA_DIR + 'train.csv')
    event_type = pd.read_csv(DATA_DIR + 'event_type.csv')
    log_feature = pd.read_csv(DATA_DIR + 'log_feature.csv')
    resource_type = pd.read_csv(DATA_DIR + 'resource_type.csv')
    severity_type = pd.read_csv(DATA_DIR + 'severity_type.csv')
    
    return train, event_type, log_feature, resource_type, severity_type

# 绘制故障严重程度分布
def plot_fault_severity_distribution(train):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='fault_severity', data=train, palette='viridis')
    if font:
        plt.title('Fault Severity Distribution', fontproperties=font)
        plt.xlabel('Fault Severity', fontproperties=font)
        plt.ylabel('Count', fontproperties=font)
        plt.xticks([0, 1, 2], ['0 (Low)', '1 (Medium)', '2 (High)'])
    else:
        plt.title('Fault Severity Distribution')
        plt.xlabel('Fault Severity')
        plt.ylabel('Count')
        plt.xticks([0, 1, 2], ['0 (Low)', '1 (Medium)', '2 (High)'])
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'fault_severity_distribution.png')
    print("Fault severity distribution saved to", VIS_DIR + 'fault_severity_distribution.png')

# 绘制事件类型分布
def plot_event_type_distribution(event_type):
    top_events = event_type['event_type'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_events.index, y=top_events.values, palette='viridis', hue=top_events.index, legend=False)
    if font:
        plt.title('Top 10 Event Types Distribution', fontproperties=font)
        plt.xlabel('Event Type', fontproperties=font)
        plt.ylabel('Frequency', fontproperties=font)
    else:
        plt.title('Top 10 Event Types Distribution')
        plt.xlabel('Event Type')
        plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'event_type_distribution.png')
    print("Event type distribution saved to", VIS_DIR + 'event_type_distribution.png')

# 绘制日志特征分布
def plot_log_feature_distribution(log_feature):
    # 前10个最常见的日志特征
    top_logs = log_feature['log_feature'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_logs.index, y=top_logs.values, palette='viridis', hue=top_logs.index, legend=False)
    if font:
        plt.title('Top 10 Log Features Distribution', fontproperties=font)
        plt.xlabel('Log Feature', fontproperties=font)
        plt.ylabel('Frequency', fontproperties=font)
    else:
        plt.title('Top 10 Log Features Distribution')
        plt.xlabel('Log Feature')
        plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'log_feature_distribution.png')
    print("Log feature distribution saved to", VIS_DIR + 'log_feature_distribution.png')

# 绘制资源类型分布
def plot_resource_type_distribution(resource_type):
    resource_counts = resource_type['resource_type'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=resource_counts.index, y=resource_counts.values, palette='viridis', hue=resource_counts.index, legend=False)
    if font:
        plt.title('Resource Type Distribution', fontproperties=font)
        plt.xlabel('Resource Type', fontproperties=font)
        plt.ylabel('Frequency', fontproperties=font)
    else:
        plt.title('Resource Type Distribution')
        plt.xlabel('Resource Type')
        plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'resource_type_distribution.png')
    print("Resource type distribution saved to", VIS_DIR + 'resource_type_distribution.png')

# 绘制严重程度类型分布
def plot_severity_type_distribution(severity_type):
    severity_counts = severity_type['severity_type'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=severity_counts.index, y=severity_counts.values, palette='viridis', hue=severity_counts.index, legend=False)
    if font:
        plt.title('Severity Type Distribution', fontproperties=font)
        plt.xlabel('Severity Type', fontproperties=font)
        plt.ylabel('Frequency', fontproperties=font)
    else:
        plt.title('Severity Type Distribution')
        plt.xlabel('Severity Type')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'severity_type_distribution.png')
    print("Severity type distribution saved to", VIS_DIR + 'severity_type_distribution.png')

# 绘制日志特征音量分布
def plot_volume_distribution(log_feature):
    plt.figure(figsize=(10, 6))
    sns.histplot(log_feature['volume'], bins=50, kde=True, color='green')
    if font:
        plt.title('Log Feature Volume Distribution', fontproperties=font)
        plt.xlabel('Volume', fontproperties=font)
        plt.ylabel('Frequency', fontproperties=font)
    else:
        plt.title('Log Feature Volume Distribution')
        plt.xlabel('Volume')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(VIS_DIR + 'volume_distribution.png')
    print("Log feature volume distribution saved to", VIS_DIR + 'volume_distribution.png')

# 主函数
def main():
    # 加载数据
    train, event_type, log_feature, resource_type, severity_type = load_raw_data()
    
    # 绘制各种分布
    plot_fault_severity_distribution(train)
    plot_event_type_distribution(event_type)
    plot_log_feature_distribution(log_feature)
    plot_resource_type_distribution(resource_type)
    plot_severity_type_distribution(severity_type)
    plot_volume_distribution(log_feature)
    
    print("Data exploration visualization completed!")

if __name__ == '__main__':
    main()