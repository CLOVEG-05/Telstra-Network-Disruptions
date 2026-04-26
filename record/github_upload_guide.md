# GitHub 上传指南

## 1. 在 GitHub 上创建新仓库

1. 访问 https://github.com/new
2. 填写仓库名称: `telstra-recruiting-network`
3. 选择 Private (私有) 或 Public (公开)
4. 点击 "Create repository"

## 2. 本地初始化并上传

在项目目录下执行:

```bash
cd "d:\kaggle network\telstra-recruiting-network"
git init
git add .
git commit -m "Initial commit: Telstra Network Fault Prediction with 4-model ensemble"
```

## 3. 添加远程仓库

```bash
git remote add origin https://github.com/yourusername/telstra-recruiting-network.git
```

**注意**: 请将 `yourusername` 替换为您的 GitHub 用户名

## 4. 推送到 GitHub

```bash
git branch -M main
git push -u origin main
```

## 5. 验证上传

访问 `https://github.com/yourusername/telstra-recruiting-network` 查看项目

---

## Git 常用命令

```bash
# 查看状态
git status

# 查看远程仓库
git remote -v

# 查看提交历史
git log --oneline

# 创建新分支
git checkout -b feature/new-feature

# 合并分支
git checkout main
git merge feature/new-feature

# 拉取更新
git pull origin main
```

---

## 注意事项

1. **数据文件**: `.gitignore` 已配置排除所有CSV数据文件，仅保留 `sample_submission.csv` 作为示例
2. **模型文件**: 排除 `.pth`, `.pkl`, `.joblib` 文件
3. **敏感信息**: 确保没有上传任何API密钥或密码
4. **大文件**: 如果有文件超过50MB，Git LFS可能需要配置