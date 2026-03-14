# train_eff 依赖安装指南

## 文件说明

- **requirements.txt** - 项目所有依赖项列表
- **install_from_tsinghua.py** - Python 版安装脚本（跨平台）
- **install_from_tsinghua.ps1** - PowerShell 脚本（Windows 推荐）

## 快速安装

### 方法 1️⃣：使用 PowerShell 脚本（Windows 推荐）

```powershell
# 使用管理员权限运行 PowerShell，然后执行：
.\install_from_tsinghua.ps1
```

### 方法 2️⃣：使用 Python 脚本（跨平台）

```bash
python install_from_tsinghua.py
```

查看已安装的包版本：
```bash
python install_from_tsinghua.py --show
```

### 方法 3️⃣：直接使用 pip 命令

```bash
# 在项目根目录运行
pip install -i https://pypi.tsinghua.edu.cn/simple -r requirements.txt --upgrade
```

## 依赖项说明

| 包名 | 版本 | 说明 |
|------|------|------|
| torch | >=2.0.0 | PyTorch 深度学习框架 |
| torchvision | >=0.15.0 | PyTorch 计算机视觉库 |
| numpy | >=1.21.0 | 数值计算库 |
| Pillow | >=9.0.0 | 图像处理库 |
| tqdm | >=4.62.0 | 进度条库 |
| scikit-learn | >=1.0.0 | 机器学习库（用于评估指标） |
| timm | >=0.6.0 | PyTorch Image Models（用于 ConvNeXtV2） |
| matplotlib | >=3.5.0 | 可视化库 |
| pandas | >=1.3.0 | 数据处理库 |

## 镜像源信息

- **清华大学开源软件镜像站首页**: https://mirrors.tsinghua.edu.cn/
- **PyPI 镜像**: https://pypi.tsinghua.edu.cn/simple

## 常见问题

### Q: 如何更换回官方 PyPI 源？
```bash
pip install -r requirements.txt --upgrade
# 或配置 ~/.pip/pip.conf (Linux/Mac) 或 %APPDATA%\pip\pip.ini (Windows)
```

### Q: 安装 torch 太慢怎么办？
If you need to install specific CUDA versions:
```bash
# 示例：安装 CUDA 11.8 版本
pip install -i https://pypi.tsinghua.edu.cn/simple torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

### Q: 如何检查安装是否成功？
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import torchvision; print(f'TorchVision: {torchvision.__version__}')"
```

## 配置永久镜像源

如果不想每次都指定镜像源，可以创建配置文件：

### Windows
创建或编辑 `%APPDATA%\pip\pip.ini`：
```ini
[global]
index-url = https://pypi.tsinghua.edu.cn/simple
```

### Linux/Mac
创建或编辑 `~/.pip/pip.conf`：
```ini
[global]
index-url = https://pypi.tsinghua.edu.cn/simple
```

之后直接运行：
```bash
pip install -r requirements.txt --upgrade
```
