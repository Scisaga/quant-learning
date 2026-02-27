# LightGBM 超参搜索（Optuna）

本示例演示如何使用 Optuna 对 LightGBM 在 `Alpha158/Alpha360` 数据集上的配置进行超参数搜索，并用 `optuna-dashboard` 进行可视化。

## 运行前准备

在仓库根目录安装依赖：

```bash
cd backend/qlib/examples/hyperparameter/LightGBM
pip install -r requirements.txt
```

## Alpha158

终端 1（创建 Study + 启动 Dashboard）：

```bash
optuna create-study --study-name LGBM_158 --storage sqlite:///db.sqlite3
optuna-dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```

终端 2（开始搜索）：

```bash
cd backend/qlib/examples/hyperparameter/LightGBM
python hyperparameter_158.py
```

## Alpha360

终端 1（创建 Study + 启动 Dashboard）：

```bash
optuna create-study --study-name LGBM_360 --storage sqlite:///db.sqlite3
optuna-dashboard --port 5000 --host 0.0.0.0 sqlite:///db.sqlite3
```

终端 2（开始搜索）：

```bash
cd backend/qlib/examples/hyperparameter/LightGBM
python hyperparameter_360.py
```

## 备注

- Dashboard 默认访问：`http://localhost:5000`（或你的机器 IP + 端口）。
- 你可以把 `sqlite:///db.sqlite3` 替换为 MySQL/PostgreSQL 等远程存储，以便多机协同与持久化。
