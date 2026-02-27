# 数据模块示例（data_demo）

本目录用于演示 Qlib 数据相关模块的常见用法，重点包括：

- 数据缓存（cache）的使用方式与效果；
- 数据在内存中的复用（memory reuse），避免重复构建/加载带来的开销。

## 文件说明

- `data_cache_demo.py`：演示数据缓存相关用法。
- `data_mem_resuse_demo.py`：演示数据内存复用相关用法。

## 运行方式

在仓库根目录执行：

```bash
python backend/qlib/examples/data_demo/data_cache_demo.py
python backend/qlib/examples/data_demo/data_mem_resuse_demo.py
```

## 备注

- 首次运行若本地没有数据，脚本可能会触发数据下载或提示你先准备数据（取决于你环境里的 Qlib 配置）。
- 若你通过环境变量指定数据目录，请确认 `PROVIDER_URI` 指向正确的 Qlib 数据路径。
