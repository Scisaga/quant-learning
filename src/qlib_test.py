'''解决 gym 废弃提示 与 gymnasium 不兼容问题'''
import sys
import gymnasium as gym
sys.modules['gym'] = gym

import qlib
from qlib.data import D
from qlib.config import C
from qlib.constant import REG_CN
from typing import Any, Dict

C.joblib_backend = "threading"  # Windows/py3.12: 避免创建子进程

# ========== 按文档的正确导入 ==========
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel            # ← 修正导入路径
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis, backtest_daily
from qlib.contrib.strategy import TopkDropoutStrategy   # ← 官方示例路径

def main():
    # 1) 初始化（建议显式设置 region）
    qlib.init(provider_uri="data/qlib_data/cn_data", region=REG_CN)

    # 2) 小测试：直接拉几列特征，验证数据 OK
    feats = [
        "$close",
        "Mean($close, 5)",
        "Std($return, 20)",
        "Ref($open, 1)",
    ]
    inst = D.instruments("csi300")
    df = D.features(inst, feats, "2018-01-01", "2020-12-31", freq="day")
    print(df.head())
    print("rows:", len(df))
    
    print(
        D.features(["SH000300"], ["$close"], "2005-01-01", "2030-01-01", freq="day")
        .index.get_level_values("datetime")[[0, -1]]
    )

    # 3) 构造 Alpha158 数据集
    '''!!! 默认方式下载的数据集只到2020-09-24，后续数据需要自行补充'''
    handler = Alpha158(instruments="csi300", start_time="2015-01-01", end_time="2022-12-31")
    dataset = DatasetH(handler=handler, segments={
        "train": ("2015-01-01", "2017-12-31"),
        "valid": ("2018-01-01", "2019-12-31"),
        "test":  ("2019-01-01", "2020-09-24"),
    })

    # 4) 训练一个基线 LGBM（官方工作流示例也使用该类）
    model = LGBModel(
        loss="mse",
        num_leaves=64,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    # 打开一个实验记录上下文
    with R.start(experiment_name="exp_lgb_alpha158"):

        # 记录参数
        R.log_params(model=model)

        # 用数据集训练模型
        model.fit(dataset)

        # 5) 预测 -> 按策略要求转为带 'score' 列的 DataFrame
        # 让模型在 test 切片 上做预测，返回预测结果。
        pred = model.predict(dataset, segment="test")   # 通常返回 Series(index: (datetime, instrument))
        
        # 把上一步得到的 Series 改名为 "score" 并转成 DataFrame。
        pred_score = pred.rename("score").to_frame()    # ← 文档要求 signal 形如 DataFrame[['score']]
        
        # 把预测结果持久化到本地文件，便于之后复用或载入
        payload: Dict[str, Any] = {"pred.pkl": pred_score}
        R.save_objects(**payload, artifact_path="prediction")  # type: ignore[arg-type]
        

        # 6) 组合策略：把预测作为 signal 交给 TopkDropoutStrategy
        strategy = TopkDropoutStrategy(topk=50, n_drop=10, signal=pred_score)

        # 7) 回测：费用与成交价放进 exchange_kwargs（文档示例）
        report, positions = backtest_daily(
            start_time="2020-01-01",
            end_time="2020-08-31",
            strategy=strategy,
            exchange_kwargs={
                "freq": "day",
                "deal_price": "close",
                "open_cost": 0.001,
                "close_cost": 0.001,
                "min_cost": 5,
                # "limit_threshold": 0.095,  # 需要时可开启
            },
        )

        # 8) 保存与风险指标（文档示例：计算超额收益/含成本与不含成本）
        artifacts: Dict[str, Any] = {
            "report.pkl": report,          # DataFrame
            "positions.pkl": positions,    # dict/DataFrame 都可
        }
        R.save_objects(**artifacts, artifact_path="backtest")
        print("Excess w/o cost:")
        print(risk_analysis(report["return"] - report["bench"]))
        print("Excess with cost:")
        print(risk_analysis(report["return"] - report["bench"] - report["cost"]))

if __name__ == "__main__":
    main()
