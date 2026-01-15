from __future__ import annotations

import argparse
import sys
from typing import Optional

from pathlib import Path

# Suppress gym deprecation warning by aliasing gymnasium to gym (Qlib RL deps sometimes import gym).
try:
    import gymnasium as gym  # type: ignore

    sys.modules["gym"] = gym
except Exception:
    pass

import qlib
from qlib.config import C
from qlib.constant import REG_CN
from qlib.contrib.data.handler import _DEFAULT_INFER_PROCESSORS, _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.contrib.data.loader import Alpha158DL
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord


def _comma_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _filter_instruments_by_feature_bins(
    *,
    provider_uri: str,
    instruments: list[str],
    daily_feature_fields: list[str],
    freq: str = "day",
) -> list[str]:
    if not daily_feature_fields:
        return instruments

    base = Path(provider_uri).resolve()
    features_root = base / "features"
    if not features_root.exists():
        return []

    required = [f"{f.lower()}.{freq.lower()}.bin" for f in daily_feature_fields]

    kept: list[str] = []
    for inst in instruments:
        inst_dir = features_root / inst.lower()
        if not inst_dir.exists():
            continue
        ok = True
        for fn in required:
            if not (inst_dir / fn).exists():
                ok = False
                break
        if ok:
            kept.append(inst)
    return kept


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train LGBModel on Alpha158 (+ optional PIT financial features).")
    parser.add_argument("--provider-uri", default="data/qlib_data/cn_data")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--benchmark", default="SH000300")
    parser.add_argument("--exp-name", default="tutorial_exp")

    parser.add_argument("--start-time", default="2015-01-01")
    parser.add_argument("--end-time", default="2020-12-31")
    parser.add_argument("--fit-start-time", default=None)
    parser.add_argument("--fit-end-time", default=None)
    parser.add_argument("--train", default="2015-01-01,2016-12-31", help="train start,end")
    parser.add_argument("--valid", default="2017-01-01,2018-12-31", help="valid start,end")
    parser.add_argument("--test", default="2019-01-01,2020-12-31", help="test start,end")

    parser.add_argument("--label-expr", default="Ref($close, -1) / $close - 1")
    parser.add_argument(
        "--pit-fields",
        nargs="?",
        const="",
        default="assettequity_q,netprofit_q,roeavg_q,yoyni_q",
        help="comma-separated (pass empty to disable)",
    )
    parser.add_argument(
        "--pit-feature-prefix",
        default="pit_",
        help="offline PIT daily feature prefix: $<prefix><field> (default: $pit_<field>)",
    )

    args = parser.parse_args(argv)

    train_start, train_end = _comma_list(args.train)
    valid_start, valid_end = _comma_list(args.valid)
    test_start, test_end = _comma_list(args.test)
    fit_start = args.fit_start_time or train_start
    fit_end = args.fit_end_time or train_end

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)

    pit_fields = _comma_list(args.pit_fields)
    pit_daily_fields = [f"{args.pit_feature_prefix}{f}" for f in pit_fields] if pit_fields else []
    if not pit_fields:
        raise ValueError("pit-fields is empty; this script requires precomputed PIT daily features.")

    pit_feature_fields = [f"${f}" for f in pit_daily_fields]
    pit_feature_names = [f.upper() for f in pit_daily_fields]
    cleanup = None

    try:
        print(f"[config] pit_fields={pit_fields}", flush=True)
        print(f"[config] pit_feature_prefix={args.pit_feature_prefix}", flush=True)
        print(f"[config] handler_range={args.start_time}..{args.end_time}", flush=True)

        from qlib.data import D

        inst_cfg = D.instruments(args.market)
        inst_list = D.list_instruments(inst_cfg, start_time=args.start_time, end_time=args.end_time, freq="day", as_list=True)
        print(
            f"[info] market={args.market} instruments={len(inst_list)} (union over {args.start_time}..{args.end_time})",
            flush=True,
        )

        kept = _filter_instruments_by_feature_bins(
            provider_uri=args.provider_uri, instruments=inst_list, daily_feature_fields=pit_daily_fields
        )
        if len(kept) != len(inst_list):
            missing = [x for x in inst_list if x not in set(kept)]
            example = ", ".join(missing[:10])
            raise RuntimeError(
                "PIT daily features are missing for some instruments. "
                f"expected prefix={args.pit_feature_prefix!r} fields={pit_fields!r}; "
                f"missing_count={len(missing)}/{len(inst_list)} example_missing=[{example}]. "
                "Run src/scripts/dump_pit_daily_features.py first."
            )

        instruments_for_handler = inst_list

        base_fields, base_names = Alpha158DL.get_feature_config()
        fields = base_fields + pit_feature_fields
        names = base_names + pit_feature_names

        label = ([args.label_expr], ["LABEL0"])
        infer_processors = check_transform_proc(_DEFAULT_INFER_PROCESSORS, fit_start, fit_end)
        learn_processors = check_transform_proc(_DEFAULT_LEARN_PROCESSORS, fit_start, fit_end)

        print("[stage] build handler/dataset", flush=True)
        handler = init_instance_by_config(
            {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "instruments": instruments_for_handler if instruments_for_handler else args.market,
                    "start_time": args.start_time,
                    "end_time": args.end_time,
                    "process_type": "append",
                    "data_loader": {
                        "class": "QlibDataLoader",
                        "module_path": "qlib.data.dataset.loader",
                        "kwargs": {"config": {"feature": (fields, names), "label": label}, "freq": "day"},
                    },
                    "infer_processors": infer_processors,
                    "learn_processors": learn_processors,
                },
            }
        )

        dataset = init_instance_by_config(
            {
                "class": "DatasetH",
                "module_path": "qlib.data.dataset",
                "kwargs": {
                    "handler": handler,
                    "segments": {
                        "train": (train_start, train_end),
                        "valid": (valid_start, valid_end),
                        "test": (test_start, test_end),
                    },
                },
            }
        )

        model_config = {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 400,
                "num_threads": 20,
            },
        }
        model = init_instance_by_config(model_config)

        with R.start(experiment_name=args.exp_name):
            # Save key experiment parameters for downstream report generation.
            R.log_params(
                provider_uri=args.provider_uri,
                market=args.market,
                benchmark=args.benchmark,
                start_time=args.start_time,
                end_time=args.end_time,
                train=args.train,
                valid=args.valid,
                test=args.test,
                label_expr=args.label_expr,
                pit_fields=args.pit_fields,
                pit_feature_prefix=args.pit_feature_prefix,
            )
            run_config = {
                "provider_uri": args.provider_uri,
                "market": args.market,
                "benchmark": args.benchmark,
                "start_time": args.start_time,
                "end_time": args.end_time,
                "segments": {
                    "train": (train_start, train_end),
                    "valid": (valid_start, valid_end),
                    "test": (test_start, test_end),
                },
                "label_expr": args.label_expr,
                "pit_fields": pit_fields,
                "pit_fields_raw": args.pit_fields,
                "pit_feature_prefix": args.pit_feature_prefix,
                "model": model_config,
            }

            R.save_objects(
                **{
                    "run_config.pkl": run_config,
                    "model_config.pkl": model_config,
                    "lgb_params.pkl": dict(model_config.get("kwargs", {})),
                }
            )

            print("[stage] fit model", flush=True)
            model.fit(dataset)
            R.save_objects(trained_model=model)

            rec = R.get_recorder()
            print("[stage] generate signal", flush=True)
            sr = SignalRecord(model, dataset, rec)
            sr.generate()

            print(f"recorder_id={rec.id} experiment_id={rec.experiment_id}")
        return 0
    finally:
        if cleanup is not None:
            cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
