-- Migration: 0001_init_schema
-- Description: train_runs, backtest_runs, grid_runs, grid_jobs 表及索引
-- Usage: psql $DATABASE_URL -f backend/migrations/0001_init_schema.sql
-- Or: python -m backend.scripts.init_db

-- train_runs：训练元数据
CREATE TABLE IF NOT EXISTS train_runs (
    id BIGSERIAL PRIMARY KEY,
    recorder_id VARCHAR(64) NOT NULL UNIQUE,
    experiment_name VARCHAR(255) NOT NULL,
    market VARCHAR(64) NOT NULL,
    benchmark VARCHAR(32) NOT NULL,
    label_expr TEXT,
    pit_fields TEXT,
    pit_feature_prefix VARCHAR(32),
    train_start DATE,
    train_end DATE,
    valid_start DATE,
    valid_end DATE,
    test_start DATE,
    test_end DATE,
    handler_start DATE,
    handler_end DATE,
    model_config JSONB,
    minio_model_path TEXT,
    status VARCHAR(32) DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_train_runs_recorder_id ON train_runs(recorder_id);
CREATE INDEX IF NOT EXISTS idx_train_runs_market ON train_runs(market);
CREATE INDEX IF NOT EXISTS idx_train_runs_created_at ON train_runs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_train_runs_segments ON train_runs(train_start, train_end);

-- backtest_runs：回测元数据
CREATE TABLE IF NOT EXISTS backtest_runs (
    id BIGSERIAL PRIMARY KEY,
    train_run_id BIGINT,
    recorder_id VARCHAR(64) NOT NULL,
    backtest_start DATE,
    backtest_end DATE,
    strategy_config JSONB,
    annualized_return DOUBLE PRECISION,
    information_ratio DOUBLE PRECISION,
    max_drawdown DOUBLE PRECISION,
    ic DOUBLE PRECISION,
    icir DOUBLE PRECISION,
    rank_ic DOUBLE PRECISION,
    rank_icir DOUBLE PRECISION,
    excess_return_without_cost JSONB,
    excess_return_with_cost JSONB,
    minio_report_html TEXT,
    minio_train_log TEXT,
    minio_report_log TEXT,
    status VARCHAR(32) DEFAULT 'completed',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_train_run_id ON backtest_runs(train_run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_recorder_id ON backtest_runs(recorder_id);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_ic ON backtest_runs(ic);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_ir ON backtest_runs(information_ratio);

-- grid_runs：网格批次
CREATE TABLE IF NOT EXISTS grid_runs (
    id BIGSERIAL PRIMARY KEY,
    markets TEXT[],
    label_horizons INT[],
    pit_grid VARCHAR(32),
    start_date DATE,
    end_date DATE,
    train_years INT,
    valid_years INT,
    test_years INT,
    step_years INT,
    minio_summary_path TEXT,
    minio_results_path TEXT,
    total_jobs INT,
    ok_jobs INT,
    failed_jobs INT,
    status VARCHAR(32) DEFAULT 'running',
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_grid_runs_started_at ON grid_runs(started_at DESC);

-- grid_jobs：网格内单个 job
CREATE TABLE IF NOT EXISTS grid_jobs (
    id BIGSERIAL PRIMARY KEY,
    grid_run_id BIGINT NOT NULL,
    job_key VARCHAR(256) NOT NULL,
    market VARCHAR(64),
    benchmark VARCHAR(32),
    label_horizon INT,
    label_expr TEXT,
    pit VARCHAR(64),
    "window" JSONB,
    recorder_id VARCHAR(64),
    train_run_id BIGINT,
    backtest_run_id BIGINT,
    status VARCHAR(32),
    minio_report_html TEXT,
    minio_train_log TEXT,
    minio_report_log TEXT,
    error TEXT,
    metrics JSONB,
    params JSONB,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_grid_jobs_grid_run_id ON grid_jobs(grid_run_id);
CREATE INDEX IF NOT EXISTS idx_grid_jobs_status ON grid_jobs(status);
CREATE INDEX IF NOT EXISTS idx_grid_jobs_job_key ON grid_jobs(job_key);
CREATE INDEX IF NOT EXISTS idx_grid_jobs_market ON grid_jobs(market);
