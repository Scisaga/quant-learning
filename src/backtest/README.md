## Backtest report

Generate an HTML report from a trained Qlib recorder (prediction + backtest + analysis):

`python src/backtest/generate_html_report.py --exp-name tutorial_exp --recorder-id <RECORDER_ID>`

If `--recorder-id` is omitted, it selects the latest recorder under `--exp-name`.

Output: `reports/qlib_report_<exp>_<rid>_<timestamp>.html`

