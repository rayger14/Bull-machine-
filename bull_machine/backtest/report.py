
from .schemas import Summary, config_hash
import pandas as pd
from pathlib import Path
import json

def write_report(run_id: str, cfg: dict, metrics: dict, trades: pd.DataFrame, equity: pd.DataFrame, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    trades_path = out/f"{run_id}_trades.csv"
    equity_path = out/f"{run_id}_equity.csv"
    summary_path = out/f"{run_id}_summary.json"
    md_path = out/f"{run_id}_report.md"
    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path)
    summary = Summary(run_id=run_id, cfg_hash=config_hash(cfg), seed=cfg.get('engine',{}).get('seed',0),
                      commit=cfg.get('meta',{}).get('commit','unknown'),
                      metrics=metrics, notes=cfg.get('meta',{}).get('notes',[]))
    summary_path.write_text(summary.to_json(), encoding='utf-8')
    md = [f"# Run {run_id}", "", "## Metrics", json.dumps(metrics, indent=2), "",
          "## Artifacts", f"- trades: {trades_path}", f"- equity: {equity_path}", f"- summary: {summary_path}"]
    md_path.write_text("\n".join(md), encoding='utf-8')
    return {"trades":str(trades_path), "equity":str(equity_path), "summary":str(summary_path), "report":str(md_path)}
