"""Offline synthetic monthly-panel demonstration; no credentials or downloads."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from reflexivity.sovereign import chronological_validation, paired_validation_metrics


def main():
    rng = np.random.default_rng(51)
    rows = []
    for month in pd.period_range('2006-01', '2016-12', freq='M'):
        date = pd.bdate_range(month.start_time, month.end_time)[-1]
        for cid in 'ABCDEFGH':
            row = {'real_date': date, 'month': month, 'cid': cid}
            for feature in ['underpriced', 'simple_underpriced', 'log_cds',
                            'cds_momentum', 'cds_vol', 'bond_carry',
                            'bond_momentum', 'bond_vol']:
                row[feature] = rng.normal()
            for horizon in [21, 63, 126]:
                row[f'label_end_{horizon}'] = date + pd.offsets.BDay(horizon + 1)
                row[f'cds_y_{horizon}'] = row['underpriced'] + rng.normal()
                row[f'bond_y_{horizon}'] = -row['underpriced'] + rng.normal()
            rows.append(row)
    panel = pd.DataFrame(rows)
    cfg = json.loads(Path('config_sovereign.json').read_text())
    cfg['bootstrap_reps'] = 50
    predictions, audit = chronological_validation(panel, cfg)
    metrics = paired_validation_metrics(predictions, cfg)
    assert not predictions.empty
    valid = audit[audit.status.eq('ok')]
    assert (pd.to_datetime(valid.max_training_label_end) <
            pd.to_datetime(valid.first_test_origin)).all()
    assert predictions.groupby(['outcome','signal','horizon','real_date']).size().ge(8).all()
    out = Path('results/demo')
    out.mkdir(parents=True, exist_ok=True)
    stamp = out/'manifest.json'
    if stamp.exists() and not json.loads(stamp.read_text()).get('synthetic'):
        raise ValueError('Output belongs to a non-synthetic run')
    predictions.to_csv(out/'synthetic_predictions.csv', index=False)
    metrics.to_csv(out/'synthetic_metrics.csv', index=False)
    audit.to_csv(out/'training_audit.csv', index=False)
    stamp.write_text(json.dumps({'synthetic': True, 'seed': 51,
        'scope': 'monthly feature-panel prediction demonstration',
        'daily_adapter_tested': False, 'bootstrap_reps': 50,
        'prediction_rows': len(predictions), 'training_label_boundary_passed': True}, indent=2))
    (out/'report.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Sovereign Risk Pricing Synthetic Demo</title>
<style>body{font:15px/1.7 system-ui;max-width:1000px;margin:36px auto;padding:20px}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}.scroll{overflow:auto}</style>
<h1>Sovereign Risk Pricing</h1><h2>Synthetic monthly-panel demonstration</h2>
<p>The feature panel and targets are artificial, with a deliberately planted relationship.
These outputs exercise chronological prediction and paired evaluation. They are not empirical
findings, provider score validation, or trading performance.</p>
<p>Training-label boundaries and a minimum of eight countries per evaluated month passed.</p>
<div class="scroll">''' + metrics.to_html(index=False,escape=True) + '</div></html>')
    print('Synthetic demo completed:', out/'report.html')


if __name__ == '__main__':
    main()
