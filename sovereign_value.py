#!/usr/bin/env python3
"""Runnable sovereign valuation-divergence study. Uses cached live data by default.
No simulated fallback, trading orders, or disclosure of OAuth values.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'work/cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reflexivity.data import download_live, read_env, read_qdf
from reflexivity.sovereign import build_panel, associations, chronological_validation, paired_validation_metrics

SOURCES={
    'macro_scores':'https://macrosynergy.com/academy/notebooks/emerging-market-sovereign-credit-risk-scores/',
    'premium_scores':'https://macrosynergy.com/academy/notebooks/emerging-market-sovereign-credit-risk-premium-scores/',
    'bond_returns':'https://macrosynergy.com/academy/notebooks/emerging-market-bond-index-returns/',
    'CDS_units':'https://macrosynergy.com/academy/notebooks/sovereign-cds-spreads-and-carry/'}
CARRY_FALLBACK=['AED','DOP','EGP','INR','NGN','OMR','QAR','RSD','SAR','UYU']


def cache_path(cfg):
    identity={k:cfg[k] for k in ('cids','categories','start_date','end_date')}
    identity['client_version']='1.8.0'
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    return ROOT/'work/downloads'/key/'raw.parquet'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def table(frame,columns=None):
    if frame.empty:return '<p>没有足够样本。</p>'
    return frame[columns].to_html(index=False,float_format=lambda x:f'{x:.3f}',escape=True) if columns else frame.to_html(index=False,float_format=lambda x:f'{x:.3f}',escape=True)


def robust_winners(metrics):
    if metrics.empty:return []
    good=metrics[metrics.horizon.eq(63)&metrics.status.eq('ok')]
    winners=[]
    for (outcome,signal),part in good.groupby(['outcome','signal']):
        if set(part.block_months)=={6,12} and part.joint_low.gt(0).all():
            winners.append({'outcome':outcome,'signal':signal})
    return winners


def figures(panel,metrics,watch,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,7))
    view=watch.dropna(subset=['underpriced']).sort_values('underpriced')
    ax.barh(view.cid,view.underpriced,color=np.where(view.underpriced>0,'#b45309','#0369a1'))
    ax.axvline(0,color='#475569',lw=.8)
    ax.set_xlabel('Negative provider risk-premium score: positive = undercompensated risk')
    ax.set_title(f'Historical sovereign valuation gap | {watch.real_date.max():%Y-%m-%d}')
    fig.tight_layout();fig.savefig(out/'historical_gap.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    data=metrics[metrics.horizon.eq(63)&metrics.block_months.eq(12)].copy() if not metrics.empty else metrics
    if not data.empty:
        labels=data.outcome+' / '+data.signal
        x=data.relative_mse_improvement.to_numpy()*100
        low=data.joint_low.to_numpy()*100;high=data.joint_high.to_numpy()*100
        for i in range(len(data)):
            ax.plot([low[i],high[i]],[i,i],color='#64748b',lw=2)
            ax.scatter([x[i]],[i],color='#0369a1',zorder=3)
        ax.set_yticks(range(len(data)),labels)
    ax.axvline(0,color='#b45309',lw=1)
    ax.set_xlabel('Forecast MSE reduction vs price-only baseline (%)')
    ax.set_title('63-business-day validation | simultaneous 95% bands, 12-month blocks')
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,cid in zip(axes,['BRL','TRY','ZAR']):
        p=panel[panel.cid.eq(cid)]
        ax.plot(p.real_date,p.underpriced,label='Provider underpriced-risk score',color='#b45309')
        ax.plot(p.real_date,p.simple_underpriced,label='Fixed three-factor vs CDS score',color='#0369a1',alpha=.8)
        ax.axhline(0,color='#64748b',lw=.6);ax.set_ylabel(cid)
    axes[0].legend(loc='upper left',ncol=2,fontsize=8)
    axes[0].set_title('Predeclared country examples: BRL / TRY / ZAR (different score units)')
    fig.tight_layout();fig.savefig(out/'country_history.png',dpi=150);plt.close(fig)


def make_report(panel,coverage,association,metrics,sens_association,sens_metrics,watch,manifest,out):
    """English report for locally supplied/authorized data."""
    sections=[]
    for name,frame in [('Primary and sensitivity metrics',metrics),('Associations',association),
                       ('Sensitivity metrics',sens_metrics),('Coverage',coverage)]:
        sections.append('<h2>'+html.escape(name)+'</h2><div class="scroll">'+frame.to_html(index=False,escape=True)+'</div>')
    note=('A primary comparison meets the configured simultaneous-interval criterion.' if manifest['primary_robust_improvements'] else 'No primary comparison meets the configured criterion across both block lengths. The results do not establish stable incremental predictive value.')
    content='<!doctype html><html lang="en"><meta charset="utf-8"><title>Sovereign Risk Pricing</title><style>body{font:15px/1.7 system-ui;max-width:1100px;margin:32px auto;padding:20px}.scroll{overflow:auto}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}</style><h1>Sovereign Risk Pricing</h1><p>'+html.escape(note)+'</p><p>Chronological relative validation does not independently certify provider vintages or executable trading P&amp;L. CDS and dollar-bond outcomes are evaluated separately.</p><img src="validation.png" alt="Primary validation comparison">'+''.join(sections)+'</html>'
    (out/'report.html').write_text(content,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config_sovereign.json')
    parser.add_argument('--input',type=Path,help='Explicit QDF file; otherwise reuse matching live cache')
    parser.add_argument('--refresh',action='store_true',help='Re-download from DataQuery using local .env')
    parser.add_argument('--output',type=Path,default=ROOT/'results/sovereign_value')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    source=args.input.resolve() if args.input else cache_path(cfg)
    if args.input or (source.exists() and not args.refresh):
        raw=read_qdf(source)
        audit_path=source.with_name('download_audit.json')
        metadata=json.loads(audit_path.read_text()) if audit_path.exists() else {'mode':'file'}
    else:
        print('Downloading live data; OAuth values and SDK output are suppressed.',flush=True)
        # Never forward SDK HTTP bodies, exception messages, or token-bearing URLs.
        buffer=io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
                raw,metadata=download_live(cfg,ROOT/'work/downloads',refresh=args.refresh,credentials=read_env(ROOT/'.env'))
        except Exception as exc:
            print(f'DataQuery download failed ({type(exc).__name__}); no simulation fallback.',file=sys.stderr)
            return 1
        finally:buffer.close()
    print(f'Loaded {len(raw):,} live/file rows; latest {raw.real_date.max():%Y-%m-%d}.',flush=True)
    panel,coverage,_=build_panel(raw,cfg)
    panel.to_parquet(out/'monthly_panel.parquet',index=False)
    coverage.to_csv(out/'coverage.csv',index=False)
    print(f'Built {len(panel):,} monthly states; fitting frozen association tests.',flush=True)
    assoc=associations(panel,cfg);assoc.to_csv(out/'associations.csv',index=False)
    print('Running paired annual expanding-window validation.',flush=True)
    pred,audit=chronological_validation(panel,cfg)
    metrics=paired_validation_metrics(pred,cfg)
    pred.to_csv(out/'validation_predictions.csv',index=False);audit.to_csv(out/'training_audit.csv',index=False)
    metrics.to_csv(out/'validation_metrics.csv',index=False)
    sensitivity_a=[];sensitivity_m=[]
    subsets={'exclude_RUB_from_2022':panel[~(panel.cid.eq('RUB')&panel.real_date.ge(cfg['russia_sensitivity_cutoff']))],
             'all_seven_components':panel[panel.available_component_count.eq(7)]}
    for name,part in subsets.items():
        print(f'Running sensitivity: {name}, {len(part):,} monthly states.',flush=True)
        if part.empty:
            sensitivity_a.append(pd.DataFrame([{'sample':name,'status':'empty_sample'}]))
            sensitivity_m.append(pd.DataFrame([{'sample':name,'status':'empty_sample'}]));continue
        a=associations(part,cfg,horizons=(63,)).assign(sample=name);sensitivity_a.append(a)
        p,t=chronological_validation(part,cfg)
        m=paired_validation_metrics(p[p.horizon.eq(63)] if not p.empty else p,cfg)
        if m.empty:m=pd.DataFrame([{'status':'no_eligible_validation_rows'}])
        sensitivity_m.append(m.assign(sample=name))
        t.to_csv(out/f'training_audit_{name}.csv',index=False)
        p[p.horizon.eq(63)].to_csv(out/f'predictions_{name}.csv',index=False) if not p.empty else None
    sa=pd.concat(sensitivity_a,ignore_index=True);sm=pd.concat(sensitivity_m,ignore_index=True)
    sa.to_csv(out/'sensitivity_associations.csv',index=False);sm.to_csv(out/'sensitivity_metrics.csv',index=False)
    watch=panel[panel.real_date.eq(panel.real_date.max())].copy().sort_values('underpriced',ascending=False)
    watch['cds_available']=watch.cds.notna()
    watch['provider_market_input']=np.where(watch.cid.isin(CARRY_FALLBACK),'bond_carry','CDS')
    watch.to_csv(out/'historical_watchlist.csv',index=False)
    valid=audit[audit.status.eq('ok')]
    source_checks={role:bool((panel.loc[panel[role].notna(),f'{role}_source_date']<=panel.loc[panel[role].notna(),'real_date']).all()) for role in cfg['categories']}
    verification={'sources_not_after_origins':source_checks,
        'labels_strictly_before_test_year':bool((pd.to_datetime(valid.max_training_label_end)<pd.to_datetime(valid.year.astype(str)+'-01-01')).all()),
        'paired_predictions_finite':bool(not pred.empty and np.isfinite(pred[['baseline_prediction','enhanced_prediction','target_relative']]).all().all()),
        'at_least_eight_countries_each_evaluation_month':bool(not pred.empty and pred.groupby(['outcome','signal','horizon','real_date']).size().ge(cfg['minimum_cross_section']).all()),
        'no_synthetic_CDS_for_carry_fallback':bool(panel.loc[panel.cid.isin(CARRY_FALLBACK),'cds'].isna().all()),
        'used_live_or_explicit_file_not_demo':True,'independent_vintage_verification':False}
    critical=[*source_checks.values(),verification['labels_strictly_before_test_year'],verification['paired_predictions_finite'],
        verification['at_least_eight_countries_each_evaluation_month'],verification['no_synthetic_CDS_for_carry_fallback']]
    verification['integrity_checks_passed']=all(critical)
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    manifest={'status':'completed' if all(critical) else 'integrity_check_failed','generated_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),
        'config':cfg,'raw_path':str(source),'raw_sha256':sha(source),'config_sha256':sha(args.config),
        'code_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'reflexivity/sovereign.py']},
        'raw_rows':len(raw),'raw_start':str(raw.real_date.min().date()),'raw_end':str(raw.real_date.max().date()),
        'data_retrieved_at_utc':metadata.get('retrieved_at_utc'),
        'calendar':'Monday-Friday; no local holiday calendar; missing returns not zero-filled',
        'raw_CDS_units':'percent; one unit equals 100 basis points',
        'monthly_states':len(panel),'monthly_start':str(panel.real_date.min().date()),'latest_screening_date':str(panel.real_date.max().date()),
        'raw_countries':int(raw.cid.nunique()),'CDS_countries':int(panel[panel.cds.notna()].cid.nunique()),
        'primary_robust_improvements':robust_winners(metrics),'sources':SOURCES,'carry_fallback_countries':CARRY_FALLBACK,
        'missing_catalogue_tickers':metadata.get('missing_catalogue_tickers',[]),'download_failures':metadata.get('download_failures',[]),
        'vintage_status':cfg['score_vintage_status'],'trading_costs':'not_modelled','causal_claim':False,
        'python_version':sys.version.split()[0],'pandas_version':pd.__version__,'numpy_version':np.__version__}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    figures(panel,metrics,watch,out)
    make_report(panel,coverage,assoc,metrics,sa,sm,watch,manifest,out)
    print(f"Completed: {out/'report.html'}; integrity={all(critical)}; robust primary improvements={len(manifest['primary_robust_improvements'])}",flush=True)
    return 0 if all(critical) else 2

if __name__=='__main__':raise SystemExit(main())
