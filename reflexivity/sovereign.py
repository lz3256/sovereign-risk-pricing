"""Sovereign price/fundamental divergence: historical exploratory validation.

DataQuery scores are information-state series; historical constituent/model
versions have not been independently verified. Chronological validation here
must not be presented as a fully certified point-in-time trading backtest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .mechanism import weighted_within, within_beta

MACRO_ROLES = ['fiscal','external','investment','fx_debt','governance','growth','inflation']
SIMPLE_ROLES = ['fiscal','external','growth']
MARKET_ROLES = {'cds','bond_cash','bond_excess','bond_carry','rating'}


def cross_z(s,min_count=8):
    valid=s.dropna()
    if len(valid)<min_count or valid.std(ddof=1)<1e-10:
        return pd.Series(np.nan,index=s.index)
    return (s-valid.mean())/valid.std(ddof=1)


def future_sum(series,horizon,delay=1):
    # Information at t -> execute at t+delay close -> earn subsequent returns.
    values=pd.concat([series.shift(-(delay+j)) for j in range(1,horizon+1)],axis=1)
    return values.sum(axis=1,min_count=horizon)


def monthly_state(part,calendar,month_dates,max_age,require_grade,max_grade):
    """Choose latest actual posted value then grade it, never resurrect an old
    approved score behind a newer explicit quality rejection.
    """
    part=part.set_index('real_date').reindex(calendar)
    actual=part.value.notna()
    positions=pd.Series(np.arange(len(calendar)),index=calendar)
    source=positions.where(actual).ffill()
    age=positions-source
    index=source.fillna(0).astype(int).to_numpy()
    values=part.value.to_numpy(dtype=float)[index]
    grades=part.grading.to_numpy(dtype=float)[index]
    dates=pd.Series(calendar,index=calendar).where(actual).ffill()
    okay=source.notna().to_numpy() & age.le(max_age).to_numpy()
    if require_grade:
        okay &= np.isfinite(grades) & (grades<=max_grade)
    state=pd.DataFrame({'value':np.where(okay,values,np.nan),'grade':grades,
        'source_date':dates,'source_age_business_days':age},index=calendar)
    return state.reindex(month_dates)


def build_panel(raw,cfg):
    end=min(pd.Timestamp(cfg['end_date']),raw.real_date.max())
    calendar=pd.bdate_range(max(pd.Timestamp(cfg['start_date']),raw.real_date.min()),end)
    month_dates=pd.Series(calendar,index=calendar).groupby(calendar.to_period('M')).max()
    # A partial final month cannot masquerade as a complete monthly origin.
    expected_close=month_dates.index.to_timestamp('M')
    expected_close=pd.DatetimeIndex([pd.bdate_range(d.replace(day=1),d)[-1] for d in expected_close])
    month_dates=month_dates[expected_close<=end]
    frames,coverage=[],[]
    grouped={(c,x):g for (c,x),g in raw.groupby(['cid','xcat'],observed=True)}
    horizons=[21,63,126]
    for cid in cfg['cids']:
        frame=pd.DataFrame(index=pd.DatetimeIndex(month_dates.values))
        daily={}
        for role,xcat in cfg['categories'].items():
            part=grouped.get((cid,xcat),pd.DataFrame(columns=raw.columns))
            state=monthly_state(part,calendar,frame.index,cfg['month_max_quote_age_business_days'],
                role not in MARKET_ROLES,cfg['max_grading'])
            frame[role]=state.value
            frame[f'{role}_grade']=state.grade
            frame[f'{role}_source_date']=state.source_date
            coverage.append({'cid':cid,'role':role,'xcat':xcat,'raw_rows':len(part),
                'monthly_eligible':int(state.value.notna().sum()),'source_grade_required':role not in MARKET_ROLES,
                'first_eligible':None if state.value.dropna().empty else str(state.value.dropna().index.min().date()),
                'last_eligible':None if state.value.dropna().empty else str(state.value.dropna().index.max().date())})
            daily[role]=part.set_index('real_date').value.reindex(calendar).astype(float)
        cds=daily['cds'].where(daily['cds']>0)
        log_cds=np.log(cds)
        cds_changes=100*log_cds.diff()
        frame['log_cds']=np.log(frame.cds.where(frame.cds>0))
        frame['cds_momentum']=100*(log_cds-log_cds.shift(63)).reindex(frame.index)
        frame['cds_vol']=cds_changes.rolling(126,min_periods=100).std(ddof=1).reindex(frame.index)*np.sqrt(252)
        returns=daily['bond_excess']
        frame['bond_momentum']=returns.rolling(63,min_periods=63).sum().reindex(frame.index)
        frame['bond_vol']=returns.rolling(126,min_periods=100).std(ddof=1).reindex(frame.index)*np.sqrt(252)
        for horizon in horizons:
            frame[f'cds_y_{horizon}']=(100*(log_cds.shift(-(horizon+1))-log_cds.shift(-1))).reindex(frame.index)
            frame[f'bond_y_{horizon}']=future_sum(returns,horizon).reindex(frame.index)
            frame[f'label_end_{horizon}']=pd.Series(calendar,index=calendar).shift(-(horizon+1)).reindex(frame.index)
        frame['underpriced']=-frame.gap
        frame['available_component_count']=frame[MACRO_ROLES].notna().sum(axis=1)
        frame['simple_macro']=frame[SIMPLE_ROLES].mean(axis=1).where(frame[SIMPLE_ROLES].notna().all(axis=1))
        frame['macro_change_3m']=frame.macro-frame.macro.shift(3)
        frame['gap_change_3m']=frame.underpriced-frame.underpriced.shift(3)
        frame['cid']=cid
        frame['real_date']=frame.index
        frame['month']=frame.real_date.dt.to_period('M')
        frames.append(frame.reset_index(drop=True))
    panel=pd.concat(frames,ignore_index=True).sort_values(['real_date','cid']).reset_index(drop=True)
    # Standardize both on the SAME currently observed CDS+three-factor universe.
    panel['simple_macro_z']=np.nan
    panel['market_cds_z']=np.nan
    for _,g in panel.groupby('real_date'):
        valid=g[g.simple_macro.notna() & g.log_cds.notna()]
        if len(valid)>=cfg['minimum_cross_section']:
            panel.loc[valid.index,'simple_macro_z']=cross_z(valid.simple_macro,cfg['minimum_cross_section'])
            panel.loc[valid.index,'market_cds_z']=cross_z(valid.log_cds,cfg['minimum_cross_section'])
    panel['simple_underpriced']=panel.simple_macro_z-panel.market_cds_z
    return panel,pd.DataFrame(coverage),calendar


class MonthBootstrap:
    def __init__(self,calendar,reps=499,block_months=6,seed=1729):
        self.calendar=pd.PeriodIndex(calendar,freq='M')
        self.ids=np.arange(len(calendar))//block_months
        n=self.ids.max()+1
        self.weights=np.random.default_rng(seed).multinomial(n,np.ones(n)/n,size=reps)
        self.block_months=block_months

    def row_ids(self,months):
        pos=self.calendar.get_indexer(pd.PeriodIndex(months,freq='M'))
        if (pos<0).any():raise ValueError('Month outside bootstrap calendar')
        return self.ids[pos]


def association_fit(data,signal,target,controls,boot):
    clean=data.dropna(subset=[signal,target,*controls]).copy()
    if len(clean)<200 or clean.cid.nunique()<8:
        return None,'too_few_complete_rows_or_countries'
    groups=[pd.factorize(clean.cid)[0],pd.factorize(clean.month)[0]]
    values=clean[[target,signal,*controls]].to_numpy(dtype=float)
    beta=within_beta(values,np.ones(len(clean)),groups)
    if beta is None:return None,'rank_deficient'
    blocks=boot.row_ids(clean.month)
    if len(np.unique(blocks))<10:return None,'fewer_than_10_calendar_blocks'
    draws=np.full(len(boot.weights),np.nan)
    for i,w in enumerate(boot.weights):
        weights=w[blocks].astype(float)
        if not weights.any():continue
        b=within_beta(values,weights,groups)
        if b is not None:draws[i]=b[0]
    valid=draws[np.isfinite(draws)]
    if len(valid)<.7*len(draws):return None,'too_few_valid_resamples'
    lo,hi=np.quantile(valid,[.025,.975])
    return {'beta':float(beta[0]),'ci_low':float(lo),'ci_high':float(hi),'n':len(clean),
        'countries':int(clean.cid.nunique()),'months':int(clean.month.nunique()),
        'start':str(clean.real_date.min().date()),'end':str(clean.real_date.max().date()),
        'controls':','.join(controls),'valid_bootstrap':len(valid),'draws':draws},'ok'


def apply_joint_bands(table,draws,index,columns=('joint_low','joint_high')):
    if not len(index):return
    matrix=np.column_stack([draws[i] for i in index])
    matrix=matrix[np.isfinite(matrix).all(axis=1)]
    if len(matrix)<40:return
    point=table.loc[index,'beta'].to_numpy()
    std=matrix.std(axis=0,ddof=1)
    if (std<=1e-12).any():return
    critical=np.quantile(np.max(np.abs((matrix-point)/std),axis=1),.95)
    table.loc[index,columns[0]]=point-critical*std
    table.loc[index,columns[1]]=point+critical*std


def associations(panel,cfg,horizons=(21,63,126)):
    calendar=pd.period_range(panel.month.min(),panel.month.max(),freq='M')
    rows,draws=[],{}
    for block in cfg['block_months']:
        boot=MonthBootstrap(calendar,cfg['bootstrap_reps'],block)
        for outcome in ['cds','bond']:
            controls=(['log_cds','cds_momentum','cds_vol'] if outcome=='cds' else ['bond_carry','bond_momentum','bond_vol'])
            for signal in ['underpriced','simple_underpriced']:
                for horizon in horizons:
                    target=f'{outcome}_y_{horizon}'
                    fit,status=association_fit(panel,signal,target,controls,boot)
                    row={'outcome':outcome,'signal':signal,'horizon':horizon,'block_months':block,
                        'status':status,'beta':np.nan,'ci_low':np.nan,'ci_high':np.nan,'n':0,
                        'expected_sign':'positive' if outcome=='cds' else 'negative',
                        'unit':'CDS_log_change_pp_per_score' if outcome=='cds' else 'bond_notional_excess_return_pp_per_score',
                        'interpretation':'historical_association_model_vintages_unverified'}
                    if fit:
                        draws[len(rows)]=fit.pop('draws');row.update(fit)
                    rows.append(row)
    result=pd.DataFrame(rows);result['joint_low']=np.nan;result['joint_high']=np.nan
    # Joint bands cover the four predeclared primary 63-day outcome/signal tests.
    for block in cfg['block_months']:
        idx=result.index[result.block_months.eq(block)&result.horizon.eq(63)&result.status.eq('ok')]
        apply_joint_bands(result,draws,idx)
    return result


def fit_ridge(train,features,target,alpha):
    """Train-only scaling, feature ridge; country intercepts not penalized."""
    mean=train[features].mean().to_numpy();std=train[features].std(ddof=1).to_numpy()
    std=np.where(std>1e-10,std,1.)
    cids=sorted(train.cid.unique())
    x=train[features].to_numpy(dtype=float)
    dummies=np.column_stack([(train.cid==c).to_numpy(dtype=float) for c in cids[1:]]) if len(cids)>1 else np.empty((len(train),0))
    design=np.column_stack([np.ones(len(train)),(x-mean)/std,dummies])
    ridge=np.zeros(design.shape[1]);ridge[1:1+len(features)]=alpha
    beta=np.linalg.solve(design.T@design+np.diag(ridge),design.T@train[target].to_numpy(dtype=float))
    return {'features':features,'mean':mean,'std':std,'cids':cids,'beta':beta}


def predict_ridge(model,test):
    if not test.cid.isin(model['cids']).all():raise ValueError('Unseen country cannot get a fixed effect')
    x=test[model['features']].to_numpy(dtype=float)
    dummies=np.column_stack([(test.cid==c).to_numpy(dtype=float) for c in model['cids'][1:]]) if len(model['cids'])>1 else np.empty((len(test),0))
    return np.column_stack([np.ones(len(test)),(x-model['mean'])/model['std'],dummies])@model['beta']


def eligible_training(data,cutoff,label_end):
    # Purge the whole overlapping label, not just the information origin.
    return data.real_date.lt(cutoff) & data[label_end].lt(cutoff)


def chronological_validation(panel,cfg):
    predictions,audit=[],[]
    for outcome in ['cds','bond']:
        controls=(['log_cds','cds_momentum','cds_vol'] if outcome=='cds' else ['bond_carry','bond_momentum','bond_vol'])
        for signal in ['underpriced','simple_underpriced']:
            for horizon in [21,63,126]:
                target,end=f'{outcome}_y_{horizon}',f'label_end_{horizon}'
                data=panel.dropna(subset=[target,end,signal,*controls]).copy()
                # Relative repricing target. Both observed and forecast values
                # are centered on the SAME evaluation universe for each month.
                data['target_relative']=data[target]-data.groupby('real_date')[target].transform('mean')
                data=data[data.groupby('real_date').cid.transform('count').ge(cfg['minimum_cross_section'])]
                for year in range(cfg['first_validation_year'],int(panel.real_date.dt.year.max())+1):
                    start=pd.Timestamp(year=year,month=1,day=1)
                    stop=pd.Timestamp(year=year+1,month=1,day=1)
                    train=data[eligible_training(data,start,end)].copy()
                    counts=train.groupby('cid').size();known=set(counts[counts>=12].index)
                    train=train[train.cid.isin(known)]
                    test=data[data.real_date.ge(start)&data.real_date.lt(stop)&data.cid.isin(known)].copy()
                    train=train[train.groupby('real_date').cid.transform('count').ge(cfg['minimum_cross_section'])]
                    # A country can lose all training rows after the monthly
                    # completeness screen; it must not receive an unseen effect.
                    test=test[test.cid.isin(train.cid.unique())]
                    test=test[test.groupby('real_date').cid.transform('count').ge(cfg['minimum_cross_section'])]
                    train['target_relative']=train[target]-train.groupby('real_date')[target].transform('mean')
                    status='ok'
                    if len(train)<cfg['minimum_training_rows'] or train.month.nunique()<cfg['minimum_training_months']:
                        status='insufficient_training'
                    elif test.empty:status='no_complete_test_labels'
                    # Center targets after applying the evaluation-country mask.
                    test['target_relative']=test[target]-test.groupby('real_date')[target].transform('mean')
                    entry={'outcome':outcome,'signal':signal,'horizon':horizon,'year':year,'status':status,
                        'training_rows':len(train),'training_months':int(train.month.nunique()),'test_rows':len(test),
                        'max_training_origin':None if train.empty else str(train.real_date.max().date()),
                        'max_training_label_end':None if train.empty else str(train[end].max().date()),
                        'first_test_origin':None if test.empty else str(test.real_date.min().date()),
                        'scores_historical_vintages_verified':False}
                    audit.append(entry)
                    if status!='ok':continue
                    base=fit_ridge(train,controls,'target_relative',cfg['ridge_alpha'])
                    enhanced=fit_ridge(train,[*controls,signal],'target_relative',cfg['ridge_alpha'])
                    test['baseline_prediction']=predict_ridge(base,test)
                    test['enhanced_prediction']=predict_ridge(enhanced,test)
                    for col in ['baseline_prediction','enhanced_prediction']:
                        test[col]-=test.groupby('real_date')[col].transform('mean')
                    keep=['real_date','cid','month',target,end,'target_relative','baseline_prediction','enhanced_prediction']
                    p=test[keep].rename(columns={target:'observed_raw',end:'label_end'})
                    p=p.assign(outcome=outcome,signal=signal,horizon=horizon,test_year=year)
                    predictions.append(p)
    return (pd.concat(predictions,ignore_index=True) if predictions else pd.DataFrame()),pd.DataFrame(audit)


def paired_validation_metrics(predictions,cfg):
    rows,draws=[],{}
    if predictions.empty:return pd.DataFrame()
    calendar=pd.period_range(predictions.month.min(),predictions.month.max(),freq='M')
    for (outcome,signal,horizon),g in predictions.groupby(['outcome','signal','horizon']):
        g=g.copy()
        g['base_se']=(g.baseline_prediction-g.target_relative)**2
        g['enhanced_se']=(g.enhanced_prediction-g.target_relative)**2
        monthly=g.groupby('month')[['base_se','enhanced_se']].mean()
        # Equal month weighting prevents a changing number of countries from
        # silently changing the importance of a historical period.
        base,enhanced=monthly.mean()
        rank_base=[];rank_enhanced=[]
        for _,part in g.groupby('real_date'):
            for pred,collector in [('baseline_prediction',rank_base),('enhanced_prediction',rank_enhanced)]:
                collector.append(part[pred].rank().corr(part.target_relative.rank()))
        for block in cfg['block_months']:
            boot=MonthBootstrap(calendar,cfg['bootstrap_reps'],block)
            ids=boot.row_ids(monthly.index)
            samples=[]
            if len(np.unique(ids))>=10:
                for w in boot.weights:
                    weights=w[ids].astype(float)
                    if weights.sum()==0: samples.append(np.nan);continue
                    b,e=np.average(monthly.to_numpy(),axis=0,weights=weights)
                    samples.append(1-e/b if b>1e-12 else np.nan)
            samples=np.array(samples);valid=samples[np.isfinite(samples)]
            status='ok' if len(valid)>=.7*cfg['bootstrap_reps'] else 'insufficient_calendar_blocks_or_valid_draws'
            row={'outcome':outcome,'signal':signal,'horizon':horizon,'block_months':block,'status':status,
                'beta':float(1-enhanced/base),'baseline_mse':float(base),'enhanced_mse':float(enhanced),
                'ci_low':np.nan,'ci_high':np.nan,'n_predictions':len(g),'evaluation_months':len(monthly),
                'countries':int(g.cid.nunique()),'start':str(g.real_date.min().date()),'end':str(g.real_date.max().date()),
                'baseline_rank_correlation':float(np.nanmean(rank_base)),
                'enhanced_rank_correlation':float(np.nanmean(rank_enhanced)),
                'interpretation':'chronological_relative_validation_not_certified_PIT_or_trading_pnl'}
            if status=='ok':
                row['ci_low'],row['ci_high']=np.quantile(valid,[.025,.975])
                draws[len(rows)]=samples
            rows.append(row)
    result=pd.DataFrame(rows);result['joint_low']=np.nan;result['joint_high']=np.nan
    for block in cfg['block_months']:
        idx=result.index[result.block_months.eq(block)&result.horizon.eq(63)&result.status.eq('ok')]
        apply_joint_bands(result,draws,idx)
    result=result.rename(columns={'beta':'relative_mse_improvement'})
    return result
