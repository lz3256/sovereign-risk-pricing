"""Descriptive price/financing feedback: latest revised quarterly observations.

No macro observation is represented as information available at its period end.
This module deliberately does not construct a trading signal or PIT backtest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def quarter_spot(spot, cutoff, max_quote_age_days=7):
    """Use completed quarters only and retain the actual last quotation date."""
    cutoff = pd.Timestamp(cutoff)
    data = spot.copy()
    data['real_date'] = pd.to_datetime(data.real_date)
    data = data[data.real_date.le(cutoff)].sort_values(['cid','real_date'])
    data['quarter'] = data.real_date.dt.to_period('Q')
    last = data.groupby(['cid','quarter'],as_index=False).tail(1).copy()
    last['period_end'] = last.quarter.dt.end_time.dt.normalize()
    last = last[last.period_end.le(cutoff)].copy()
    last['quote_age_days'] = (last.period_end - last.real_date).dt.days
    last['quarter_spot'] = last.spot.where(last.quote_age_days.le(max_quote_age_days))
    if last.duplicated(['cid','quarter']).any():
        raise ValueError('Duplicate quarterly spot key')
    return last[['cid','quarter','period_end','real_date','quote_age_days','quarter_spot']].rename(
        columns={'real_date':'quote_date'})


def build_quarterly_panel(spot,lbs,cutoff):
    prices = quarter_spot(spot,cutoff)
    bank = lbs.copy()
    bank['quarter'] = pd.to_datetime(bank.real_date).dt.to_period('Q')
    cols = ['L_MEASURE','L_DENOM','L_CP_SECTOR']
    if bank.duplicated(['cid','quarter',*cols]).any():
        raise ValueError('Duplicate quarterly bank data')
    bank['key'] = bank.L_MEASURE + '_' + bank.L_DENOM + '_' + bank.L_CP_SECTOR
    wide = bank.pivot(index=['cid','quarter'],columns='key',values='value').reset_index()
    end = pd.Timestamp(cutoff).to_period('Q')
    if end.end_time.normalize() > pd.Timestamp(cutoff):
        end -= 1
    start = min(prices.quarter.min(),bank.quarter.min())
    quarters = pd.period_range(start,end,freq='Q')
    cids = sorted(set(prices.cid)&set(bank.cid))
    grid = pd.MultiIndex.from_product([cids,quarters],names=['cid','quarter']).to_frame(index=False)
    panel = grid.merge(prices,on=['cid','quarter'],how='left',validate='one_to_one').merge(
        wide,on=['cid','quarter'],how='left',validate='one_to_one').sort_values(['cid','quarter'])
    panel['depreciation'] = panel.groupby('cid',sort=False).quarter_spot.transform(lambda x:100*np.log(x/x.shift(1)))
    for lag in [1,2]:
        panel[f'dep_lag{lag}'] = panel.groupby('cid',sort=False).depreciation.shift(lag)
    panel['dep_next'] = panel.groupby('cid',sort=False).depreciation.shift(-1)
    for denom,sector in [('USD','N'),('TO1','N'),('USD','C')]:
        key = f'{denom}_{sector}'
        stock = panel[f'S_{key}']
        previous = stock.groupby(panel.cid,sort=False).shift(1)
        panel[f'flow_{key}'] = (100*panel[f'F_{key}']/previous).where(previous.gt(0))
        panel[f'flow_lag_{key}'] = panel.groupby('cid',sort=False)[f'flow_{key}'].shift(1)
    # Country-excluded median of currencies is only a broad common FX proxy,
    # not a trade-weighted dollar index. Used only in the one-country case.
    cross = panel.pivot(index='quarter',columns='cid',values='depreciation')
    common = cross.drop(columns='BRL',errors='ignore').median(axis=1)
    panel['common_fx'] = panel.quarter.map(common)
    panel['common_fx_lag'] = panel.quarter.map(common.shift(1))
    panel['common_fx_next'] = panel.quarter.map(common.shift(-1))
    panel['historical_available_at'] = None
    panel['use_policy'] = 'retrospective_mechanism_only_latest_revised_snapshot'
    return panel.reset_index(drop=True)


def weighted_within(values,weights,groups,tolerance=1e-10,max_iterations=300):
    """Alternating weighted projections for unbalanced country/time fixed effects."""
    residual = np.asarray(values,dtype=float).copy()
    weights = np.asarray(weights,dtype=float)
    if np.any(weights<0) or not np.any(weights>0):
        raise ValueError('Weights must be nonnegative and not all zero')
    for _ in range(max_iterations):
        largest = 0.
        for codes in groups:
            codes = np.asarray(codes,dtype=int)
            denominator = np.bincount(codes,weights=weights)
            numerator = np.column_stack([np.bincount(codes,weights=weights*residual[:,j],
                minlength=len(denominator)) for j in range(residual.shape[1])])
            means = np.divide(numerator,denominator[:,None],out=np.zeros_like(numerator),where=denominator[:,None]>0)
            adjustment = means[codes]
            largest = max(largest,float(np.max(np.abs(adjustment[weights>0]))))
            residual -= adjustment
        if largest < tolerance:
            return residual
    raise ValueError('Fixed-effect projection did not converge')


def within_beta(values,weights,groups):
    residual = weighted_within(values,weights,groups)
    x,y = residual[:,1:],residual[:,0]
    xx = x.T @ (weights[:,None]*x)
    xy = x.T @ (weights*y)
    if np.linalg.matrix_rank(xx) != xx.shape[0]:
        return None
    return np.linalg.solve(xx,xy)


class QuarterBootstrap:
    def __init__(self,calendar,reps=499,block_quarters=4,seed=1729):
        self.calendar = pd.PeriodIndex(calendar,freq='Q')
        self.block_ids = np.arange(len(calendar))//block_quarters
        n = self.block_ids.max()+1
        self.weights = np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=reps)
        self.block_quarters = block_quarters

    def row_blocks(self,quarters):
        pos = self.calendar.get_indexer(pd.PeriodIndex(quarters,freq='Q'))
        if (pos<0).any():
            raise ValueError('Quarter outside bootstrap calendar')
        return self.block_ids[pos]


def fit_quarterly(data,signal,target,controls,bootstrap,time_effects=True,min_rows=80):
    clean = data.dropna(subset=[signal,target,*controls]).copy()
    if len(clean)<min_rows:
        return None,'too_few_complete_quarters'
    columns = [signal,*controls]
    groups = [pd.factorize(clean.cid)[0]]
    if time_effects:
        groups.append(pd.factorize(clean.quarter)[0])
    values = clean[[target,*columns]].to_numpy(dtype=float)
    beta = within_beta(values,np.ones(len(clean)),groups)
    if beta is None:
        return None,'rank_deficient'
    blocks = bootstrap.row_blocks(clean.quarter)
    occupied = len(np.unique(blocks))
    if occupied < 10:
        return None,'fewer_than_10_time_blocks'
    samples = np.full(len(bootstrap.weights),np.nan)
    for i,w in enumerate(bootstrap.weights):
        weights = w[blocks].astype(float)
        if not weights.any():
            continue
        sample = within_beta(values,weights,groups)
        if sample is not None:
            samples[i] = sample[0]
    good = samples[np.isfinite(samples)]
    if len(good)<max(40,int(.7*len(samples))):
        return None,'insufficient_valid_bootstrap_draws'
    low,high = np.quantile(good,[.025,.975])
    return {'beta':float(beta[0]),'ci_low':float(low),'ci_high':float(high),
        'n':len(clean),'countries':int(clean.cid.nunique()),'quarters':int(clean.quarter.nunique()),
        'start':str(clean.quarter.min()),'end':str(clean.quarter.max()),
        'blocks':occupied,'valid_bootstrap':len(good),'block_quarters':bootstrap.block_quarters,
        'controls':','.join(controls),'time_fixed_effects':time_effects,'samples':samples},'ok'


def financing_models(panel,reps=499):
    calendar = pd.period_range(panel.quarter.min(),panel.quarter.max(),freq='Q')
    rows,draws = [],{}
    # Primary: USD loans/deposits to nonbanks. Others are predeclared sensitivity
    # specifications; neither sectors nor currency totals are added together.
    for block in [4,8]:
        boot = QuarterBootstrap(calendar,reps,block)
        for scope in ['panel_21','Brazil']:
            subset = panel if scope=='panel_21' else panel[panel.cid.eq('BRL')]
            for family in ['USD_N','TO1_N','USD_C']:
                flow,prior = f'flow_{family}',f'flow_lag_{family}'
                specifications = [
                    ('A_past_depreciation_to_financing','dep_lag1',flow,['dep_lag2',prior]),
                    ('B_financing_to_future_depreciation',flow,'dep_next',['depreciation',prior])]
                for stage,signal,target,controls in specifications:
                    if scope=='Brazil':
                        controls = controls + (['common_fx','common_fx_lag'] if stage.startswith('A_') else ['common_fx_next'])
                    fit,status = fit_quarterly(subset,signal,target,controls,boot,
                        time_effects=scope=='panel_21',min_rows=80 if scope=='panel_21' else 40)
                    row = {'scope':scope,'family':family,'stage':stage,'status':status,
                      'primary':family=='USD_N','block_quarters':block,
                      'beta':np.nan,'ci_low':np.nan,'ci_high':np.nan,'n':0,
                      'expected_feedback_sign':'negative','inference':'retrospective_association_not_causal_or_PIT',
                      'unit':'flow_pp_per_depreciation_pp' if stage.startswith('A_') else 'depreciation_pp_per_flow_pp'}
                    if fit:
                        draws[len(rows)] = fit.pop('samples')
                        row.update(fit)
                    rows.append(row)
    result = pd.DataFrame(rows)
    result['primary_family_sim_low'] = np.nan
    result['primary_family_sim_high'] = np.nan
    # Joint bands across the four primary panel/case/stage comparisons, separately
    # for each predeclared block length. No claim of causal mediation.
    for block in [4,8]:
        g = result[result.primary & result.block_quarters.eq(block) & result.status.eq('ok')]
        if g.empty:
            continue
        matrix = np.column_stack([draws[i] for i in g.index])
        matrix = matrix[np.isfinite(matrix).all(axis=1)]
        if len(matrix)<40:
            continue
        std = matrix.std(axis=0,ddof=1)
        if (std<=1e-12).any():
            continue
        point = g.beta.to_numpy()
        critical = np.quantile(np.max(np.abs((matrix-point)/std),axis=1),.95)
        result.loc[g.index,'primary_family_sim_low'] = point-critical*std
        result.loc[g.index,'primary_family_sim_high'] = point+critical*std
    return result
