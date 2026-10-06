import unittest
import numpy as np
import pandas as pd
from reflexivity.sovereign import (future_sum, monthly_state, build_panel, fit_ridge,
    predict_ridge, eligible_training, chronological_validation)


class SovereignIntegrity(unittest.TestCase):
    def test_execution_delay_and_missing_return_window(self):
        s=pd.Series(np.arange(10,dtype=float))
        self.assertEqual(future_sum(s,3).iloc[0],2+3+4)
        s.iloc[3]=np.nan
        self.assertTrue(pd.isna(future_sum(s,3).iloc[0]))
        self.assertTrue(pd.isna(future_sum(s,3).iloc[-1]))

    def test_new_bad_grade_does_not_resurrect_old_value(self):
        dates=pd.bdate_range('2024-01-01',periods=7)
        part=pd.DataFrame({'real_date':dates[[0,3]],'value':[2.,9.],'grading':[1.,3.]})
        state=monthly_state(part,dates,dates,3,True,2)
        self.assertEqual(state.value.iloc[2],2.)
        self.assertTrue(state.value.iloc[3:].isna().all())
        self.assertTrue((state.source_date.dropna()<=state.source_date.dropna().index).all())

    def test_quote_age_is_business_days_not_reference_period_lag(self):
        dates=pd.bdate_range('2024-01-05',periods=5)
        p=pd.DataFrame({'real_date':[dates[0]],'value':[7.],'grading':[1.],'eop_lag':[99.]})
        state=monthly_state(p,dates,dates,3,False,2)
        self.assertEqual(state.value.iloc[3],7.)
        self.assertTrue(pd.isna(state.value.iloc[4]))

    def test_absent_category_stays_missing_without_type_failure(self):
        dates=pd.bdate_range('2024-01-01',periods=5)
        part=pd.DataFrame(columns=['real_date','value','grading'])
        state=monthly_state(part,dates,dates,3,True,2)
        self.assertTrue(state.value.isna().all())
        self.assertTrue(state.source_date.isna().all())

    def test_complete_weekend_month_and_partial_month(self):
        roles={'cds':'C','bond_excess':'B','gap':'P','macro':'M','fiscal':'F','external':'E',
               'growth':'G','investment':'I','fx_debt':'D','governance':'V','inflation':'N'}
        cfg={'cids':['BRL'],'categories':roles,'start_date':'2023-09-01','end_date':'2023-10-03',
             'month_max_quote_age_business_days':3,'max_grading':2,'minimum_cross_section':8}
        dates=pd.bdate_range('2023-09-01','2023-10-03')
        raw=pd.concat([pd.DataFrame({'real_date':dates,'cid':'BRL','xcat':x,'value':1.,'grading':1.})
                       for x in roles.values()],ignore_index=True)
        panel,_,_=build_panel(raw,cfg)
        self.assertEqual(panel.real_date.tolist(),[pd.Timestamp('2023-09-29')])
        self.assertTrue(panel.simple_underpriced.isna().all())

    def test_training_purges_boundary_and_train_only_scaling(self):
        data=pd.DataFrame({'real_date':pd.to_datetime(['2010-12-01','2010-11-01','2011-01-01']),
             'end':pd.to_datetime(['2011-01-01','2010-12-31','2011-02-01'])})
        self.assertEqual(eligible_training(data,pd.Timestamp('2011-01-01'),'end').tolist(),[False,True,False])
        train=pd.DataFrame({'cid':['A','A','B','B'],'x':[1.,2.,3.,4.],'y':[2.,4.,6.,8.]})
        model=fit_ridge(train,['x'],'y',10)
        np.testing.assert_allclose(model['mean'],[2.5])
        with self.assertRaises(ValueError):predict_ridge(model,pd.DataFrame({'cid':['C'],'x':[99.]}))

    def test_future_test_outcomes_do_not_change_same_year_forecasts(self):
        rng=np.random.default_rng(51);months=pd.period_range('2006-01','2013-12',freq='M')
        rows=[]
        for month in months:
            date=pd.bdate_range(month.start_time,month.end_time)[-1]
            for cid in 'ABCDEFGH':
                r={'real_date':date,'month':month,'cid':cid}
                for x in ['underpriced','simple_underpriced','log_cds','cds_momentum','cds_vol',
                          'bond_carry','bond_momentum','bond_vol']:r[x]=rng.normal()
                for h in [21,63,126]:
                    r[f'label_end_{h}']=date+pd.offsets.BDay(h+1)
                    r[f'cds_y_{h}']=r['underpriced']+rng.normal()
                    r[f'bond_y_{h}']=-r['underpriced']+rng.normal()
                rows.append(r)
        panel=pd.DataFrame(rows)
        cfg={'first_validation_year':2012,'minimum_cross_section':8,'minimum_training_rows':200,
             'minimum_training_months':24,'ridge_alpha':10.}
        p,a=chronological_validation(panel,cfg)
        changed=panel.copy();future=changed.real_date.dt.year.ge(2012)
        for h in [21,63,126]:
            for o in ['cds','bond']:changed.loc[future,f'{o}_y_{h}']*=1000
        q,_=chronological_validation(changed,cfg)
        columns=['baseline_prediction','enhanced_prediction']
        np.testing.assert_allclose(p.loc[p.test_year.eq(2012),columns],q.loc[q.test_year.eq(2012),columns])
        good=a[a.status.eq('ok')]
        self.assertTrue((pd.to_datetime(good.max_training_label_end)<pd.to_datetime(good.first_test_origin)).all())
        self.assertTrue(p.groupby(['outcome','signal','horizon','real_date']).size().ge(8).all())
        self.assertLess(p.groupby(['outcome','signal','horizon','real_date']).baseline_prediction.mean().abs().max(),1e-10)

        # Z has enough raw training observations, but all fall in months that
        # become incomplete when rarely observed countries are removed.
        edge=panel[~(panel.real_date.dt.year.isin([2010,2011])&panel.cid.isin(['G','H']))].copy()
        rare=panel[panel.cid.eq('A')&panel.real_date.dt.year.isin([2010,2011])].copy()
        additions=[rare.assign(cid='Z')]
        for suffix in ['a','b']:
            additions.append(rare.assign(cid=[f'U{m}{suffix}' for m in rare.month]))
        additions.append(panel[panel.cid.eq('A')&panel.real_date.dt.year.eq(2012)].assign(cid='Z'))
        edge=pd.concat([edge,*additions],ignore_index=True)
        edge_pred,_=chronological_validation(edge,cfg)
        self.assertFalse(edge_pred.cid.eq('Z').any())

if __name__=='__main__':unittest.main()
