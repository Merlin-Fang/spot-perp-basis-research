"""Reproduce the inspected 2022 four-asset forecast checkpoint, never final data.

Run from the repository root with python -B -m scripts.checkpoint_survival_baseline.
Uses archived May-2021 fits with maintained predictors; does not refit or backtest.
Only compact summaries/provenance are written; predictions stay in memory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from spot_perp_basis.data.features import FEATURES, features
from spot_perp_basis.data.labels import labels
from spot_perp_basis.data.windows import require_training_times
from spot_perp_basis.evaluation.statistics import block_interval
from spot_perp_basis.evaluation.survival_audit import matched_ipcw_scores, probability_audit
from spot_perp_basis.models import logistic, trees
from spot_perp_basis.models.survival import km_fit, km_predict
from spot_perp_basis.theory import theory_signal, cost_bound
from spot_perp_basis.training.fit import hourly_risk_set, array_hash


ROOT = Path(__file__).resolve().parents[1]
B36 = 'experiments/experiment3_6_nonlinear_survival'
B37 = 'experiments/experiment3_7_controlled_risk_value'
ASSETS = ('BTCUSDT', 'ETHUSDT', 'DOGEUSDT', 'ADAUSDT')
BTC_ETH = ASSETS[:2]
RATES = ('zero_rate', 'nyfed_effr_lag7d')
START = pd.Timestamp('2020-01-01', tz='UTC')
CUTOFF = pd.Timestamp('2021-05-01', tz='UTC')
EVAL = pd.Timestamp('2022-01-01', tz='UTC')
END = pd.Timestamp('2023-01-01', tz='UTC')
OUTPUT = ROOT/'results/checkpoints/2026-11-09'


def origin_segment(frame, begin, end):
    """Origin lower bound + strict 16h purge; retain earlier context for warmup."""
    begin, end = pd.Timestamp(begin), pd.Timestamp(end)
    if begin.tz is None or end.tz is None or not START <= begin < end <= END:
        raise ValueError('Only explicit timezone-aware development windows')
    # Check before invoking the label builder, not after reading later data.
    require_training_times(frame.index)
    if len(frame) and not frame.index.equals(pd.date_range(frame.index.min(),frame.index.max(),freq='h',name=frame.index.name)):
        raise ValueError('Require an explicit hourly grid; retain missing-hour rows')
    p = labels(frame, begin, end)
    return frame.reindex(p.index), p


class Sources:
    """Exact source allowlist plus Python IO/network denial for the command."""
    def __init__(self, rates):
        self.allowed = {f'{B36}/{n}' for n in ('config.json','selection.json','split_manifest.json','input_receipt.json')}
        self.allowed |= {f'{B37}/{n}' for n in ('config.json','selection.json','statistical_results.csv','attribution_differences.csv','statistical_calendar_uncertainty.csv')}
        for symbol in ASSETS:
            for rate in rates:
                stem = f'{symbol}_{rate}'
                self.allowed |= {
                    f'{B36}/causal_inputs_v1/{stem}.parquet',
                    f'{B36}/fits_v1/final/{stem}_minimal.json',
                    f'{B37}/fits_v1/final/{stem}_minimal.json',
                    f'{B36}/risk_sets_v1/{stem}_minimal_fitmay2021.parquet',
                    f'{B36}/labels_v1/{stem}_fitmay2021.parquet',
                    f'{B36}/labels_v1/{stem}_development2022.parquet',
                    f'{B37}/forecasts_v1/development2022/{stem}_minimal.parquet',
                    f'{B37}/forecasts_v1/development2022/{stem}_minimal.parquet.receipt.json',
                }
            if symbol in BTC_ETH:
                self.allowed.add(f'data/processed/five_asset_v1/minute/{symbol}/2022-01.parquet')
        self.allowed.add('manifests/five_asset_v1_dataset.json')
        self.hashes = {}

    def guard(self, event, args):
        if event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('Checkpoint is offline')
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
        if not path.is_relative_to(ROOT):
            return
        rel = str(path.relative_to(ROOT))
        if rel.split('/')[0] in ('experiments', 'data', 'manifests'):
            if rel not in self.allowed:
                raise PermissionError('Not an allowed pre-2023 checkpoint source: '+rel)
            mode = args[1] or ''
            flags = args[2]
            if any(c in str(mode) for c in 'wa+') or (flags & 3):
                raise PermissionError('Research inputs are read-only')

    def path(self, relative):
        if relative not in self.allowed:
            raise ValueError('Source outside the explicit pre-2023 allowlist')
        path = ROOT/relative
        self.hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def json(self, relative):
        return json.loads(self.path(relative).read_text())

    def frame(self, relative, *, begin=START, end=END, columns=None):
        if not START <= begin < end <= END:
            raise ValueError('Forbidden source period before IO')
        # Every frame in this allowlist is a development-only physical artifact.
        key = 'decision_time_utc' if relative.startswith('data/') else 'origin_time_utc'
        d = pd.read_parquet(self.path(relative), columns=columns,
                            filters=[(key,'>=',begin),(key,'<',end)])
        clock = pd.DatetimeIndex(d[key]) if key in d else d.index
        require_training_times(clock.sort_values() if '/risk_sets_v1/' in relative else clock,
                               allow_duplicates='/risk_sets_v1/' in relative)
        if len(clock) and (clock.min() < begin or clock.max() >= end):
            raise ValueError('Source filtering failed')
        return d


def check_fit(src, symbol, rate, f):
    """Reconstruct the original risk matrix, training transforms and both KMs."""
    stem = f'{symbol}_{rate}'
    old = src.json(f'{B36}/fits_v1/final/{stem}_minimal.json')
    new = src.json(f'{B37}/fits_v1/final/{stem}_minimal.json')
    assert pd.Timestamp(old['fit_cutoff_utc']) == pd.Timestamp(new['fit_cutoff_utc']) == CUTOFF
    assert old['features'] == new['matrix_manifest']['feature_columns'] == list(FEATURES['minimal'])
    ff, lab = origin_segment(f, START, CUTOFF)
    stored = src.frame(f'{B36}/labels_v1/{stem}_fitmay2021.parquet', end=CUTOFF)
    pd.testing.assert_frame_equal(lab, stored)
    risk = hourly_risk_set(ff, lab, 'minimal', CUTOFF)
    stored_risk = src.frame(f'{B36}/risk_sets_v1/{stem}_minimal_fitmay2021.parquet', end=CUTOFF)
    pd.testing.assert_frame_equal(risk, stored_risk)
    assert (risk.origin_time_utc >= START).all() and risk.available_at_utc.lt(CUTOFF).all()
    x = ff.reindex(pd.DatetimeIndex(risk.origin_time_utc))[list(FEATURES['minimal'])].to_numpy()
    raw = np.c_[x, risk.hazard_hour.to_numpy()]
    assert array_hash(raw) == new['matrix_manifest']['raw_X_and_hour_sha256']
    assert array_hash(risk.response.to_numpy(float)) == new['matrix_manifest']['response_sha256']
    assert src.hashes[f'{B36}/fits_v1/final/{stem}_minimal.json'] == new['matrix_manifest']['original_tree_fit_sha256']
    assert src.hashes[f'{B36}/risk_sets_v1/{stem}_minimal_fitmay2021.parquet'] == old['riskset_sha256']
    tree, linear = old['boosters'][0], new['models']['matched_hourly']
    assert (tree['depth'],tree['learning_rate'],old['selected_rounds'],linear['alpha'],linear['K']) == (3,.1,80,.001,16)
    np.testing.assert_allclose(x.mean(axis=0), linear['mean'], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.where(x.std(axis=0)>1e-12,x.std(axis=0),1), linear['scale'], atol=1e-12, rtol=1e-12)
    for j, cuts in enumerate(tree['cuts']):
        np.testing.assert_array_equal(np.unique(np.quantile(raw[:,j],np.arange(1,32)/32)), cuts)
    m = ff.minimal_eligible36 & ~lab.boundary_window_purged & lab.survival_origin_eligible
    censor = km_fit(lab.loc[m,'followup_hours'],lab.loc[m,'event_observed'],reverse=True)
    assert censor == old['scoring_censor_km'] == new['scoring_censor_km']
    em = ff.minimal_eligible & ff.rho.ne(0) & ~lab.boundary_window_purged & lab.survival_origin_eligible
    own = km_fit(lab.loc[em,'followup_hours'],lab.loc[em,'event_observed'])
    assert own == old['baselines']['asset_km']
    return tree, linear, own, censor, {
        'source_first_origin':str(f.index.min()),
        'unique_training_origins': risk.origin_time_utc.nunique(), 'training_trials': len(risk),
        'training_event_trials': int(risk.response.sum()), 'censor_training_origins': int(m.sum()),
        'KM_training_origins': int(em.sum()), 'maximum_trial_publication': str(risk.available_at_utc.max()),
        'raw_matrix_sha256': array_hash(raw), 'response_sha256': array_hash(risk.response.to_numpy(float)),
        'same_risk_set_and_training_transforms_verified': True}


def minute_discretization(src, symbol, context):
    """Fixed January 2–8, 2022 origins, full 8h paths, exact zero-finance rho."""
    begin = pd.Timestamp('2022-01-02', tz='UTC')
    stop = pd.Timestamp('2022-01-09', tz='UTC')
    end = stop+pd.Timedelta(hours=8, minutes=1)
    rel = f'data/processed/five_asset_v1/minute/{symbol}/2022-01.parquet'
    info = src.json('manifests/five_asset_v1_dataset.json')
    expected = next(x['sha256'] for x in info['partitions'] if x['path']==rel)
    d = src.frame(rel, begin=begin, end=end,
        columns=['spot_close','perp_close','paired_valid','active_sample'])
    assert src.hashes[rel] == expected
    clock = pd.date_range(begin,end,freq='min',inclusive='left',name=d.index.name)
    d = d.reindex(clock)
    valid = d.paired_valid.eq(True) & d.active_sample.eq(True)
    rho = pd.Series(np.nan,index=clock)
    rho.loc[valid] = theory_signal(d.loc[valid,'spot_close'],d.loc[valid,'perp_close'],0.)[0]
    origins = pd.date_range(begin,stop,freq='h',inclusive='left')
    expected_rho = context.rho.reindex(origins)
    np.testing.assert_allclose(rho.reindex(origins),expected_rho,equal_nan=True,atol=1e-12,rtol=1e-12)
    result = {'symbol':symbol,'origin_start':str(begin),'origin_end_exclusive':str(stop),
              'horizon_hours':8,'candidate_origins':len(origins),'complete_active_paths':0,
              'invalid_or_zero_origin':0,'missing_or_inactive_path':0,
              'minute_events_by_8h':0,'hourly_events_by_8h':0,'binary_disagreements':0,
              'first_crossing_time_disagreements':0,'low_fee_qualified_complete_paths':0}
    for t in origins:
        v = rho.loc[t:t+pd.Timedelta(hours=8)].to_numpy()
        if not np.isfinite(v[0]) or v[0]==0:
            result['invalid_or_zero_origin'] += 1
            continue
        if len(v)!=481 or not np.isfinite(v).all():
            result['missing_or_inactive_path'] += 1
            continue
        direction = -np.sign(v[0])
        crosses = v[1:]<=0 if direction==-1 else v[1:]>=0
        minute = np.flatnonzero(crosses)
        hourly = np.flatnonzero(crosses[59::60])
        result['complete_active_paths'] += 1
        result['minute_events_by_8h'] += int(len(minute)>0)
        result['hourly_events_by_8h'] += int(len(hourly)>0)
        result['binary_disagreements'] += int(bool(len(minute))!=bool(len(hourly)))
        result['first_crossing_time_disagreements'] += int(len(minute)>0 and len(hourly)>0 and minute[0]+1 != (hourly[0]+1)*60)
        result['low_fee_qualified_complete_paths'] += int(abs(v[0])>cost_bound(2.25,.18))
    result['binary_disagreement_fraction'] = result['binary_disagreements']/result['complete_active_paths'] if result['complete_active_paths'] else None
    return result


def audit_eight_hour_labels(context, label):
    """Independent forward scan, without using the first-passage constructor."""
    rho=context.rho.to_numpy(float)
    valid=context.theory_available.to_numpy(bool) & np.isfinite(rho)
    positions=context.index.get_indexer(label.index)
    counts={'origins_checked':len(label),'events':0,'survivors':0,'early_censored':0,'unsupported':0}
    for row,i in zip(label.itertuples(),positions):
        expected=np.nan;publication=pd.NaT
        if i<0:raise ValueError('Origin not in explicit hourly clock')
        if not valid[i] or rho[i]==0:
            counts['unsupported']+=1
        else:
            for k in range(1,9):
                if i+k>=len(rho) or not valid[i+k]:
                    counts['early_censored']+=1
                    assert row.followup_hours==k-1 and not row.event_observed
                    discovered=context.index[min(i+k,len(rho)-1)]
                    assert row.record_available_at_utc==discovered
                    break
                cross=rho[i+k]<=0 if rho[i]>0 else rho[i+k]>=0
                if cross:
                    expected=1.;publication=context.index[i+k];counts['events']+=1
                    assert row.event_time_utc==publication and row.followup_hours==k
                    break
            else:
                expected=0.;publication=context.index[i+8];counts['survivors']+=1
        assert (np.isnan(expected) and pd.isna(row.event_by_8h)) or expected==row.event_by_8h
        assert (pd.isna(publication) and pd.isna(row.available_8h_utc)) or publication==row.available_8h_utc
    return counts


def joint_calendar_summary(parts, assets):
    """Require the complete named asset scope; use shared occupied-block counts."""
    records=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['symbol','origin_time_utc','gain'])
    missing=[s for s in assets if not records.symbol.eq(s).any()]
    invalid=records.gain.isna().any() or not np.isfinite(records.gain.to_numpy(float)).all()
    if missing or invalid:
        return {'required_assets':list(assets),'macro_gain':None,'ci_low':None,'ci_high':None,
                'interval_status':'unavailable_required_asset_or_censor_support','missing_assets':missing}
    records=records.loc[records.symbol.isin(assets)].copy()
    if records.duplicated(['symbol','origin_time_utc']).any():
        raise ValueError('Duplicate asset/origin in the comparison')
    interval=block_interval(records,2022)
    bins=((pd.DatetimeIndex(records.origin_time_utc)-EVAL).total_seconds()//(14*86400)).astype(int)
    occupied=sorted(set(bins));lookup={b:j for j,b in enumerate(occupied)}
    draws=np.random.default_rng(20261102).multinomial(len(occupied),np.full(len(occupied),1/len(occupied)),size=1000)
    asset_draws=[];valid_draw=np.ones(1000,bool);blocks_by_asset={}
    for symbol in assets:
        m=records.symbol.eq(symbol).to_numpy();ix=np.array([lookup[b] for b in bins[m]])
        n=np.bincount(ix,minlength=len(occupied));s=np.bincount(ix,weights=records.gain.to_numpy()[m],minlength=len(occupied))
        blocks_by_asset[symbol]=int(np.count_nonzero(n))
        den=draws@n;valid_draw &= den>0
        asset_draws.append(np.divide(draws@s,den,out=np.full(1000,np.nan),where=den>0))
    interval['valid_joint_draw_fraction']=float(valid_draw.mean())
    if np.isfinite(interval['ci_low']):
        ci=np.quantile(np.mean(asset_draws,axis=0)[valid_draw],[.025,.975])
        np.testing.assert_allclose(ci,[interval['ci_low'],interval['ci_high']],atol=1e-14,rtol=1e-14)
    interval.update(blocks_by_asset=blocks_by_asset,required_assets=list(assets))
    interval['scope']='two_asset_development_description_not_four_asset_final_estimand' if tuple(assets)==BTC_ETH else 'four_asset_2022_development_not_final_three_year_estimand'
    return interval


def run(src, rates):
    split = src.json(f'{B36}/split_manifest.json')['splits']
    assert split == {'fit2020':['2020-01-01','2021-01-01'],'selection2021':['2021-01-01','2021-05-01'],
        'fitmay2021':['2020-01-01','2021-05-01'],'calibration2021':['2021-05-01','2021-09-01'],
        'development2021':['2021-09-01','2022-01-01'],'development2022':['2022-01-01','2023-01-01']}
    selection = [src.json(f'{b}/selection.json') for b in (B36,B37)]
    assert selection[0]['2022_observations_used_for_selection'] is False
    assert selection[0]['trading_PnL_used_for_selection'] is False
    assert selection[1]['2022_outcomes_used'] is False and selection[1]['financial_PnL_used'] is False
    for rate in rates:
        assert all(selection[0]['choices'][rate+'|minimal'][k]==v for k,v in [('depth',3),('learning_rate',.1),('rounds',80)])
        assert selection[1]['choices'][rate+'|minimal']['alpha']==.001
    registered = src.json(f'{B36}/input_receipt.json')
    expected = {x['path']:x['sha256'] for x in registered['artifacts']}
    old_scores = pd.read_csv(src.path(f'{B37}/statistical_results.csv'))
    old_gains = pd.read_csv(src.path(f'{B37}/attribution_differences.csv'))
    old_intervals = pd.read_csv(src.path(f'{B37}/statistical_calendar_uncertainty.csv'))
    comparisons={'logistic_minus_tree':('matched_logistic','tree'),
                 'KM_minus_tree':('KM','tree'),'KM_minus_logistic':('KM','matched_logistic')}
    cases, gain_records, diagnostics = [], {r:{k:[] for k in comparisons} for r in rates}, []
    for symbol in ASSETS:
        for rate in rates:
            stem = f'{symbol}_{rate}'
            f = src.frame(f'{B36}/causal_inputs_v1/{stem}.parquet')
            assert f.symbol.eq(symbol).all() and f.financing.eq(rate).all()
            assert f.index.equals(pd.date_range(f.index.min(),f.index.max(),freq='h',name=f.index.name))
            rebuilt = features(f)
            pd.testing.assert_frame_equal(rebuilt[list(FEATURES['minimal'])],f[list(FEATURES['minimal'])])
            good = f.theory_available & f.rho.notna()
            assert f.loc[good,'rate_available_at_utc'].notna().all()
            assert f.loc[good,'rate_available_at_utc'].le(f.index[good]).all()
            known_funding=f.funding_known.eq(True)
            assert f.loc[known_funding,'last_funding_available_at_utc'].le(f.index[known_funding]).all()
            age=(f.index[known_funding]-pd.DatetimeIndex(f.loc[known_funding,'last_funding_settlement_utc'])).total_seconds()/3600
            assert ((age>=0)&(age<=12)).all()
            np.testing.assert_allclose(theory_signal(f.loc[good,'spot_close'],f.loc[good,'perp_close'],f.loc[good,'annual_rate'])[0],f.loc[good,'rho'],atol=1e-12,rtol=1e-12)
            tree, linear, own, censor, training = check_fit(src,symbol,rate,f)
            current, lab = origin_segment(f,EVAL,END)
            independent_labels = audit_eight_hour_labels(f.loc[f.index<END],lab)
            stored_lab = src.frame(f'{B36}/labels_v1/{stem}_development2022.parquet',begin=EVAL)
            pd.testing.assert_frame_equal(lab,stored_lab)
            x = current[list(FEATURES['minimal'])].to_numpy()
            s,q = trees.survival(tree,x,80)
            p,ql = logistic.predict(linear,x)
            validity = {'tree':probability_audit(q,s,1-s),
                        'matched_logistic':probability_audit(ql,np.cumprod(1-ql,axis=1),p)}
            allowed = current.minimal_eligible36.eq(True).to_numpy()
            issued = pd.DataFrame({'tree':np.where(allowed,1-s[:,7],np.nan),
                'matched_logistic':np.where(allowed,p[:,7],np.nan),
                'KM':np.where(allowed,1-km_predict(own,[8])[0],np.nan)},index=current.index)
            archived = src.frame(f'{B37}/forecasts_v1/development2022/{stem}_minimal.parquet',begin=EVAL,
                columns=['tree__8h','matched_hourly__8h','asset_km__8h'])
            pd.testing.assert_index_equal(issued.index,archived.index)
            forecast_rel = f'{B37}/forecasts_v1/development2022/{stem}_minimal.parquet'
            forecast_receipt = src.json(forecast_rel+'.receipt.json')
            assert src.hashes[forecast_rel]==forecast_receipt['sha256']
            max_error = {}
            for name,col in [('tree','tree__8h'),('matched_logistic','matched_hourly__8h'),('KM','asset_km__8h')]:
                np.testing.assert_array_equal(issued[name].notna(),archived[col].notna())
                np.testing.assert_allclose(issued[name],archived[col],equal_nan=True,atol=1e-13,rtol=1e-13)
                max_error[name] = float((issued[name]-archived[col]).abs().max())
            qualified = current.low_fee_margin.gt(0)
            pred = issued.loc[qualified,['tree','matched_logistic']]
            score,loss,mask,known,w = matched_ipcw_scores(pred,lab.loc[qualified],censor)
            common_score,common_loss,common_mask,_,_ = matched_ipcw_scores(issued.loc[qualified],lab.loc[qualified],censor)
            gain = loss.matched_logistic-loss.tree
            if gain.notna().all() and len(gain):
                gain_records[rate]['logistic_minus_tree'].append(pd.DataFrame({'origin_time_utc':gain.index,'symbol':symbol,'gain':gain.to_numpy()}))
            for comparison,(reference,model) in comparisons.items():
                if comparison=='logistic_minus_tree':continue
                delta=common_loss[reference]-common_loss[model]
                if delta.notna().all() and len(delta):
                    gain_records[rate][comparison].append(pd.DataFrame({'origin_time_utc':delta.index,'symbol':symbol,'gain':delta.to_numpy()}))
            refs = old_scores.loc[(old_scores.phase=='development2022')&(old_scores.symbol==symbol)&(old_scores.financing==rate)&(old_scores.cohort=='minimal')&(old_scores.panel=='attribution')&(old_scores.population=='paper_low:qualified')&(old_scores.horizon_hours==8)]
            for model,oldname in [('tree','tree'),('matched_logistic','matched_hourly')]:
                reference = refs.loc[refs.model==oldname].iloc[0]
                assert reference.forecast_origins==len(loss)
                assert np.isclose(reference.ipcw_brier,score[model]['ipcw_brier'],atol=1e-13,rtol=1e-13)
            refgain = old_gains.loc[(old_gains.phase=='development2022')&(old_gains.symbol==symbol)&(old_gains.financing==rate)&(old_gains.cohort=='minimal')&(old_gains.population=='paper_low:qualified')&(old_gains.effect=='effect_C_nonlinear')].iloc[0]
            assert np.isclose(gain.mean(),refgain.ipcw_brier_gain,atol=1e-13,rtol=1e-13)
            selected = lab.loc[loss.index]
            events = selected.event_observed & selected.followup_hours.le(8)
            alive = selected.followup_hours.ge(8) & ~events
            blocks = ((loss.index-EVAL).total_seconds()//(14*86400)).astype(int)
            cases.append({'symbol':symbol,'financing':rate,'full_evaluation_clock':len(current),
                'fit_status':{'tree':tree['status'],'matched_logistic':linear['status'],'KM':own['status']},
                'independent_eight_hour_label_audit':independent_labels,
                'feature_ready_origins':int(allowed.sum()),'low_fee_candidate_origins':int(qualified.sum()),
                'finite_issued_predictions':{m:int(issued[m].notna().sum()) for m in issued},
                'scoring_origins':len(loss),'events_by_8h':int(events.sum()),'survivors_to_8h':int(alive.sum()),
                'early_censored_origins':int((~(events|alive)).sum()),
                'all_followup_censored_origins':int((~selected.event_observed).sum()),
                'occupied_calendar_blocks':len(set(blocks)),
                'scores':score,'KM_common_panel':{'name':'tree_logistic_KM_common','scoring_origins':len(common_loss),
                    'pair_origins_lost':len(loss)-len(common_loss),'scores':common_score},
                'pair_scoring_keys_sha256':hashlib.sha256(loss.index.as_unit('ns').asi8.tobytes()).hexdigest(),
                'KM_common_scoring_keys_sha256':hashlib.sha256(common_loss.index.as_unit('ns').asi8.tobytes()).hexdigest(),
                'logistic_minus_tree_gain':float(gain.mean()),'training':training,
                'probability_validity':validity,'archived_prediction_max_absolute_error':max_error,
                'historical_score_and_gain_parity':True,'censor_G8':float(km_predict(censor,[8])[0]),
                'censor_maximum_followup':censor['maximum_followup'],
                'IPCW_nonzero_weight_min':float(w[w>0].min()) if (w>0).any() else None,
                'IPCW_nonzero_weight_max':float(w.max()) if len(w) else None,'weight_truncation_applied':False})
            if rate=='zero_rate' and symbol in BTC_ETH:diagnostics.append(minute_discretization(src,symbol,f))
            print('Verified',symbol,rate,len(loss),'Low-Fee scoring origins',flush=True)
    for rel,h in src.hashes.items():
        if rel in expected:assert h==expected[rel],rel
    summaries={}
    for scope,assets in [('BTC_ETH',BTC_ETH),('four_asset',ASSETS)]:
        summaries[scope]={}
        for rate,parts in gain_records.items():
            cells=[c for c in cases if c['financing']==rate and c['symbol'] in assets]
            supported=len(cells)==len(assets) and all(c['KM_common_panel']['scoring_origins']>0 and all(np.isfinite(c['KM_common_panel']['scores'][m]['ipcw_brier']) for m in ['KM','matched_logistic','tree']) for c in cells)
            means={m:float(np.mean([c['KM_common_panel']['scores'][m]['ipcw_brier'] for c in cells])) if supported else None for m in ['KM','matched_logistic','tree']}
            summaries[scope][rate]={'equal_asset_Brier':means,'required_assets':list(assets),
                'common_panel_status':'supported' if supported else 'unavailable_required_cell',
                'comparisons':{name:joint_calendar_summary([p for p in rows if p.symbol.iloc[0] in assets],assets) for name,rows in parts.items()}}
            if scope=='four_asset':
                u=summaries[scope][rate]['comparisons']['logistic_minus_tree']
                if u['macro_gain'] is not None:
                    ref=old_intervals.loc[(old_intervals.phase=='development2022')&(old_intervals.financing==rate)&(old_intervals.cohort=='minimal')&(old_intervals.population=='paper_low:qualified')&(old_intervals.effect=='effect_C_nonlinear')].iloc[0]
                    for key in ['macro_gain','observation_weighted_gain','ci_low','ci_high','leave_one_block_out_min','leave_one_block_out_max']:
                        np.testing.assert_allclose(u[key],ref[key],atol=1e-13,rtol=1e-13)
                    assert u['paired_origins']==ref.paired_origins
    prior_check={'status':'prior_summary_not_present'}
    prior_path=OUTPUT/'baseline.json'
    if prior_path.exists():
        prior=json.loads(prior_path.read_text())
        assert prior['version']=='btc-eth-development-checkpoint-v1' and prior['assets']==list(BTC_ETH)
        for old in prior['cases']:
            new=next(c for c in cases if c['symbol']==old['symbol'] and c['financing']==old['financing'])
            assert clean({k:new[k] for k in old})==old
        for rate in rates:
            new=summaries['BTC_ETH'][rate]['comparisons']['logistic_minus_tree']
            for key,val in prior['joint_calendar_uncertainty'][rate].items():
                assert clean(new[key])==val,(rate,key)
        assert diagnostics==prior['minute_hour_diagnostic']
        prior_check={'status':'all_prior_cases_primary_intervals_and_minute_diagnostic_unchanged',
                     'path':str(prior_path.relative_to(ROOT)),'sha256':hashlib.sha256(prior_path.read_bytes()).hexdigest()}
    return {'version':'four-asset-development-checkpoint-v2','checkpoint_date':'2026-11-09',
        'chronology':split,'assets':ASSETS,'rates':rates,'horizon_hours':8,
        'forecast_method':'recomputed_with_maintained_predictors_from_archived_May_2021_fits_no_refitting',
        'cases':cases,'scope_summaries':summaries,'minute_hour_diagnostic':diagnostics,
        'prior_BTC_ETH_preservation':prior_check,
        'source_sha256':dict(sorted(src.hashes.items())),
        'holdout_contents_accessed':False,'new_fits_or_calibration':False,'financial_backtest_run':False,
        'reproduction_command':'.venv/bin/python -B -m scripts.checkpoint_survival_baseline',
        'limitations':['2022_previously_inspected_development','four_asset_single_year_not_registered_final_three_year_aggregate',
            'conditional_IPCW_assumptions','hourly_target_not_continuous_first_passage','one_week_measurement_sample']}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):return float(value) if np.isfinite(value) else None
    return value


def checkpoint_report(value):
    """Compact course report, regenerated from the audited scalar summary."""
    score_rows=[];uncertainty_rows=[];minute_rows=[];training_rows=[];mean_rows=[];reference_rows=[]
    fmt=lambda x: 'unavailable' if x is None or not np.isfinite(x) else f'{x:.9f}'
    for c in value['cases']:
        scores=c['scores'];km=c['KM_common_panel']['scores']['KM']['ipcw_brier']
        score_rows.append(f"| {c['symbol']} | {c['financing']} | {c['scoring_origins']} | {c['events_by_8h']}/{c['survivors_to_8h']}/{c['early_censored_origins']} | {fmt(km)} | {fmt(scores['matched_logistic']['ipcw_brier'])} | {fmt(scores['tree']['ipcw_brier'])} | {fmt(c['logistic_minus_tree_gain'])} |")
        t=c['training']
        training_rows.append(f"| {c['symbol']} | {c['financing']} | {t['unique_training_origins']} | {t['training_trials']} | {t['training_event_trials']} | {t['KM_training_origins']} | {t['censor_training_origins']} |")
    for scope,by_rate in value['scope_summaries'].items():
        for rate,s in by_rate.items():
            m=s['equal_asset_Brier'];u=s['comparisons']['logistic_minus_tree']
            mean_rows.append(f"| {scope} | {rate} | {fmt(m['KM'])} | {fmt(m['matched_logistic'])} | {fmt(m['tree'])} | {s['common_panel_status']} |")
            uncertainty_rows.append(f"| {scope} | {rate} | {u.get('paired_origins','unavailable')} | {fmt(u['macro_gain'])} | [{fmt(u['ci_low'])}, {fmt(u['ci_high'])}] | {u.get('occupied_calendar_blocks','unavailable')} | [{fmt(u.get('leave_one_block_out_min'))}, {fmt(u.get('leave_one_block_out_max'))}] |")
            a,b=s['comparisons']['KM_minus_tree'],s['comparisons']['KM_minus_logistic']
            reference_rows.append(f"| {scope} | {rate} | {fmt(a['macro_gain'])} [{fmt(a['ci_low'])}, {fmt(a['ci_high'])}] | {fmt(b['macro_gain'])} [{fmt(b['ci_low'])}, {fmt(b['ci_high'])}] |")
    for d in value['minute_hour_diagnostic']:
        minute_rows.append(f"| {d['symbol']} | {d['complete_active_paths']}/{d['candidate_origins']} | {d['minute_events_by_8h']} | {d['hourly_events_by_8h']} | {d['binary_disagreements']} ({100*d['binary_disagreement_fraction']:.2f}%) | {d['first_crossing_time_disagreements']} | {d['low_fee_qualified_complete_paths']} |")
    return '''# November 9, 2026 — four-asset survival baseline checkpoint

**READY — four-asset extension v2, prepared November 5.** All eight asset/financing cases reproduce historical predictions and matched-model scores. BTC favors matched logistic; ETH favors tree over logistic but unconditional KM beats both. DOGE/ADA have larger tree gains over both references. Four-asset development contrasts are positive while the preserved BTC/ETH intervals span zero. These are previously inspected development findings, not confirmatory significance. No economic backtest or model selection ran.

## Response to Instructor Feedback — Scope

We adopted the single eight-hour forecasting focus and deferred trading-policy work. We retained BTC, ETH, DOGE and ADA because previously inspected development results exhibit qualitative cross-asset heterogeneity. This scope decision is informed by known development evidence, not blinded to outcomes; it is not a new independent confirmation. We report every asset, both financing controls, and both the original BTC/ETH and four-asset summaries, without post-hoc model or favorable asset selection. The frozen final protocol is unchanged.

## Question, population and chronology

Can an origin-fixed theoretical deviation cross its directional zero boundary within the next **eight hourly observations**? Primary population: active, nonzero-theory, six-feature-ready hourly BTCUSDT/ETHUSDT/DOGEUSDT/ADAUSDT origins strictly outside the original Low-Fee bound, regardless of position occupancy. Zero-rate financing is primary; seven-day-lagged EFFR is a separate secondary context. These are conditional paper-parameter, last-trade-price measurements.

Exact historical UTC half-open design: fit **[2020-01-01,2021-01-01)**; original model selection **[2021-01-01,2021-05-01)**; refit **[2020-01-01,2021-05-01)**; historical secondary calibration **[2021-05-01,2021-09-01)**; late development evaluation **[2021-09-01,2022-01-01)**; reproduced development evaluation **[2022-01-01,2023-01-01)**. Context clocks start January 8, 2020 for BTC/ETH, January 31 for ADA and July 10 for DOGE; prelaunch/invalid hours remain explicit. Fit/calibration origins require t+16h<segment end and used trial/label publication<cutoff. Warmup is separate from permitted origin bounds. Historical selection/late-2021 evaluation removed the final 16h; 2022 keeps year-end early-censored origins. The final study's 20/4 schedule is not used here.

Historical choices were selected across the original supported assets in early 2021 and are reused without reselection. We **recompute predictions from archived May-2021 fits**, using maintained code, and recompute labels/scores; no real-data coefficients, trees or calibration maps were newly fitted. Calibration is not applied to these raw models. **2022 was already inspected in Experiments 3.1–3.7 and is not an untouched independent holdout.** The frozen 2023–2025 design and sealed contents remain separate.

## Event time, censoring and information clock

Fix d=−sign(rho_origin). A short-perpetual direction first exits at future hourly rho<=0; a long direction at rho>=0. Zero touch counts. Strict fee entry and inclusive-zero exit differ. First future missing/inactive/theory-unavailable hour censors at the preceding consecutive valid hour, with gap discovery/publication at the gap, not earlier. Administrative censoring uses the prefix's last observation; no 2023 observation completes a 2022 label. Each observed hourly at-risk trial k<=16 has response 1 only at the first event, otherwise 0 while observed at risk; no post-censor negatives.

The deterministic tests derive these answers by hand (t is the origin). Event/trial cases assume an otherwise feature-ready origin; minimum history is tested separately.

| Example | Event / followup | Eight-hour outcome / availability | Trials |
| --- | --- | --- | --- |
| First directional crossing at hour 1 | event at 1 | 1, t+1h | k=1: response 1 |
| First crossing at hour 8 | event at 8 | 1, t+8h | k=1..7:0; k=8:1 |
| First crossing at hour 9 | event at 9 | 0 known at t+8h | k=1..8:0; k=9:1 |
| Positive rho then exact zero | event at zero touch | 1 at touch | stop at touch |
| Negative rho then zero/positive | opposite-direction event | 1 at crossing | same directional rule |
| Missing or inactive hour 2, later convergence | censored, followup 1 | unknown; gap known t+2h | k=1:0 only |
| No event in 32-hour prefix | censored, followup 31 | 0 known t+8h | observed k=1..16:0 |
| Crossing exactly at fit/calibration cutoff | excluded from prefix | insufficient followup stays unknown | strict 16h boundary purge applies |
| Insufficient past features | not feature-ready | no forecast | no eligible fit trials |
| Last completed minute at exact hour | bar [t−1min,t) | available at t | bar [t,t+1min) unavailable |
| Previous scheduled hour missing | rho-change1 missing | no forecast until restored context | no skipping the missing hour |

The original 26 synthetic checks are retained: event/censor examples, probabilities, funding availability, origin/publication bounds, future-mutation invariance, training-only transforms and rolling-KM publication. Three added checks cover DOGE/ADA source boundaries, four-way shared calendar draws with unequal asset samples, and unavailable/sparse support without dropping assets. An independent forward scan now checks **70,080 asset/control/evaluation-origin copies**, with exact archived full-prefix label parity. Financing copies are not independent observations.

## Eight-hour forecasting results

Lower Brier is better; gain=Logistic−Tree, positive favors tree. E/S/C denotes event by 8h / observed survivor through 8h / early-censored. Scores below are newly recomputed; tree/logistic scores and gains match archived Experiment 3.7 attribution values to 1e−13. Every issued KM/tree/logistic eight-hour prediction also matches its archived series to that tolerance. The **named tree/logistic/KM common scoring intersection loses zero origins** in every case; KM uses the same censor curve on that intersection.

| Asset | Financing | Scored origins | E/S/C | Unconditional KM | Matched hourly logistic | Custom tree | Logistic−Tree |
| --- | --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(score_rows)+'''

Every asset/control has **8,760 requested hours**. Each model issues 8,758 finite feature-ready forecasts for BTC/ETH and 8,756 for DOGE/ADA; missing inputs are not imputed. Qualified counts equal scored counts above. All fits and all eight primary/common scoring cells are supported; no replacement result is needed. Full-followup censored counts are BTC 0/0, ETH 10/5, DOGE 1/1 and ADA 1/0 (zero/EFFR). Only early censoring in E/S/C contributes zero IPCW loss; later censoring can supply a known eight-hour survivor. KM uses original two-feature fitting eligibility; primary models share six-feature risk sets. All three models use identical scoring keys and the same per-asset/control censor KM; primary-pair and three-model key hashes agree in every case.

| Asset | Financing | Risk origins | Hourly trials | Event trials | KM fit origins | Censor-KM fit origins |
| --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(training_rows)+'''

The tree remains custom NumPy histogram Newton boosting: depth3, eta .1, 80 rounds, 32 bins, minimum200 trials/leaf, leaf L2=5. Matched logistic uses the same six ordered covariates/trials, 16 interval intercept values, shared standardized slopes and alpha .001. It is not the older coarse logistic. Training matrices, responses, all tree cuts, fitted logistic moments and both KM curves were reconstructed and reconciled to the archived fits, without refitting real models.

Equal-asset means give every required asset weight 1/2 or 1/4, regardless of candidate density. The full-scope mean is unavailable if any required common scoring cell fails; no asset is dropped or replaced.

| Scope | Financing | Mean KM Brier | Mean logistic Brier | Mean tree Brier | Common-panel support |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(mean_rows)+'''


## Probability and IPCW checks

All finite hourly hazards and cumulative probabilities are in [0,1], S(h)=product(k<=h)(1−q_k) is nonincreasing, and p(h)=1−S(h) is nondecreasing. Constant hazard .2 gives p8=1−.8^8=0.83222784 for both maintained prediction interfaces. Entirely unavailable curves count as unavailable; partial/infinite/out-of-range or incoherent curves fail instead of being repaired. Every issued real model curve across all 16 hours passed these checks.

Eight-hour IPCW Brier is mean[E*(1−p)^2/G(T−1) + A*p^2/G(8)]. Early censoring contributes zero **while remaining in the denominator**. The fitting-only reverse KM preserves discrete ties: shift censored duration to followup+1, removing tied event rows from the censor risk set. A hand case with event@3, survivor@8 and censor@2; p=(.25,.2,.9); G(2)=.8, G(8)=.5 has weights (1.25,2,0) and Brier=(.75²/.8+.2²/.5)/3=0.2610416667. Independent tie and orientation examples pass.

Support requires fitted censor KM, horizon<=maximum fitted followup, G(8)>=.05 and every event G(T−1)>=.05. **This rejects unsupported cells; it does not replace denominators by .05. No additional weight truncation/Winsorization was applied.** G8 is approximately .98987–.99002 for BTC/ETH, .726115 for DOGE and .924367 for ADA; maximum nonzero weights are about 1.01023, 1.37719 and 1.08182 respectively. DOGE's shorter, gap-heavy early history makes censor-distribution stability especially relevant. Canonical scores and paired losses share the same support/weights; unsupported scores cannot become supported intervals via clipped tails.

## Dependence and shared-calendar uncertainty

Use January-1-anchored 14-day **occupied calendar blocks**, seed20261102 and 1,000 multinomial block-count draws. Within each named scope/panel, one count matrix is shared across all contributing assets; draw-specific asset means are then averaged equally. The four-asset run shares calendar draws across all four, never independent asset bootstraps. BTC/ETH retain their original 24-block-union calculation and values; the four-asset union has 27 blocks. Independent count-matrix replay checks every interval. Fixed KM contrasts use the same three-model keys and the same seeded draws within each scope/control because their block union is identical here.

Primary contrast is Logistic−Tree, positive favors tree.

| Scope | Financing | Paired origins | Mean gain | 2.5–97.5% block interval | Union blocks | Leave-one-block range |
| --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(uncertainty_rows)+'''

BTC/ETH/DOGE/ADA occupy 22/23/27/27 blocks under both controls (minimum8 per asset). Total origins exceed30 and all 1,000 joint draws have positive denominators (minimum80%). Four-asset primary gains, intervals and leave-one-block results match Experiment 3.7 within 1e−13. Previous BTC/ETH cases, primary intervals and minute diagnostics remain numerically identical. No unsupported asset was removed.

Both BTC/ETH primary intervals include zero. Four-asset intervals are positive in this inspected development sample, with larger DOGE/ADA gains and a negative BTC effect retained. This is cross-asset heterogeneity, **not a confirmatory significance claim or the final three-year result**. Intervals condition on fitted models/censor curves and omit fitting/selection uncertainty. Overlapping outcomes, risk trials and financing copies do not add independent calendar regimes.

Fixed reference contrasts are descriptive; positive KM−Tree favors tree and positive KM−Logistic favors logistic. No comparator was selected after scoring, and no "any favorable endpoint wins" rule is used.

| Scope | Financing | KM−Tree mean [block interval] | KM−Logistic mean [block interval] |
| --- | --- | --- | --- |
'''+ '\n'.join(reference_rows)+'''


## Minute versus hour measurement diagnostic

Raw inputs are completed one-minute bars; forecasting origins and primary first-passage checks are hourly; hazard k indexes future hourly intervals. Minute execution/accounting is outside this checkpoint. For positive initial rho: a minute-30 crossing persisting gives minute time30 versus hourly time60; a crossing reversing before the next hour can be entirely absent from hourly labels; an exact-hour zero touch agrees. An inactive/missing minute between valid hourly endpoints can censor a minute process while the hourly process continues. A missing/inactive hourly endpoint censors the primary process itself.

The unchanged prior **BTC/ETH-only** measurement sample uses 168 hourly origins **January 2–8, 2022** per asset, with observations through January 9 08:00 UTC. It remains a measurement limitation/control, with no DOGE/ADA minute-input access or expanded minute-level model study. January 2 avoids a prior partition's midnight endpoint; the window was chosen before inspecting the original diagnostic. January-2022 partitions are checksum-verified. Minute rho uses the exact maintained zero-financing formula and ordinary iota from completed minute closes; hourly values match source hourly rho. Full 481-point active valid paths are required; no minutes are filled.

| Asset | Complete/candidate origins | Minute events by8h | Hour events by8h | Binary disagreement | Different times among common events | Low-Fee complete origins |
| --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(minute_rows)+'''

There were zero invalid/zero origins or incomplete/inactive paths in this bounded sample. Disagreement percentages use **all complete valid origins**, not only Low-Fee-qualified candidates. Both binary and exact-time disagreements are reported; these are different quantities. A single overlapping week is not a representative whole-year frequency estimate. The hourly target is retained unchanged; no new target or alternative model was trained.

## Reproduction, preservation and submission

From the repository root in the existing locked environment:

```bash
.venv/bin/python -B -m unittest discover -s tests -p 'test_survival_checkpoint.py' -v
.venv/bin/python -B -m scripts.test_portable
.venv/bin/python -B -m scripts.checkpoint_survival_baseline --write-report
```

The command uses a pre-2023 source allowlist, checks periods before Parquet reads, rejects network access and forbids historical-input writes. It reuses local Experiment 3.6 features/fits/labels/risk sets and Experiment 3.7 fits/forecasts/statistics. The two original BTC/ETH January-2022 minute partitions remain the entire minute diagnostic scope. A public checkout needs these local sources restored; synthetic tests are portable. No final YAML loader/access authorization was activated. New scalar/provenance output is `results/checkpoints/2026-11-09/baseline_four_asset.json`; original `baseline.json` is preserved byte-for-byte. Prior report/verification copies are retained locally as `btc_eth_baseline_v1.md` and `btc_eth_verification_v1.json` in that ignored directory. No bulk predictions, caches or trade ledgers are published.

[Verification record](2026-11-09_verification.json) records commands, parity, source/code hashes and preservation. **29 focused tests and all 189 portable tests passed**, retaining all original 186 and adding three scope/support checks. Only the three previously documented archive-dependent preservation integrations are omitted; existing test methods/assertions remain unchanged. Frozen specifications, prior experiments and datasets remain unchanged. No 2023–2025 contents, downloads, final evaluation or market-data financial replay were used. Synthetic estimator/accounting tests remain tests, not market experiments.

For Canvas, submit the scope-response note, chronology, hand censoring examples, all eight score rows, both scope summaries and calendar intervals. Disclose prior 2022 inspection and the informed scope decision, censor-KM/selection assumptions, launch-aware fitting support, overlapping hours/common regimes, alternative financing, trade-price measurement and the short BTC/ETH minute diagnostic. Four-asset development gains do not establish executable value, uniform asset superiority or independent holdout performance. Final-data preparation, annual pipeline engineering and separately authorized holdout evaluation remain future work.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zero-rate-only',action='store_true',help='Omit the already available EFFR secondary control')
    parser.add_argument('--write-report',action='store_true',help='Regenerate the compact two-control course report')
    args=parser.parse_args()
    if args.write_report and args.zero_rate_only:parser.error('--write-report requires both declared controls')
    sys.dont_write_bytecode=True
    rates=RATES[:1] if args.zero_rate_only else RATES
    sources=Sources(rates)
    sys.addaudithook(sources.guard)
    value=clean(run(sources,rates))
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/('baseline_four_asset_zero_rate.json' if args.zero_rate_only else 'baseline_four_asset.json')
    content=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if not path.exists() or path.read_text()!=content:path.write_text(content)
    if args.write_report:
        report=ROOT/'docs/checkpoints/2026-11-09_baseline.md'
        report.parent.mkdir(parents=True,exist_ok=True)
        prose=checkpoint_report(value)
        if not report.exists() or report.read_text()!=prose:report.write_text(prose)
    print('Compact summary:',path.relative_to(ROOT),flush=True)


if __name__=='__main__':
    main()
