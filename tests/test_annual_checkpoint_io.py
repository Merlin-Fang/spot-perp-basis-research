"""Checkpoint parity and adversarial operational contracts on synthetic data."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.preflight_annual_forecasting import synthetic_contexts
from spot_perp_basis.training.annual import (
    ASSETS, MODELS, AnnualForecastSnapshot, fit_development_year, issue_forecasts,
    attach_outcomes, evaluate_forecasts, identity,
)
from spot_perp_basis.training.checkpoints import (
    CheckpointContractError, CheckpointUnsupportedError, checkpoint_bytes,
    checkpoint_provenance, implementation_binding, load_checkpoint,
    save_checkpoint, validate_contexts, validate_forecasts,
)


ROOT=Path(__file__).resolve().parents[1]


def canonical(p):return json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False)


def repair_identities(p):
    """Adversarial tests rehash state to exercise semantics beyond a checksum."""
    for a in p['assets'].values():
        f=a['fitted'];a['fitting_identity']=identity(f);a['calibration_identity']=identity(a['calibration_maps'])
        a['annual_reference_identity']=identity({'empirical':f['empirical'],'censor':f['scoring_censor_km']})
        a['normalization_and_quantiles_identity']=identity({'mean':f['matched_logistic'].get('mean'),
            'scale':f['matched_logistic'].get('scale'),'tree_cuts':f['tree'].get('cuts')})
    return AnnualForecastSnapshot(canonical(p))


class AnnualCheckpointIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contexts=synthetic_contexts()
        cls.snapshot=fit_development_year(cls.contexts,2022,'zero_rate')
        cls.provenance=checkpoint_provenance(cls.snapshot,cls.contexts,source_kind='synthetic',binding=implementation_binding(ROOT))
        cls.origins=pd.DatetimeIndex(['2022-01-01','2022-01-01T01:00','2022-04-01','2022-12-31T23:00'],tz='UTC')
        cls.forecasts=issue_forecasts(cls.snapshot,cls.contexts,origins=cls.origins)
        cls.attached=attach_outcomes(cls.snapshot,cls.forecasts,cls.contexts,cls.origins[-1])
        cls.content=checkpoint_bytes(cls.snapshot,cls.provenance)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'annual_2022_zero_rate.checkpoint.json'

    def saved(self):
        sha=save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        return load_checkpoint(self.path,expected_sha256=sha,expected_year=2022,
            expected_financing='zero_rate',expected_provenance=self.provenance)

    def bad_state(self,mutate):
        p=self.snapshot.read();mutate(p)
        explicit={s:{k:v for k,v in a.items() if k.endswith('_identity') and v!=self.snapshot.read()['assets'][s][k]} for s,a in p['assets'].items()}
        repaired=repair_identities(p).read()
        for s,values in explicit.items():repaired['assets'][s].update(values)
        return AnnualForecastSnapshot(canonical(repaired))

    def expect_bad(self,mutate):
        with self.assertRaises(CheckpointContractError):checkpoint_bytes(self.bad_state(mutate),self.provenance)

    def test_roundtrip_preserves_entire_snapshot_and_forecasts_exactly(self):
        loaded=self.saved()
        self.assertEqual(loaded.content,self.snapshot.content);self.assertEqual(loaded.sha256,self.snapshot.sha256)
        reproduced=issue_forecasts(loaded,self.contexts,origins=self.origins)
        pd.testing.assert_frame_equal(reproduced,self.forecasts,check_exact=True)
        self.assertEqual(validate_forecasts(loaded,reproduced,origins=self.origins)['rows'],144)

    def test_loaded_predictions_do_not_refit_or_recalibrate(self):
        loaded=self.saved()
        with (patch('spot_perp_basis.training.annual.fit_window',side_effect=AssertionError('refit')),
              patch('spot_perp_basis.training.annual.fit_calibrators',side_effect=AssertionError('recalibration'))):
            reproduced=issue_forecasts(loaded,self.contexts,origins=self.origins)
        pd.testing.assert_frame_equal(reproduced,self.forecasts,check_exact=True)

    def test_saved_bytes_are_deterministic_across_directories_and_idempotent(self):
        sha=save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        before=self.path.stat().st_mtime_ns
        again=save_checkpoint(self.saved(),self.path,provenance=self.provenance)
        self.assertEqual(sha,again);self.assertEqual(self.path.stat().st_mtime_ns,before)
        second=Path(self.temp.name)/'second'/self.path.name
        self.assertEqual(save_checkpoint(self.snapshot,second,provenance=self.provenance),sha)
        self.assertEqual(self.path.read_bytes(),second.read_bytes())
        self.assertEqual(list(self.path.parent.glob('.annual-*.tmp')),[])

    def test_existing_conflicting_file_is_never_overwritten(self):
        self.path.write_bytes(b'protected existing content');before=self.path.stat().st_mtime_ns
        with self.assertRaises(CheckpointContractError):save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        self.assertEqual(self.path.read_bytes(),b'protected existing content');self.assertEqual(self.path.stat().st_mtime_ns,before)
        self.assertEqual(list(self.path.parent.glob('.annual-*.tmp')),[])

    def test_atomic_publish_failure_leaves_no_checkpoint_or_temporary(self):
        with patch('spot_perp_basis.training.checkpoints.os.link',side_effect=OSError('injected publish failure')):
            with self.assertRaises(OSError):save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        self.assertFalse(self.path.exists());self.assertEqual(list(self.path.parent.glob('.annual-*.tmp')),[])

    def test_byte_corruption_and_truncation_reject_pinned_digest(self):
        sha=save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        for content in (self.content[:-20],self.content.replace(b'0.001',b'0.002',1)):
            with self.subTest(length=len(content)):
                self.path.write_bytes(content)
                with self.assertRaisesRegex(CheckpointContractError,'checksum'):
                    load_checkpoint(self.path,expected_sha256=sha,expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_duplicate_nonfinite_and_unknown_json_fail_even_with_new_file_hash(self):
        cases=[b'{"format":1,"format":2}',b'{"value":NaN}',b'{"value":1e999}',b'{}',b'not JSON']
        for content in cases:
            with self.subTest(content=content):
                self.path.write_bytes(content)
                with self.assertRaises(CheckpointContractError):load_checkpoint(self.path,expected_sha256=hashlib.sha256(content).hexdigest(),expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_unexpected_provenance_and_snapshot_identity_fail(self):
        sha=save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        wrong=copy.deepcopy(self.provenance);wrong['inputs']['BTCUSDT']['sha256']='0'*64
        with self.assertRaisesRegex(CheckpointContractError,'provenance'):
            load_checkpoint(self.path,expected_sha256=sha,expected_year=2022,expected_financing='zero_rate',expected_provenance=wrong)
        envelope=json.loads(self.content);envelope['snapshot_sha256']='0'*64
        content=(canonical(envelope)+'\n').encode();self.path.write_bytes(content)
        with self.assertRaisesRegex(CheckpointContractError,'Snapshot checksum'):
            load_checkpoint(self.path,expected_sha256=hashlib.sha256(content).hexdigest(),expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_final_year_and_wrong_file_identity_rejected_before_read(self):
        with patch('spot_perp_basis.training.checkpoints._read',side_effect=AssertionError('unexpected IO')):
            for year in (2023,2024,2025):
                with self.assertRaises(CheckpointContractError):load_checkpoint(self.path,expected_sha256='0'*64,expected_year=year,expected_financing='zero_rate',expected_provenance=self.provenance)
            with self.assertRaises(CheckpointContractError):load_checkpoint(self.path,expected_sha256='0'*64,expected_year=2022,expected_financing='nyfed_effr_lag7d',expected_provenance=self.provenance)
            with self.assertRaises(CheckpointContractError):load_checkpoint(self.path,expected_sha256='invalid',expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_symlink_and_missing_checkpoint_fail_without_fallback(self):
        target=Path(self.temp.name)/'target';target.write_bytes(b'protected');self.path.symlink_to(target)
        with self.assertRaises(CheckpointContractError):save_checkpoint(self.snapshot,self.path,provenance=self.provenance)
        self.assertEqual(target.read_bytes(),b'protected');self.path.unlink()
        with self.assertRaises(FileNotFoundError):load_checkpoint(self.path,expected_sha256='0'*64,expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_environment_and_payload_control_mismatch_fail(self):
        changed=copy.deepcopy(self.provenance);changed['binding']['environment']['numpy']='different_runtime'
        with self.assertRaisesRegex(CheckpointContractError,'environment'):
            save_checkpoint(self.snapshot,self.path,provenance=changed)
        envelope=json.loads(self.content);envelope['snapshot']['financing']='nyfed_effr_lag7d'
        envelope['snapshot_sha256']=hashlib.sha256(canonical(envelope['snapshot']).encode()).hexdigest()
        content=(canonical(envelope)+'\n').encode();self.path.write_bytes(content)
        with self.assertRaisesRegex(CheckpointContractError,'identity differs'):
            load_checkpoint(self.path,expected_sha256=hashlib.sha256(content).hexdigest(),expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_reformatted_and_oversized_checkpoint_fail(self):
        content=json.dumps(json.loads(self.content),indent=2).encode();self.path.write_bytes(content)
        with self.assertRaisesRegex(CheckpointContractError,'Noncanonical'):
            load_checkpoint(self.path,expected_sha256=hashlib.sha256(content).hexdigest(),expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)
        with self.path.open('wb') as f:f.truncate(16*1024*1024+1)
        with self.assertRaisesRegex(CheckpointContractError,'bounded regular'):
            load_checkpoint(self.path,expected_sha256='0'*64,expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_payload_registry_schema_and_calendar_are_strict(self):
        changes=[lambda p:p.update(protocol='other'),lambda p:p.update(schema='v999'),
            lambda p:p.update(final_execution_enabled=True),lambda p:p.update(extra_model=True),
            lambda p:p['assets'].pop('ADAUSDT'),lambda p:p.update(feature_order=list(reversed(p['feature_order']))),
            lambda p:p['windows']['fit'].__setitem__(0,'2020-02-01T00:00:00+00:00'),
            lambda p:p.update(information_cutoff_exclusive='2022-01-01')]
        for change in changes:
            with self.subTest(change=change):self.expect_bad(change)
        envelope=json.loads(self.content);envelope['model_registry'].pop()
        content=(canonical(envelope)+'\n').encode();self.path.write_bytes(content)
        with self.assertRaisesRegex(CheckpointContractError,'registry'):
            load_checkpoint(self.path,expected_sha256=hashlib.sha256(content).hexdigest(),expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)

    def test_incomplete_tree_and_changed_parameters_fail_after_rehash(self):
        for change in [lambda p:p['assets']['BTCUSDT']['fitted']['tree']['trees'].pop(),
            lambda p:p['assets']['BTCUSDT']['fitted']['tree'].update(learning_rate=.2),
            lambda p:p['assets']['BTCUSDT']['fitted']['tree']['cuts'].pop(),
            lambda p:p['assets']['BTCUSDT']['fitted']['manifest']['parameters'].update(rounds=81)]:
            with self.subTest(change=change):self.expect_bad(change)

    def test_missing_or_wrong_logistic_preprocessing_fails(self):
        for change in [lambda p:p['assets']['BTCUSDT']['fitted']['matched_logistic']['scale'].__setitem__(0,0),
            lambda p:p['assets']['BTCUSDT']['fitted']['matched_logistic']['beta'].pop(),
            lambda p:p['assets']['BTCUSDT']['fitted']['matched_logistic'].update(alpha=.01),
            lambda p:p['assets']['BTCUSDT']['fitted']['empirical']['hazard_0.01'].update(features=['rho','abs_rho','future_label'])]:
            with self.subTest(change=change):self.expect_bad(change)

    def test_KM_roles_curves_and_reference_hashes_are_checked(self):
        for change in [lambda p:p['assets']['BTCUSDT']['fitted']['scoring_censor_km'].update(reverse_censor=False),
            lambda p:p['assets']['BTCUSDT']['fitted']['empirical']['asset_km']['survival'].__setitem__(1,1.1),
            lambda p:p['assets']['BTCUSDT']['fitted']['empirical']['magnitude_curves'].pop(),
            lambda p:p['assets']['BTCUSDT'].update(annual_reference_identity='0'*64),
            lambda p:p['assets']['BTCUSDT'].update(annual_KM8=.123)]:
            with self.subTest(change=change):self.expect_bad(change)

    def test_calibration_identity_publication_and_monotonicity_checked(self):
        for change in [lambda p:p['assets']['BTCUSDT']['calibration_maps']['calibrated_tree'].update(slope=-1),
            lambda p:p['assets']['BTCUSDT']['calibration_maps']['calibrated_tree'].update(maximum_label_publication_utc='2022-01-01T00:00:00+00:00'),
            lambda p:p['assets']['BTCUSDT']['fitted']['manifest'].update(maximum_trial_publication_utc='2021-09-01T00:00:00+00:00'),
            lambda p:p['assets']['BTCUSDT']['calibration_maps']['calibrated_tree'].update(calibration_source='different_pool')]:
            with self.subTest(change=change):self.expect_bad(change)

    def test_unsupported_checkpoint_has_explicit_diagnostics_and_no_saved_fallback(self):
        def change(p):
            model=p['assets']['ADAUSDT']['fitted']['matched_logistic']
            model['status']='unsupported_numerical_nonconvergence'
        s=self.bad_state(change)
        with self.assertRaises(CheckpointUnsupportedError) as failure:save_checkpoint(s,self.path,provenance=self.provenance)
        self.assertTrue(any('ADAUSDT:raw_matched_logistic' in f for f in failure.exception.failures));self.assertFalse(self.path.exists())
        envelope=json.loads(self.content);envelope['snapshot']=s.read();envelope['snapshot_sha256']=s.sha256
        content=(canonical(envelope)+'\n').encode();self.path.write_bytes(content)
        with self.assertRaises(CheckpointUnsupportedError):load_checkpoint(self.path,expected_sha256=hashlib.sha256(content).hexdigest(),expected_year=2022,expected_financing='zero_rate',expected_provenance=self.provenance)
        forecasts=issue_forecasts(s,self.contexts,origins=self.origins)
        self.assertEqual(validate_forecasts(s,forecasts,origins=self.origins)['rows'],144)
        self.assertTrue(forecasts.loc[forecasts.symbol.eq('ADAUSDT')&forecasts.model.eq('raw_matched_logistic'),'event_probability'].isna().all())

    def test_input_schema_missing_extra_and_future_outcome_columns_fail(self):
        for column in ('spot_close','rate_available_at_utc'):
            c={**self.contexts,'BTCUSDT':self.contexts['BTCUSDT'].drop(columns=column)}
            with self.assertRaises(CheckpointContractError):validate_contexts(self.snapshot,c)
        c={**self.contexts,'BTCUSDT':self.contexts['BTCUSDT'].assign(future_event=1)}
        with self.assertRaises(CheckpointContractError):validate_contexts(self.snapshot,c)

    def test_input_asset_control_clock_and_types_fail(self):
        with self.assertRaises(CheckpointContractError):validate_contexts(self.snapshot,{s:f for s,f in self.contexts.items() if s!='ADAUSDT'})
        edits=[lambda f:f.assign(symbol='ADAUSDT'),lambda f:f.assign(financing='nyfed_effr_lag7d'),
            lambda f:f.drop(f.index[10]),lambda f:f.assign(theory_available='True'),
            lambda f:f.assign(spot_close='100')]
        for edit in edits:
            with self.subTest(edit=edit):
                with self.assertRaises(ValueError):validate_contexts(self.snapshot,{**self.contexts,'BTCUSDT':edit(self.contexts['BTCUSDT'].copy())})

    def test_post2022_context_is_rejected_without_consuming_values(self):
        f=self.contexts['BTCUSDT'].iloc[:1].copy();f.index=pd.DatetimeIndex(['2023-01-01'],tz='UTC')
        with self.assertRaises(ValueError):validate_contexts(self.snapshot,{**self.contexts,'BTCUSDT':f})

    def test_input_publication_exact_boundary_and_partial_fields(self):
        f=self.contexts['BTCUSDT'].copy();t=f.index[-1]
        f['bar_available_at_utc']=f.bar_available_at_utc.dt.as_unit('ns');f.loc[t,'bar_available_at_utc']=t+pd.Timedelta(nanoseconds=1)
        with self.assertRaises(ValueError):validate_contexts(self.snapshot,{**self.contexts,'BTCUSDT':f})
        f=self.contexts['BTCUSDT'].drop(columns='bar_open_time_utc')
        with self.assertRaises(CheckpointContractError):validate_contexts(self.snapshot,{**self.contexts,'BTCUSDT':f})

    def test_provenance_excludes_evaluation_and_binds_training_inputs(self):
        c={s:f.copy() for s,f in self.contexts.items()}
        for f in c.values():f.loc[f.index>=self.origins[0],'spot_close']*=1.01
        a=checkpoint_provenance(self.snapshot,c,source_kind='synthetic',binding=self.provenance['binding'])
        self.assertEqual(a,self.provenance)
        c['BTCUSDT'].loc[c['BTCUSDT'].index[100],'spot_close']*=1.01
        b=checkpoint_provenance(self.snapshot,c,source_kind='synthetic',binding=self.provenance['binding'])
        self.assertNotEqual(b,self.provenance)

    def test_output_complete_keys_and_schema_preserve_missing_rows(self):
        self.assertEqual(validate_forecasts(self.snapshot,self.forecasts,origins=self.origins)['rows'],144)
        bad=[self.forecasts.iloc[:-1],pd.concat([self.forecasts,self.forecasts.iloc[:1]]),
            self.forecasts.drop(columns='forecast_status'),self.forecasts.assign(future_selected=True),
            self.forecasts.loc[self.forecasts.symbol.ne('ADAUSDT')],self.forecasts.loc[self.forecasts.model.ne('unconditional_KM')]]
        for f in bad:
            with self.subTest(rows=len(f)):
                with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)

    def test_output_identity_probability_and_failure_statuses_fail(self):
        for column,value in [('symbol','BNBUSDT'),('model','new_model'),('financing','nyfed_effr_lag7d'),
            ('horizon_hours',24),('evaluation_year',2021),('snapshot_sha256','0'*64),
            ('event_probability',1.1),('event_probability',np.inf),('model_support_status','unknown'),
            ('forecast_status','made_up'),('forecast_available_at_utc',pd.Timestamp('2022-01-01T00:01Z'))]:
            f=self.forecasts.copy();f.loc[0,column]=value
            with self.subTest(column=column,value=value):
                with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)

    def test_nullable_missing_identity_and_declared_input_availability_fail(self):
        for name in ('financing','model','snapshot_sha256','calibration_source','forecast_status','horizon_hours'):
            f=self.forecasts.copy()
            if name=='horizon_hours':f[name]=f[name].astype('Int64')
            f.loc[0,name]=pd.NA
            with self.subTest(name=name):
                with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)
        f=self.contexts['BTCUSDT'].assign(symbol='BTCUSDT');f.loc[f.index[0],'symbol']=pd.NA
        with self.assertRaises(CheckpointContractError):validate_contexts(self.snapshot,{**self.contexts,'BTCUSDT':f})
        f=self.contexts['BTCUSDT'].copy();f.loc[f.index[0],'spot_close']=np.nan
        with self.assertRaises(CheckpointContractError):validate_contexts(self.snapshot,{**self.contexts,'BTCUSDT':f})

    def test_calibrated_output_cannot_silently_use_another_parent_or_map(self):
        f=self.forecasts.copy();ix=f.index[f.model.eq('calibrated_tree')][0];f.loc[ix,'event_probability']=.123
        with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)
        f=self.forecasts.copy();f.loc[ix,'calibration_source']='different_pool'
        with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)

    def test_rolling_publication_equality_and_naive_times_fail(self):
        f=self.forecasts.copy();ix=f.index[f.model.eq('rolling_magnitude_KM')][0]
        f.loc[ix,'rolling_latest_record_publication_utc']=f.loc[ix,'origin_time_utc']
        with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)
        f=self.forecasts.copy();f['origin_time_utc']=f.origin_time_utc.dt.tz_localize(None)
        with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins)

    def test_outcome_attachment_contract_and_fresh_pending_labels(self):
        self.assertEqual(validate_forecasts(self.snapshot,self.attached,origins=self.origins,attached=True,as_of=self.origins[-1])['rows'],144)
        early=attach_outcomes(self.snapshot,self.forecasts,self.contexts,self.origins[0])
        validate_forecasts(self.snapshot,early,origins=self.origins,attached=True,as_of=self.origins[0])
        self.assertTrue(early.loc[early.origin_time_utc.gt(self.origins[0]),'outcome_status'].eq('pending_observation').all())
        last=self.attached.loc[self.attached.origin_time_utc.eq(self.origins[-1])]
        self.assertTrue(last.followup_hours.eq(0).all())

    def test_future_or_inconsistent_attached_outcomes_fail(self):
        for column,value in [('label_available_at_utc',pd.Timestamp('2023-01-01T00:00Z')),
            ('event_by_8h',2.),('followup_hours',-1.),('outcome_status','known_8h'),('scoring_status','scored')]:
            f=self.attached.copy();f.loc[f.index[-1],column]=value
            with self.subTest(column=column):
                with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,f,origins=self.origins,attached=True,as_of=self.origins[-1])
        with self.assertRaises(CheckpointContractError):validate_forecasts(self.snapshot,self.attached,origins=self.origins)


if __name__=='__main__':unittest.main()
