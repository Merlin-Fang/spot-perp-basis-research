"""Portable final-input structural fixtures; no real market observations."""

from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pandas as pd

from spot_perp_basis.data import final_sources as s
from spot_perp_basis.data import final_inputs as b
from spot_perp_basis.data.funding import prepare_funding, join_funding
from spot_perp_basis.data.ingest import parse_klines, parse_funding_archive, checksum_value
from spot_perp_basis.data.windows import require_training_times


ROOT = Path(__file__).resolve().parents[1]


def archive(spec, *, unit="ms", duplicate=False, inactive=False, missing=False, bad=False):
    start = pd.Timestamp(spec.month+"-01", tz="UTC")
    scale = {"s": 10**9, "ms": 10**6, "us": 1000, "ns": 1}[unit]
    times = [start, start + pd.Timedelta(minutes=1), start + pd.offsets.MonthBegin() - pd.Timedelta(minutes=1)]
    if missing:
        times.pop(1)
    if duplicate:
        times.append(times[0])
    if spec.source == "funding":
        lines = ["calc_time,funding_interval_hours,last_funding_rate"]
        for i, t in enumerate(times):
            lines.append(f"{t.value//scale},{8 if i == 0 else 4},{-.001 if i == 0 else .002}")
    else:
        lines = [] if spec.source == "spot" else [",".join(["count" if c=="trades" else c for c in b.COLUMNS])]
        for i, t in enumerate(times):
            close = t.value//scale + (60*10**9)//scale - 1
            lines.append(f"{t.value//scale},10,12,8,{20 if bad else 11},{0 if inactive else 2},{close},{0 if inactive else 22},{0 if inactive else 3},0,0,0")
    stream = io.BytesIO()
    with ZipFile(stream, "w") as z:
        z.writestr(spec.filename.replace(".zip", ".csv"), "\n".join(lines)+"\n")
    return stream.getvalue()


def request_for(spec, body):
    def request(url):
        meta = {"url": url, "status": 200, "retrieved_at_utc": "synthetic_fixture"}
        return ((sha256(body).hexdigest()+"  "+spec.filename+"\n").encode() if url.endswith(".CHECKSUM") else body), meta
    return request


def put(root, spec, **kwargs):
    body = archive(spec, **kwargs)
    return s.acquire_one(root, spec, request=request_for(spec, body))


def effr_raw():
    return {"refRates": [{"type": "EFFR", "effectiveDate": t, "percentRate": 4.5}
                          for t in ["2022-12-23", "2023-12-29", "2024-12-31", "2025-12-31"]]}


class FinalDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = s.Source("DOGEUSDT", "2025-01", "spot")

    def test_official_paths_and_exact_required_scope(self):
        self.assertEqual(len(s.requirements()), 444)
        self.assertEqual(len(set(s.requirements())), 444)
        self.assertEqual(self.spec.url, "https://data.binance.vision/data/spot/monthly/klines/DOGEUSDT/1m/DOGEUSDT-1m-2025-01.zip")
        self.assertIn("futures/um/monthly/klines", s.Source("ADAUSDT", "2024-02", "usdm_perpetual").url)
        self.assertIn("futures/um/monthly/fundingRate", s.Source("ADAUSDT", "2024-02", "funding").url)

    def test_unauthorized_sources_fail_before_io(self):
        for args in [("BNBUSDT", "2025-01", "spot"), ("BTCUSDT", "2026-01", "spot"), ("BTCUSDT", "2025-01", "coinm")]:
            with self.assertRaises(ValueError): s.Source(*args)

    def test_checksum_validation_and_offline_reuse(self):
        put(self.root, self.spec)
        r = s.acquire_one(self.root, self.spec, offline=True, request=lambda u: self.fail("Offline requested network"))
        self.assertTrue(r["official_checksum_verified"])
        with self.assertRaises(ValueError): checksum_value(b"not a hash", self.spec.filename)
        with self.assertRaises(ValueError): checksum_value(("0"*64+"  wrong.zip").encode(), self.spec.filename)

    def test_corrupt_cache_fails_no_fallback(self):
        put(self.root, self.spec)
        (self.root/self.spec.relative).write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt"): s.acquire_one(self.root, self.spec, request=lambda u: self.fail("Unexpected refresh"))

    def test_failed_download_does_not_publish(self):
        good = archive(self.spec)
        req = request_for(self.spec, good)
        def broken(u):
            if u.endswith(".CHECKSUM"): return req(u)
            raise OSError("Interrupted transfer")
        with self.assertRaises(OSError): s.acquire_one(self.root, self.spec, request=broken)
        self.assertFalse((self.root/self.spec.relative).exists())
        self.assertFalse(list(self.root.rglob("*.partial")))

    def test_atomic_failure_cleanup_and_no_overwrite(self):
        p=self.root/"x"
        with patch.object(Path, "replace", side_effect=OSError("failure")):
            with self.assertRaises(OSError): s.atomic(p,b"a")
        self.assertFalse(list(self.root.rglob("*.partial")))
        s.atomic(p,b"a"); m=p.stat().st_mtime_ns; s.atomic(p,b"a")
        self.assertEqual(m,p.stat().st_mtime_ns)
        with self.assertRaises(ValueError): s.atomic(p,b"b")

    def test_duplicate_archives_reused_and_conflicts_quarantined(self):
        body = archive(self.spec); paths=[]
        for i in range(2):
            p=self.root/f"old{i}"/self.spec.source/self.spec.filename;p.parent.mkdir(parents=True);p.write_bytes(body);paths.append(p)
        found=s.candidates(self.root,[self.spec])[self.spec.key]
        self.assertEqual(set(found),set(paths))
        out=self.root/"new"
        r=s.acquire_one(out,self.spec,found,request=request_for(self.spec,body))
        self.assertEqual(r["acquisition"],"reused_verified");self.assertEqual(r["duplicate_copies"],2)
        paths[0].write_bytes(b"different vintage")
        with self.assertRaisesRegex(ValueError,"conflict"): s.acquire_one(self.root/"conflict",self.spec,paths,request=request_for(self.spec,body))
        self.assertEqual(paths[0].read_bytes(),b"different vintage")
        self.assertTrue(list((self.root/"conflict/conflicts").glob("*.json")))

    def test_missing_offline_source_fails_without_network(self):
        with self.assertRaises(FileNotFoundError): s.acquire_one(self.root,self.spec,offline=True,request=lambda u:self.fail("Network"))

    def parsed(self, unit="ms", month="2025-01", **kwargs):
        spec=s.Source("DOGEUSDT",month,"spot");r=put(self.root,spec,unit=unit,**kwargs)
        return parse_klines(self.root/spec.relative,spec.symbol,spec.month,spec.source,r["sha256"])

    def test_exact_utc_units(self):
        for unit in ("s","ms","us","ns"):
            with self.subTest(unit=unit), tempfile.TemporaryDirectory() as d:
                spec=s.Source("ADAUSDT","2024-02","spot");r=put(d,spec,unit=unit)
                f,a=parse_klines(Path(d)/spec.relative,spec.symbol,spec.month,spec.source,r["sha256"])
                self.assertEqual(a["timestamp_unit"],unit); self.assertEqual(f.index.unit,"ns")
                self.assertEqual(f.index[0],pd.Timestamp("2024-02-01",tz="UTC"))
                self.assertEqual(int(f.observed.sum()),3)

    def test_2024_2025_spot_transition(self):
        old,oa=self.parsed(month="2024-12",unit="ms")
        new,na=self.parsed(month="2025-01",unit="us")
        self.assertEqual((oa["timestamp_unit"],na["timestamp_unit"]),("ms","us"))
        self.assertEqual(old.index[-1]+b.MINUTE,new.index[0])
        self.assertEqual(na["bad_close_time"],0)

    def test_duplicate_minute_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):self.parsed(duplicate=True)

    def test_leap_month_and_missing_inactive_validity(self):
        f,a=self.parsed(month="2024-02",inactive=True,missing=True)
        self.assertEqual(len(f),29*1440);self.assertEqual(a["missing_minutes"],29*1440-2)
        self.assertTrue(f.valid.iloc[0]);self.assertEqual(f.volume.iloc[0],0)
        self.assertTrue(np.isnan(f.close.iloc[1]))

    def test_invalid_ohlc_remains_flagged_not_imputed(self):
        f,a=self.parsed(bad=True);self.assertEqual(a["bad_price"],3)
        self.assertEqual(f.close.iloc[0],20);self.assertFalse(f.valid.iloc[0])

    def test_pairing_completed_clock_previous_month_and_terminal(self):
        for month in ("2022-12","2024-02","2025-12"):
            for source in ("spot","usdm_perpetual"):put(self.root,s.Source("ADAUSDT",month,source))
            f,a,_,g=b.market_month(self.root,"ADAUSDT",month)
            self.assertTrue((pd.DatetimeIndex(f.bar_open_time_utc)+b.MINUTE).equals(f.index))
            self.assertTrue(f.symbol.eq("ADAUSDT").all())
            self.assertFalse(f.index.has_duplicates)
            if month=="2022-12":
                self.assertEqual(list(f.index),[s.START]);self.assertEqual(f.bar_open_time_utc.iloc[0],s.START-b.MINUTE)
            if month=="2025-12": self.assertTrue((f.index<s.END).all());self.assertNotIn(s.END,f.index)
            if month=="2024-02":self.assertIn(pd.Timestamp("2024-03-01",tz="UTC"),f.index)

    def test_no_cross_symbol_or_dropped_clock_keys(self):
        t=pd.date_range("2025-01-01",periods=2,freq="min",tz="UTC").as_unit("ns")
        f=pd.DataFrame({"symbol":"BTCUSDT","bar_open_time_utc":t-b.MINUTE},index=t.rename("decision_time_utc"))
        b.validate_clock(f,"BTCUSDT",t)
        with self.assertRaises(ValueError):b.validate_clock(f,"ADAUSDT",t)
        with self.assertRaises(ValueError):b.validate_clock(f.iloc[:1],"BTCUSDT",t)

    def test_funding_signed_rates_cadence_and_duplicate_rejection(self):
        spec=s.Source("ADAUSDT","2024-02","funding");r=put(self.root,spec)
        f,a=parse_funding_archive(self.root/spec.relative,spec.symbol,spec.month,r["sha256"])
        self.assertLess(f.funding_rate.iloc[0],0);self.assertEqual(a["interval_hours_values"],[4.0,8.0])
        self.assertTrue(prepare_funding(f).cadence_gap_flag.any())
        with self.assertRaisesRegex(ValueError,"Duplicate"):prepare_funding(pd.concat([f,f]))

    def test_duplicate_funding_archive_and_truncated_zip_fail(self):
        spec=s.Source("BTCUSDT","2025-01","funding")
        r=put(self.root,spec,duplicate=True)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            parse_funding_archive(self.root/spec.relative,spec.symbol,spec.month,r["sha256"])
        bad=b"truncated ZIP"
        r=s.acquire_one(self.root,self.spec,request=request_for(self.spec,bad))
        from zipfile import BadZipFile
        with self.assertRaises(BadZipFile):
            parse_klines(self.root/self.spec.relative,self.spec.symbol,self.spec.month,self.spec.source,r["sha256"])

    def test_funding_publication_delay_exact_match_and_staleness(self):
        t=pd.Timestamp("2025-01-01 00:00:00.005",tz="UTC")
        f=prepare_funding(pd.DataFrame({"settlement_time_utc":pd.DatetimeIndex([t]).as_unit("ns"),"funding_rate":[-.01],"funding_interval_hours":[4.]}))
        clock=pd.DatetimeIndex([t,t+b.MINUTE,t+pd.Timedelta(hours=12),t+pd.Timedelta(hours=12,seconds=1)]).as_unit("ns").rename("decision_time_utc")
        out=join_funding(pd.DataFrame(index=clock),f)
        self.assertEqual(out.funding_known.tolist(),[False,True,True,False]);self.assertEqual(out.last_settled_funding_rate.iloc[1],-.01)
        self.assertEqual(out.last_funding_settlement_utc.iloc[1],t)

    def test_effr_units_lag_backward_join_boundaries(self):
        rates=s.effr_table(effr_raw())
        self.assertTrue(rates.annual_rate.eq(.045).all())
        clock=pd.DatetimeIndex(["2023-01-01T00:00Z","2024-01-04T23:59Z","2024-01-05T00:00Z","2025-01-07T00:00Z"]).as_unit("ns")
        out=b.causal_effr(clock,rates)
        self.assertEqual(out.effr_observation_time_utc.iloc[1],pd.Timestamp("2022-12-23",tz="UTC"))
        self.assertEqual(out.effr_observation_time_utc.iloc[2],pd.Timestamp("2023-12-29",tz="UTC"))
        self.assertTrue((out.effr_available_at_utc<=clock).all())

    def test_effr_revisions_or_wrong_series_fail(self):
        raw=effr_raw();raw["refRates"].append(raw["refRates"][0].copy())
        with self.assertRaises(ValueError):s.effr_table(raw)
        raw=effr_raw();raw["refRates"][0]["type"]="SOFR"
        with self.assertRaises(ValueError):s.effr_table(raw)

    def test_structural_guard_blocks_outcomes_and_keeps_development_guard(self):
        from spot_perp_basis.models.calibration import sigmoid
        with b.structural_only():
            with self.assertRaises(PermissionError):sigmoid(np.array([0.]))
        with self.assertRaisesRegex(ValueError,"2020"):
            require_training_times(pd.date_range("2023-01-01",periods=2,tz="UTC"))
        self.assertFalse(json.loads((ROOT/"configs/final_data_v1.json").read_text())["evaluation_authorized"])

    def test_every_outcome_family_is_denied_by_structural_boundary(self):
        for module in b.FORBIDDEN:
            namespace={"__name__":module}
            exec("def forbidden_operation(): return 1",namespace)
            with self.subTest(module=module),b.structural_only():
                with self.assertRaises(PermissionError):namespace["forbidden_operation"]()

    def test_parquet_roundtrip_and_independent_source_check(self):
        for source in ("spot","usdm_perpetual"):put(self.root,s.Source("DOGEUSDT","2025-01",source),unit="us" if source=="spot" else "ms")
        p,a,_,_=b.market_month(self.root,"DOGEUSDT","2025-01")
        r=b.save_frame(self.root,"roundtrip.parquet",p)
        pd.testing.assert_frame_equal(p,pd.read_parquet(self.root/r["path"]),check_freq=False)
        m=(self.root/r["path"]).stat().st_mtime_ns;b.save_frame(self.root,"roundtrip.parquet",p)
        self.assertEqual(m,(self.root/r["path"]).stat().st_mtime_ns)
        self.assertEqual(b.independent_market_check(self.root,"DOGEUSDT","2025-01",p,a),6)
        with self.assertRaises(ValueError):b.safe_path(self.root,"../escape.parquet")

    def test_source_only_export_offline_cli_and_ingestion(self):
        checkout=self.root/"checkout";checkout.mkdir()
        body=subprocess.check_output(["git","archive","HEAD"],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:tar.extractall(checkout,filter="data")
        # Exact proposed publication overlay; no legacy untracked scripts/manifests.
        for name in ["src/spot_perp_basis/data/final_sources.py","src/spot_perp_basis/data/final_inputs.py","scripts/prepare_final_data.py","configs/final_data_v1.json"]:
            dest=checkout/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        data=self.root/"external_data"
        for spec in s.requirements():put(data,spec,unit="us" if spec.source=="spot" and spec.month>="2025-01" else "ms")
        raw=json.dumps(effr_raw()).encode()
        s.acquire_effr(data,request=lambda url:(raw,{"url":url,"status":200,"retrieved_at_utc":"synthetic_fixture"}))
        import os
        env={**os.environ,"PYTHONPATH":str(checkout/"src"),"PYTHONDONTWRITEBYTECODE":"1"}
        code="""from scripts.prepare_final_data import main
from spot_perp_basis.data.final_inputs import market_month, save_frame, independent_market_check, structural_only
import sys, socket
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('Network forbidden'))
assert main(['--data-root',sys.argv[1],'--offline','--verify'])==0
with structural_only():
 p,a,_,_=market_month(sys.argv[1],'DOGEUSDT','2025-01')
 save_frame(sys.argv[1],'clean_fixture.parquet',p)
 assert independent_market_check(sys.argv[1],'DOGEUSDT','2025-01',p,a)==6
print('SOURCE_ONLY_OFFLINE_FIXTURE_VERIFIED')
"""
        run=subprocess.run([sys.executable,"-B","-c",code,str(data)],cwd=checkout,env=env,capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertIn("SOURCE_ONLY_OFFLINE_FIXTURE_VERIFIED",run.stdout)


if __name__=="__main__":unittest.main()
