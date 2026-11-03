"""Run portable unit tests while denying content access to research archives.

Invoke from the repository root: python -B -m scripts.test_portable.

Three original byte-preservation integrations remain in the test files. They
need the original frozen source/artifact workspace, rather than this refactor.
Their assertions are retained and they are explicitly omitted from this suite.
"""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
ARCHIVAL_TESTS = {
    'test_collateral_scenario.CollateralScenarioTests.test_old_history_and_both_pre_experiment_freezes_preserved',
    'test_five_asset_data.FiveAssetTests.test_existing_sources_data_and_experiments_preserved',
    'test_target_audit.TargetAuditTests.test_original_history_unchanged',
}


def deny_archive_contents(event, arguments):
    if event != 'open' or not isinstance(arguments[0], (str, bytes)):
        return
    path = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]).resolve()
    if any(path.is_relative_to(ROOT/name) for name in ('data', 'results', 'experiments')):
        raise PermissionError(f'Portable tests cannot open protected research contents: {path}')


def flatten(suite):
    for case in suite:
        if isinstance(case, unittest.TestSuite):
            yield from flatten(case)
        else:
            yield case


def main():
    sys.addaudithook(deny_archive_contents)
    # Prevent cache writes even if the caller omitted python -B.
    sys.dont_write_bytecode = True
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    cases = []
    for case in flatten(suite):
        if case.id() in ARCHIVAL_TESTS:
            print('Not run (original source/artifact hash integration):', case.id(), flush=True)
        else:
            cases.append(case)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cases))
    return int(not result.wasSuccessful())


if __name__ == '__main__':
    raise SystemExit(main())
