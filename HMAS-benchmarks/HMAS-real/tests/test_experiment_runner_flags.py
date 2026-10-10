"""
Regression test for the exact bug class found (and fixed) three times this
week: an enrichment flag defined via config._flag() in config.py silently
missing from the MAS_* env dict that experiment_runner.py builds for the
Hybrid architecture's subprocess. A flag missing there runs unenriched
without any error — this test makes that fail loudly instead.
"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))

import config
import experiment_runner


def _flag_env_names() -> set[str]:
    src = inspect.getsource(config)
    return set(re.findall(r'_flag\("(MAS_\w+)"', src))


def _hybrid_subprocess_env_names() -> set[str]:
    src = inspect.getsource(experiment_runner._run_hybrid_shot_subprocess)
    return set(re.findall(r'"(MAS_\w+)":', src))


def test_every_config_flag_is_set_in_hybrid_subprocess_env():
    flags = _flag_env_names()
    env_names = _hybrid_subprocess_env_names()
    missing = flags - env_names
    assert not missing, (
        f"Flags defined via config._flag() but missing from "
        f"_run_hybrid_shot_subprocess's env dict: {missing} — the Hybrid "
        f"architecture will silently run without these enrichments."
    )
