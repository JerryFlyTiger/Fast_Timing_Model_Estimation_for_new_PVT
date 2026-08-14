"""The two composition-audit scripts must agree about what the candidate
bucket schemes are.

`scripts/phase4_composition_audit.py` computes one number under one
scheme; `scripts/phase4_composition_sensitivity.py` computes the range
over all of them, and docs/round_20260810.md section 7.3 quotes that range as
the audit's real precision. If the two lists drift apart, the quoted
range silently stops covering what the point-estimate script can
actually produce -- e.g. a scheme added to the audit but not the sweep
would make the documented range narrower than reality, which is exactly
the kind of unfalsifiable-number problem section 7 exists to fix.

The scripts live outside `src/` and manipulate `sys.path` at import
time, so they are loaded by file path rather than imported normally.
"""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(script_name):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_sweep_covers_every_audit_scheme():
    audit = _load("phase4_composition_audit.py")
    sweep = _load("phase4_composition_sensitivity.py")
    assert audit.BUCKET_SCHEMES == sweep.SCHEMES


def test_audit_has_no_default_bucket_scheme():
    """--bucket-scheme is deliberately required: no criterion selects a
    scheme (section 7.3), and a default would let a caller quote a
    scheme-conditional number without knowing it was conditional."""
    audit = _load("phase4_composition_audit.py")
    assert not hasattr(audit, "DEFAULT_SCHEME")


def test_none_scheme_really_disables_drive_matching():
    """The 'none' arm is the control that reproduces the superseded
    anchor-ratio audit; it must put every drive strength in one bucket."""
    audit = _load("phase4_composition_audit.py")
    bucket_of, _ = audit.make_bucket_of("none")
    labels = {bucket_of(f"INVM{d}") for d in (1, 2, 4, 8, 16, 48)}
    assert len(labels) == 1


def test_every_scheme_assigns_all_observed_drive_strengths():
    """Drive 14 exists in beta100 but not in train400, and drives up to
    48 exist in train400 -- a scheme with a gap would raise mid-run,
    after minutes of parsing."""
    audit = _load("phase4_composition_audit.py")
    observed = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 32, 36, 40, 48]
    for scheme in audit.BUCKET_SCHEMES:
        bucket_of, _ = audit.make_bucket_of(scheme)
        for d in observed:
            bucket_of(f"INVM{d}")  # must not raise
