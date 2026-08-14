"""Guard the two pure helpers that section 8.10's verdict rests on.

docs/round_20260810.md section 8.10 rules a pre-registered hypothesis
refuted. Its numbers come from `scripts/phase4_topology_compare.py`,
which reads per-corner scores back out of the run logs -- the alpha and
beta huber/mse pairs left no error dumps, so the logs are the only
record. A parsing slip there would move a published conclusion, and a
cold read found the glob is wider than the files it means to read
(it also matches sweep-driver and sensitivity logs sitting in the same
directories). These tests pin the two things that would silently move a
number: what gets parsed out of the logs, and how per-corner scores pool.

The score dumps are gitignored, so nothing here touches them; the run
logs are tracked, which is what makes this testable at all.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    path = REPO_ROOT / "scripts" / "phase4_topology_compare.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tc = _load()

# The target corner set of each topology, from src/models/phase4_features.py.
# Written out rather than imported so that a change to the topology
# definitions has to be acknowledged here too -- these names are what
# section 8.4's pairwise intersections are built from.
TARGETS = {
    "alpha": {"ff0p88v125c", "ff0p88vm40c", "ff1p1v125c", "ff1p1vm40c",
              "ss0p72v125c", "ss0p72vm40c", "ss0p9v125c", "ss0p9vm40c",
              "tt0p8v25c", "tt1p0v25c"},
    "beta": {"ff0p88v125c", "ff0p88vm40c", "ff0p99v125c", "ff0p99vm40c",
             "ss0p72v125c", "ss0p72vm40c", "ss0p81v125c", "ss0p81vm40c",
             "tt0p8v25c", "tt0p9v25c"},
    "final": {"ff0p99v125c", "ff0p99vm40c", "ff1p1v125c", "ff1p1vm40c",
              "ss0p81v125c", "ss0p81vm40c", "ss0p9v125c", "ss0p9vm40c",
              "tt0p9v25c", "tt1p0v25c"},
}


@pytest.mark.parametrize("key", sorted(tc.RUN_LOGS))
def test_run_logs_parse_to_exactly_that_topologys_ten_targets(key):
    """Six log sets, ten corners each, and the right ten.

    This is the check that catches a transposed RUN_LOGS entry, which
    would otherwise produce a plausible-looking but wrong gain: the
    alpha mse control lives in a different directory and under a
    different prefix from every other run, so it is the easy one to
    mis-wire.
    """
    stage, _ = key
    subdir, prefix = tc.RUN_LOGS[key]
    scores = tc.scores_from_logs(str(REPO_ROOT / "logs"), subdir, prefix)
    assert set(scores) == TARGETS[stage]
    assert all(0.0 < s < 100.0 for s in scores.values())


def test_driver_logs_are_ignored():
    """`final_control_*.log` matches more files than it should.

    Four of them are run_corner_sweep.sh's own stdout (the sweep,
    resume, and finish drivers). They are skipped only because no other
    script prints the per-corner summary header -- an assumption this
    test turns into a failure if it ever stops holding.
    """
    log_dir = REPO_ROOT / "logs" / "round_20260811"
    matched = sorted(p.name for p in log_dir.glob("final_control_*.log"))
    non_corner = [n for n in matched
                  if n[len("final_control_"):-len(".log")] not in TARGETS["final"]]
    assert non_corner, "expected the driver logs that make this test worth having"
    scores = tc.scores_from_logs(str(REPO_ROOT / "logs"), "round_20260811",
                                 "final_control")
    assert set(scores) == TARGETS["final"]


def test_pooled_line_after_the_blank_is_not_read_as_a_corner(tmp_path):
    """The summary block ends at the first blank line.

    Without that, the `  pooled: 95.8165` further down the same log
    matches the score regex and lands in the dict under the key
    "pooled:", which would then be pooled in as an eleventh corner.
    """
    log_dir = tmp_path / "round"
    log_dir.mkdir()
    (log_dir / "x_tt0p8v25c.log").write_text(
        "noise before\n"
        f"{tc.SCORE_HEADER}\n"
        "  corner                score\n"
        "  tt0p8v25c           96.4224\n"
        "\n"
        "=== pooled overall across 1 delivery corner(s) ===\n"
        "  pooled: 96.4224\n",
        encoding="utf-8",
    )
    assert tc.scores_from_logs(str(tmp_path), "round", "x") == {"tt0p8v25c": 96.4224}


def test_aborted_log_without_the_summary_block_is_skipped(tmp_path):
    """Section 8.9's silent aborts leave a log with no summary block.

    Skipping them is the same success test run_corner_sweep.sh uses; a
    parser that instead crashed or returned a partial dict would have
    made the 8/10 state of section 8.8 unreadable.
    """
    log_dir = tmp_path / "round"
    log_dir.mkdir()
    (log_dir / "x_tt0p8v25c.log").write_text(
        "=== delivery corner: tt0p8v25c ===\n  dev-train rows=533022\n",
        encoding="utf-8")
    (log_dir / "x_tt0p9v25c.log").write_text(
        f"{tc.SCORE_HEADER}\n  corner  score\n  tt0p9v25c  97.4367\n",
        encoding="utf-8")
    assert tc.scores_from_logs(str(tmp_path), "round", "x") == {"tt0p9v25c": 97.4367}


def test_over_indented_row_inside_the_block_is_rejected(tmp_path):
    """The two-space anchor is a second, independent guard.

    The blank-line break bounds WHERE the block ends; the `^ {2}` anchor
    polices what counts as a row inside it. A cold read showed those are
    not redundant -- today they are merely indistinguishable, because
    both scripts that print the header happen to use exactly two spaces.
    A third caller with a different indent would be read silently by a
    looser anchor, so the distinction gets a test rather than a comment.
    """
    log_dir = tmp_path / "round"
    log_dir.mkdir()
    (log_dir / "x_tt0p8v25c.log").write_text(
        f"{tc.SCORE_HEADER}\n"
        "  corner                score\n"
        "  tt0p8v25c           96.4224\n"
        "    stray_corner      12.3456\n",
        encoding="utf-8",
    )
    assert tc.scores_from_logs(str(tmp_path), "round", "x") == {"tt0p8v25c": 96.4224}


def test_preregistered_verdict_needs_both_conjuncts():
    """8.7's claim is `>= +0.11` AND `largest of three`.

    The first implementation checked only the second conjunct, which
    would have reported a +0.09-but-largest re-run as upholding the
    claim. Each conjunct therefore gets to fail on its own here.
    """
    assert tc.preregistered_verdict(0.12, "final").endswith("成立")
    assert "不成立" not in tc.preregistered_verdict(0.12, "final")

    only_top = tc.preregistered_verdict(0.09, "final")   # largest, too small
    assert "不成立" in only_top and "+0.0900 < +0.11" in only_top
    assert "最大的是" not in only_top

    only_big = tc.preregistered_verdict(0.12, "beta")    # big enough, not largest
    assert "不成立" in only_big and "最大的是 beta" in only_big
    assert "<" not in only_big

    both = tc.preregistered_verdict(0.0744, "beta")      # the actual 8.10 result
    assert "不成立" in both and "+0.0744 < +0.11" in both and "最大的是 beta" in both


def test_preregistered_verdict_formats_a_negative_gain_with_one_sign():
    """A gain below zero must not print as `+-0.0193`."""
    line = tc.preregistered_verdict(-0.0193, "beta")
    assert "-0.0193 < +0.11" in line
    assert "+-" not in line


def _errors_scoring(score, n=8):
    """A constant error vector whose score is exactly `score`.

    score = 100*(1 - rms), so a constant |e| = 1 - score/100 inverts it.
    Lets verdict()'s dump side be built without the real .npz files,
    which are gitignored and 40 MB each.
    """
    return np.full(n, 1.0 - score / 100.0)


def _verdict_args():
    """The four arguments verdict() takes, all self-consistent.

    Built from the tracked logs so the cross-checks pass, which is what
    makes a deliberately broken variant meaningful below.
    """
    logs_root = str(REPO_ROOT / "logs")
    huber_ref = {stage: tc.scores_from_logs(logs_root, *tc.RUN_LOGS[(stage, "huber")])
                 for stage in TARGETS}
    mse_ref = {c: _errors_scoring(s) for c, s in
               tc.scores_from_logs(logs_root, *tc.RUN_LOGS[("final", "mse")]).items()}
    flip_share = {"alpha": 63.4, "beta": 69.2, "final": 71.8}
    return logs_root, huber_ref, mse_ref, flip_share


def test_verdict_accepts_a_consistent_log_and_dump_pair():
    assert tc.verdict(*_verdict_args()) == 0


def test_verdict_stops_when_a_huber_dump_disagrees_with_its_log():
    """The log/dump agreement check is the only thing standing between a
    mis-parsed log and a published conclusion, so it must actually bite."""
    logs_root, huber_ref, mse_ref, flip_share = _verdict_args()
    victim = sorted(huber_ref["beta"])[0]
    huber_ref["beta"][victim] += 0.001          # twice the 5e-4 tolerance
    assert tc.verdict(logs_root, huber_ref, mse_ref, flip_share) == 1


def test_verdict_stops_when_the_final_mse_dump_disagrees_with_its_log():
    logs_root, huber_ref, mse_ref, flip_share = _verdict_args()
    victim = sorted(mse_ref)[0]
    mse_ref[victim] = _errors_scoring(50.0)
    assert tc.verdict(logs_root, huber_ref, mse_ref, flip_share) == 1


def test_verdict_diagnoses_a_stray_dump_instead_of_raising(capsys):
    """The dump directories are a gitignored hand-managed cache.

    One stray .npz there used to index into the log-side dict and raise
    a bare KeyError, which reads as a crash rather than as the data
    problem it is.
    """
    logs_root, huber_ref, mse_ref, flip_share = _verdict_args()
    mse_ref["ss0p72v125c"] = _errors_scoring(95.0)   # a beta corner, not final's
    assert tc.verdict(logs_root, huber_ref, mse_ref, flip_share) == 1
    assert "corner 集合" in capsys.readouterr().out


def test_pool_is_linear_in_mean_squared_error_not_in_scores():
    """Pooling averages e^2, never the scores.

    Section 8.8 records this being got wrong by hand once already
    (+0.24 instead of +0.307). With equal point counts per corner the
    two differ by a real amount, so an implementation that averaged the
    scores would fail here rather than merely be imprecise.
    """
    scores = [90.0, 99.0]
    expected = 100.0 - 100.0 * np.sqrt((0.10 ** 2 + 0.01 ** 2) / 2)
    assert tc.pool(scores) == pytest.approx(expected)
    assert tc.pool(scores) != pytest.approx(np.mean(scores), abs=1e-3)


def test_pool_reproduces_score_from_errors_on_concatenated_corners():
    """pool() and the concatenate path must agree on equal-sized corners.

    Section 8.8 pools by concatenating point errors; 8.10 pools from
    per-corner scores because the alpha/beta mse sides have no dumps.
    The two paths are only interchangeable while every target corner
    carries the same point count, which is the premise pool() documents.
    """
    rng = np.random.default_rng(0)
    corners = [np.abs(rng.normal(0.05, 0.02, size=500)) for _ in range(4)]
    from scoring.scorer import score_from_errors

    direct = score_from_errors(np.concatenate(corners))
    pooled = tc.pool([score_from_errors(e) for e in corners])
    assert pooled == pytest.approx(direct, abs=1e-9)
