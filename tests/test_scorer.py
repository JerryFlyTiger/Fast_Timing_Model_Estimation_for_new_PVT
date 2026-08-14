import numpy as np
import pytest

from scoring.scorer import compare_libs, point_errors, score_arrays, score_from_errors

from helpers import FULL_LIBS
from liberty.parser import parse_file


def test_point_errors_basic_relative_error():
    y = np.array([1.0, 2.0, 10.0])
    yhat = np.array([1.1, 1.8, 5.0])
    errs = point_errors(y, yhat)
    np.testing.assert_allclose(errs, [0.1, 0.1, 0.5])


def test_point_errors_caps_at_one():
    y = np.array([1.0])
    yhat = np.array([100.0])  # 9900% relative error
    errs = point_errors(y, yhat)
    assert errs[0] == 1.0


def test_point_errors_zero_true_value_convention():
    y = np.array([0.0, 0.0])
    yhat = np.array([0.0, 1e-9])
    errs = point_errors(y, yhat)
    assert errs[0] == 0.0  # exact match at zero -> perfect
    assert errs[1] == 1.0  # any nonzero prediction at a zero target -> fail


def test_score_perfect_prediction_is_100():
    y = np.array([1.0, 2.0, 3.0, 0.0])
    assert score_arrays(y, y) == pytest.approx(100.0)


def test_score_all_saturated_errors_is_zero():
    errs = np.ones(10)
    assert score_from_errors(errs) == pytest.approx(0.0)


def test_score_formula_matches_definition():
    y = np.array([1.0, 2.0, 4.0])
    yhat = np.array([1.1, 2.2, 3.6])
    errs = np.minimum(1.0, np.abs(y - yhat) / np.abs(y))
    expected = 100.0 - 100.0 * np.sqrt(np.mean(errs**2))
    assert score_arrays(y, yhat) == pytest.approx(expected)


def test_compare_libs_identical_file_scores_100():
    lib = parse_file(str(FULL_LIBS[0]))
    report = compare_libs(lib, lib)
    assert report.overall == pytest.approx(100.0)
    for table_type, s in report.by_table_type.items():
        assert s.score == pytest.approx(100.0)


def test_compare_libs_breakdown_covers_all_table_types_present():
    lib = parse_file(str(FULL_LIBS[0]))
    report = compare_libs(lib, lib)
    present_types = {t.table_type for t in lib.tables}
    assert set(report.by_table_type) == present_types
