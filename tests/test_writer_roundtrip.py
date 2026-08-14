import numpy as np
import pytest

from liberty.parser import parse_file, parse_text
from liberty.writer import fill_template

from helpers import BLANK_ROW, FULL_LIBS, PARTIAL_LIBS, skeleton


def _blank_out(lib):
    """Build a synthetic 'partial-style' text from a fully-populated
    LibertyFile by replacing every row's quoted content with the blank
    placeholder used by testcase/alpha_test/partial/*.lib."""
    edits = []
    for table in lib.tables:
        for start, end in table.row_spans:
            edits.append((start, end))
    edits.sort()

    pieces = []
    cursor = 0
    for start, end in edits:
        pieces.append(lib.text[cursor:start])
        pieces.append(BLANK_ROW)
        cursor = end
    pieces.append(lib.text[cursor:])
    return "".join(pieces)


@pytest.mark.parametrize("path", FULL_LIBS, ids=lambda p: p.name)
def test_full_lib_roundtrip_is_byte_exact(path):
    """blank-out a full lib -> parse the blanked text -> fill back in with
    the original values -> must reproduce the original file byte for
    byte. This is the acceptance-critical round-trip test."""
    lib = parse_file(str(path))

    blanked_text = _blank_out(lib)
    blanked_lib = parse_text(blanked_text)
    assert set(blanked_lib.tables_by_key) == set(lib.tables_by_key)
    assert all(t.is_blank for t in blanked_lib.tables)

    predictions = {key: table.values for key, table in lib.tables_by_key.items()}
    filled_text = fill_template(blanked_lib, predictions)

    assert filled_text == lib.text
    assert len(filled_text) == len(lib.text)


@pytest.mark.parametrize("path", PARTIAL_LIBS, ids=lambda p: p.name)
def test_partial_template_fill_preserves_everything_but_values(path):
    """Filling a real partial file's blanks with arbitrary fake values must
    leave every byte outside values(...) blocks identical to the
    original template, and must leave already-filled (all-zero) power
    tables completely untouched even if a prediction is supplied for
    them."""
    lib = parse_file(str(path))
    original_text = lib.text

    rng = np.random.default_rng(42)
    predictions = {}
    for key, table in lib.tables_by_key.items():
        # Supply a fake prediction for every key, including the 13
        # already-filled all-zero ones, to prove the writer ignores
        # predictions for non-blank tables.
        fake = rng.uniform(-1000, 1000, size=(7, 7))
        predictions[key] = fake

    filled_text = fill_template(lib, predictions)

    # 1. Structure outside values(...) blocks is untouched.
    assert skeleton(filled_text) == skeleton(original_text)

    # 2. Re-parse the output and check per-table semantics.
    filled_lib = parse_text(filled_text)
    for key, table in lib.tables_by_key.items():
        filled_table = filled_lib.tables_by_key[key]
        if table.is_blank:
            np.testing.assert_allclose(
                filled_table.values, predictions[key], rtol=1e-5, atol=1e-12
            )
        else:
            # Known-zero invalid power arc: must be byte-for-byte
            # unchanged, i.e. still all zero, not overwritten by the
            # fake prediction. Absolute offsets shift downstream of any
            # edit (replacement text length != blank placeholder length),
            # so compare each row's own text via its *own* freshly-parsed
            # span rather than reusing the pre-edit absolute offsets.
            assert (filled_table.values == 0).all()
            for (s, e), (fs, fe) in zip(table.row_spans, filled_table.row_spans):
                assert original_text[s:e] == filled_text[fs:fe]
