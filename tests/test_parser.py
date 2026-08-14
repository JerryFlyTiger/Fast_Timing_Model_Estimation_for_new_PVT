import pytest

from liberty.parser import TABLE_KINDS, parse_file

from helpers import FULL_LIBS, PARTIAL_LIBS


@pytest.mark.parametrize("path", FULL_LIBS, ids=lambda p: p.name)
def test_full_lib_has_no_blank_tables(path):
    lib = parse_file(str(path))
    assert lib.library_name == path.stem
    assert lib.tables, "expected at least one table"
    blanks = [t for t in lib.tables if t.is_blank]
    assert blanks == []
    for t in lib.tables:
        assert t.values.shape == (7, 7)
        assert len(t.index_1) == 7
        assert len(t.index_2) == 7
        assert t.table_type in TABLE_KINDS


@pytest.mark.parametrize("path", PARTIAL_LIBS, ids=lambda p: p.name)
def test_partial_lib_blank_and_prefilled_zero_tables(path):
    lib = parse_file(str(path))
    blanks = [t for t in lib.tables if t.is_blank]
    nonblank = [t for t in lib.tables if not t.is_blank]
    assert len(blanks) + len(nonblank) == len(lib.tables)
    # Every value-carrying (non-blank) table in a partial file is a known
    # all-zero invalid power arc (see docs/plan.md rule 3 / QA A-something):
    # source-corner all-zero rise_power/fall_power stays zero in the target.
    for t in nonblank:
        assert t.table_type in ("rise_power", "fall_power")
        assert (t.values == 0).all()


def test_full_and_partial_same_process_share_table_keys():
    """Baseline copy (Phase 1 item 6) relies on positional/key alignment:
    the same (cell, pin, group_type, arc_index, table_type) key must
    identify the same logical table across corners of the same file."""
    full = parse_file(str(FULL_LIBS[0]))
    for partial_path in PARTIAL_LIBS:
        partial = parse_file(str(partial_path))
        assert set(full.tables_by_key) == set(partial.tables_by_key)


def test_index_1_identical_across_all_corners():
    """docs/plan.md: index_1 (input transition) is fixed across corners."""
    libs = [parse_file(str(p)) for p in FULL_LIBS]
    reference = libs[0]
    for key, table in reference.tables_by_key.items():
        for other in libs[1:]:
            assert other.tables_by_key[key].index_1 == table.index_1


def test_row_spans_point_at_the_row_text_in_source():
    lib = parse_file(str(FULL_LIBS[0]))
    table = lib.tables[0]
    for row_idx, (start, end) in enumerate(table.row_spans):
        row_text = lib.text[start:end]
        parsed_row = [float(x) for x in row_text.split(",")]
        assert parsed_row == list(table.values[row_idx])
