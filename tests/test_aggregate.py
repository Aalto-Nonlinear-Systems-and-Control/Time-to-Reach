from scripts.aggregate import main


def test_table_has_a_row_per_benchmark(tmp_path):
    out = tmp_path / "table"
    assert main(["--columns", "krtb:setting_0", "hj:setting_0", "--out", str(out)]) == 0
    md = (tmp_path / "table.md").read_text()
    assert md.count("| DUFF_FIN_") == 6 and "KRTB (setting_0)" in md
    assert (tmp_path / "table.csv").read_text().count("\n") == 17  # header + 16 benchmarks
    assert "krtb:setting_0:eigenpairs_s,krtb:setting_0:bounds_s" in (tmp_path / "table.csv").read_text()
    assert "(eigenpairs " in md and " + bounds " in md  # KRTB's time by step
    assert "\\begin{tabular}" in (tmp_path / "table.tex").read_text()
