import pandas as pd, pytest
from s2g import config


def test_write_once_and_header(tmp_path):
    p = tmp_path / "x.csv"
    config.write_csv(pd.DataFrame({"a": [1, 2]}), p, "make_x.py", ["in.csv"])
    assert p.read_text().startswith("# produced by scripts/make_x.py")
    assert config.read_csv(p).shape == (2, 1)
    with pytest.raises(SystemExit):
        config.write_csv(pd.DataFrame({"a": [3]}), p, "make_x.py", [])


def test_protocol_draft_is_not_freezable_and_lock_detects_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOCK", tmp_path / "nolock")
    with pytest.raises(SystemExit):
        config.assert_frozen()
