import os
import sys

from run import RadarProcessRunner


def test_missing_optional_process_reports_path_and_package_hint(monkeypatch, capsys):
    monkeypatch.setenv("PATH", "/tmp/does-not-exist")

    runner = RadarProcessRunner("config/default.yaml")
    process = runner._start_optional_process("gpspipe", ["--json", "-uu"], "gpspipe")

    captured = capsys.readouterr()
    assert process is None
    assert "gpspipe" in captured.out
    assert "PATH" in captured.out
    assert "gpsd" in captured.out or "gpsd-clients" in captured.out
