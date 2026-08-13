import os

from postprocessing.save_data import save_data


def test_save_data_skips_missing_optional_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "rx_samps.bin").write_bytes(b"sample-data")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
FILES:
  max_chirps_per_file: -1
  save_loc: data/rx_samps.bin
  gps_loc: gps_log.txt
RUN_MANAGER:
  save_partial_files: false
  save_gps: false
  final_save_loc: null
""".strip()
    )

    (tmp_path / "uhd_stdout.log").write_text("uhd output")

    file_prefix = save_data(
        str(config_path),
        extra_files={
            "uhd_stdout.log": "uhd_stdout.log",
            "missing.log": "missing.log",
        },
    )

    assert os.path.exists(file_prefix + "_uhd_stdout.log")
    assert not os.path.exists(file_prefix + "_missing.log")
