import sys
import copy
import shutil
import argparse
import numpy as np
import scipy.signal as sp
import processing as pr
import matplotlib.pyplot as plt
from datetime import datetime
from ruamel.yaml import YAML as ym

def save_data(yaml_filename, extra_files={}, alternative_rx_samps_loc=None, num_files=1):
    # Initialize Constants
    yaml = ym()
    with open(yaml_filename) as stream:
        config = yaml.load(stream)

    file_prefix = datetime.now().strftime("data/%Y%m%d_%H%M%S")

    print(f"Copying data to {file_prefix}...")

    shutil.copy(yaml_filename, file_prefix + "_config.yaml")
    if config['FILES']['max_chirps_per_file'] == -1:
            
            shutil.move(config['FILES']['save_loc'], file_prefix + "_rx_samps.bin")
    else:
        if config['RUN_MANAGER']['save_partial_files']:
            base_filename = config['FILES']['save_loc']
            for i in range(num_files):
                f = base_filename + "." + str(i)
                shutil.copy(f, file_prefix + "_p" + str(i) + "_rx_samps.bin")
        if alternative_rx_samps_loc is not None:
            shutil.copy(alternative_rx_samps_loc, file_prefix + "_rx_samps.bin")

    for source_file, dest_tag in extra_files.items():
        shutil.copy(source_file, file_prefix + "_" + dest_tag)

    if config['RUN_MANAGER']['save_gps']:
        shutil.copy(config['FILES']['gps_loc'], file_prefix + "_gps_log.txt")

    print(f"File copying complete.")

    return file_prefix

def save_stepped_data(yaml_filename, step_results, extra_files={}):
    """
    Save data from an in-process stepped-frequency acquisition (see the
    per-step loop in sdr/main.cpp, and RadarProcessRunner.process_usrp_output's
    "[STEP BEGIN]"/"[STEP DONE]" marker parsing in run.py). Each step is a
    different RF sub-band and gets its own file set -- unlike save_data(),
    these are never merged together.

    step_results: list of (step_index, freq_hz, data_filename, log_lines)
        tuples, one per completed step (see RadarProcessRunner.step_results).
        log_lines is that step's own slice of the run's stdout, so each
        step's saved log contains only its own "[START] ..." line -- no
        changes are needed in postprocessing/processing.py to find the right
        one per step.
    extra_files: run-level (not per-step) files to copy once, same as
        save_data()'s extra_files -- e.g. {"gpspipe_stdout.log": "gpspipe_stdout.log"}.

    Returns the shared run-level file_prefix (without any per-step suffix).
    """
    yaml = ym()
    with open(yaml_filename) as stream:
        base_config = yaml.load(stream)

    file_prefix = datetime.now().strftime("data/%Y%m%d_%H%M%S")

    print(f"Copying stepped acquisition data to {file_prefix}...")

    for index, freq, data_filename, log_lines in step_results:
        step_prefix = f"{file_prefix}_step{index}_{round(freq / 1e6)}MHz"

        # Per-step config snapshot with RF0.freq overridden to this step's
        # actual frequency, so postprocessing (which reads RF0.freq from
        # "<prefix>_config.yaml") sees the right value per step.
        step_config = copy.deepcopy(base_config)
        step_config['RF0']['freq'] = freq
        with open(step_prefix + "_config.yaml", 'w') as stream:
            yaml.dump(step_config, stream)

        shutil.move(data_filename, step_prefix + "_rx_samps.bin")

        with open(step_prefix + "_uhd_stdout.log", 'w') as stream:
            stream.writelines(log_lines)

    for source_file, dest_tag in extra_files.items():
        shutil.copy(source_file, file_prefix + "_" + dest_tag)

    if base_config['RUN_MANAGER']['save_gps']:
        shutil.copy(base_config['FILES']['gps_loc'], file_prefix + "_gps_log.txt")

    print(f"File copying complete.")

    return file_prefix

if __name__ == "__main__":
    # Check if a YAML file was provided as a command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", nargs='?', default='config/default.yaml',
            help='Path to YAML configuration file')
    args = parser.parse_args()
    yaml_filename = args.yaml_file

    save_data(yaml_filename)

