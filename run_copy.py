import time
import argparse
import os
import sys
import shutil
import subprocess
import signal
import threading
import queue
import copy
from ruamel.yaml import YAML

sys.path.append("preprocessing")
from generate_chirp import generate_from_yaml_filename
sys.path.append("postprocessing")
from save_data import save_data

"""
Provides a simple interface to build, run, and manage data outputs from the SDR code

Should generally be used like this:

runner = RadarProcessRunner(yaml_filename)
runner.setup() # Build the binary and get everything ready
runner.run() # Run the radar program
runner.wait() # Wait for it to finish (if `num_pulses` == -1, then you need to call runner.stop() before this will return)
runner.stop() # Wrap everything up and save data -- you need to call this even if the radar program finishes on its own
"""
class RadarProcessRunner():
    """
    yaml_filename -- path to the YAML config file that provides all the required settings
    output_log_path -- (temporary) location to store stdout of the C++ code
    log_processing_function -- optional function if you need to read and process the stdout data in real time
    output_to_stdout -- set True to also print the output, in addition to storing it to a file
    """
    def __init__(self, yaml_filename, output_log_path="uhd_stdout.log", log_processing_function=None, output_to_stdout=True):
        self.yaml_filename = yaml_filename
        self.output_log_path = output_log_path
        self.log_processing_function = log_processing_function
        self.output_to_stdout = output_to_stdout

        self.setup_complete = False
        self.is_running = False

        self.file_queue = queue.Queue()
        self.file_queue_size = 1
        self.output_file = None
        self.output_file_path = None

    """
    Manage the stdout of the radar program, including logging it to a file and optionally sending it for additional processing
    """
    def process_usrp_output(self, out, file_out, also_print=True):
        for line in iter(out.readline, ''):
            # Save output to specified file with timestamp
            t = time.time()
            file_out.write(f"[{t:0.3f}] \t{line}")

            # If provided, pass output to external function for processing
            if self.log_processing_function is not None:
                self.log_processing_function(line)

            # If specified, also print to stdout
            if also_print:
                print(f"[{t:0.3f}] \t{line}", end="")

            # Enqueue for saving somewhere else
            if line.startswith("[CLOSE FILE]"):
                filename = (line[13:]).strip()
                if filename.startswith("../../"): # Automatically added to escape cwd of binary
                    filename = filename[6:] # Strip it out
                self.file_queue.put(filename)

        out.close()
        file_out.close()
        self.file_queue_size = self.file_queue.qsize()

    """
    Build the radar program, generate the chirp, and get ready to run
    """
    def setup(self):

        # Verify CWD
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode('utf-8')
        git_root = "".join(git_root.split())
        cwd = os.getcwd()
        if os.path.normpath(git_root) != os.path.normpath(cwd):
            print(f"This script should ONLY be run from the root of the git repo ({git_root}). Detected CWD {cwd}")
            exit(1)

        # Load YAML
        yaml = YAML()
        with open(self.yaml_filename) as stream:
            self.config = yaml.load(stream)

        # Verify file save options
        if (self.config['RUN_MANAGER']['final_save_loc'] is None) and (self.config['RUN_MANAGER']['save_partial_files'] is False) and (self.config['FILES']['max_chirps_per_file'] != -1):
            print("You must choose to save at least some of your data. In yaml: file_save_loc cannot be empty and save_partial files cannot be false at the same time, unless .")
            exit(1)

        # Chirp generation
        try:
            generate_from_yaml_filename(self.yaml_filename)
        except Exception as e:
            print(e)
            exit(1)

        # Compile UHD program
        def run_and_fail_on_nonzero(cmd):
            retval = os.system(cmd)
            if retval != 0:
                print(f"Running '{cmd}' produced non-zero return value {retval}. Quitting...")
                exit(1)

        os.chdir("sdr/build")
        # Tests fetch googletest over the network; skip them here since running
        # the radar doesn't need them, and field deployment machines may be offline.
        run_and_fail_on_nonzero("cmake -DBUILD_TESTS=OFF ..")
        run_and_fail_on_nonzero("make")
        os.chdir("../..")

        self.setup_complete = True

    """
    Start the radar program
    """
    def run(self):
        if not self.setup_complete:
            raise Exception("Must call setup() before calling run(). If setup() does not complete successfully, you cannot call run().")

        self.gpspipe_process = subprocess.Popen(["/usr/bin/gpspipe", "--json", "-uu", "--output", "gpspipe_stdout.log"])
        self.gpxlogger_process = subprocess.Popen(["/usr/bin/gpxlogger", "--reconnect", "-m2", "-f", "track.gpx", "-e", "sockets"])
        time.sleep(5)
        self.uhd_process = subprocess.Popen(["./radar", self.yaml_filename], stdout=subprocess.PIPE, bufsize=1, close_fds=True, text=True, cwd="sdr/build")
        self.uhd_output_reader_thread = threading.Thread(target=self.process_usrp_output, args=(self.uhd_process.stdout, open('uhd_stdout.log', 'w'), self.output_to_stdout))
        self.uhd_output_reader_thread.daemon = True # thread dies with the program
        self.uhd_output_reader_thread.start()
        self.is_running = True

    """
    Wait (up to `timeout` seconds, if not None) for the process to complete
    """
    def wait(self, timeout = None):
        t = time.time()
        while self.uhd_process.returncode is None:
            self.uhd_process.poll()
            time.sleep(1)
            if (timeout is not None) and (time.time() - t > timeout):
                self.stop()

    """
    Ends the radar program (if not already terminated) and saves data
    """
    def stop(self, timeout = 10):
        if not self.is_running:
            return 0

        was_force_killed = False

        print("Attemping to stop UHD process")
        self.uhd_process.send_signal(signal.SIGINT)
        print(f"Waiting up to {timeout} seconds for the process to quit")
        t = time.time()
        try:
            self.uhd_process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            print(f"UHD process did not terminate within time limit. Killing...")
            self.uhd_process.kill()
            was_force_killed = True
        self.is_running = False

        self.gpspipe_process.kill()
        self.gpxlogger_process.kill()

        self.uhd_output_reader_thread.join()

        # If necessary, concatenate data files into a single file
        alternative_rx_samps_loc = None
        if (self.config['RUN_MANAGER']['final_save_loc'] is not None) and (self.config['FILES']['max_chirps_per_file'] != -1):
            print("Calling save_from_queue()")
            self.save_from_queue()
            alternative_rx_samps_loc = self.output_file_path

        # Save output
        print("Copying data files...")
        file_prefix = save_data(self.yaml_filename, alternative_rx_samps_loc=alternative_rx_samps_loc, num_files=self.file_queue_size,
                                extra_files={"uhd_stdout.log": "uhd_stdout.log",
                                             "gpspipe_stdout.log": "gpspipe_stdout.log",
                                             "track.gpx": "track.gpx"
                                             })
        print("Finished copying data.")

        self.output_file = None

        return file_prefix

    """
    Copy data from split files into a single data output file
    """
    def save_from_queue(self):
        if self.output_file is None:
            self.output_file_path = self.config['RUN_MANAGER']['final_save_loc']
            self.output_file = open(self.output_file_path, 'wb')

        while(not self.file_queue.empty()):
            with open(self.file_queue.get(), 'rb') as f:
                shutil.copyfileobj(f, self.output_file)

        self.output_file.close()


"""
Splits a base YAML config into `num_steps` per-step configs, each retuned to a
different RF0.freq, so that a total synthetic bandwidth wider than the SDR's
sample rate can be covered by a sequence of narrower subchirp acquisitions.

RF0.freq in the base config is treated as the LOW EDGE of the full span (not
its center). GENERATE.total_bandwidth must be an integer multiple of
GENERATE.chirp_bandwidth; if total_bandwidth is absent it defaults to
chirp_bandwidth (i.e. a single step, identical to today's behavior).

Returns a list of (step_yaml_path, center_freq) tuples, in step order.
"""
def generate_stepped_configs(yaml_filename):
    yaml = YAML()
    with open(yaml_filename) as stream:
        base_config = yaml.load(stream)

    chirp_bandwidth = base_config['GENERATE']['chirp_bandwidth']
    total_bandwidth = base_config['GENERATE'].get('total_bandwidth', chirp_bandwidth)
    low_edge_freq = base_config['RF0']['freq']

    num_steps = round(total_bandwidth / chirp_bandwidth)
    if abs(num_steps * chirp_bandwidth - total_bandwidth) > 1.0:
        raise ValueError(
            f"GENERATE.total_bandwidth ({total_bandwidth/1e6:.2f} MHz) must be an "
            f"integer multiple of GENERATE.chirp_bandwidth ({chirp_bandwidth/1e6:.2f} MHz)."
        )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(yaml_filename)), "_generated_stepped")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(yaml_filename))[0]

    step_configs = []
    for i in range(num_steps):
        center_freq = low_edge_freq + chirp_bandwidth * (i + 0.5)

        step_config = copy.deepcopy(base_config)
        step_config['RF0']['freq'] = center_freq

        final_save_loc = step_config['RUN_MANAGER'].get('final_save_loc')
        if final_save_loc is not None:
            root, ext = os.path.splitext(final_save_loc)
            step_config['RUN_MANAGER']['final_save_loc'] = f"{root}_step{i}{ext}"

        step_yaml_path = os.path.join(out_dir, f"{base_name}_step{i}_{center_freq/1e6:.0f}MHz.yaml")
        with open(step_yaml_path, 'w') as stream:
            yaml.dump(step_config, stream)

        step_configs.append((step_yaml_path, center_freq))

    return step_configs

"""
Runs a full stepped-frequency acquisition: one complete build/generate/run/save
cycle per subchirp, retuning RF0.freq between steps to tile the total
synthetic bandwidth requested via GENERATE.total_bandwidth in the YAML config.
"""
def run_stepped_acquisition(yaml_filename):
    step_configs = generate_stepped_configs(yaml_filename)

    current_runner = None
    def sigint_handler(signum, frame):
        if current_runner is not None:
            current_runner.stop() # On Ctrl-C, attempt to stop the active radar process
    signal.signal(signal.SIGINT, sigint_handler)

    file_prefixes = []
    for i, (step_yaml_path, center_freq) in enumerate(step_configs):
        print(f"--- Step {i+1}/{len(step_configs)}: center freq {center_freq/1e6:.2f} MHz ---")

        current_runner = RadarProcessRunner(step_yaml_path)
        current_runner.setup()
        current_runner.run()
        current_runner.wait()
        file_prefixes.append(current_runner.stop())

    print("--- Stepped acquisition complete ---")
    for (_, center_freq), prefix in zip(step_configs, file_prefixes):
        print(f"\t{center_freq/1e6:.2f} MHz -> {prefix}")


if __name__ == "__main__":

    # Check if a YAML file was provided as a command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", nargs='?', default='config/default.yaml',
            help='Path to YAML configuration file')
    parser.add_argument("--stepped", action="store_true",
            help='Run a stepped-frequency acquisition, retuning RF0.freq across '
                 'GENERATE.total_bandwidth / GENERATE.chirp_bandwidth subchirp steps')
    args = parser.parse_args()
    yaml_filename = args.yaml_file

    # Build and run UHD radar code

    if args.stepped:
        run_stepped_acquisition(yaml_filename)
    else:
        runner = RadarProcessRunner(yaml_filename)

        def sigint_handler(signum, frame):
            runner.stop() # On Ctrl-C, attempt to stop radar process

        runner.setup()
        runner.run()
        signal.signal(signal.SIGINT, sigint_handler)
        runner.wait()
        runner.stop()
