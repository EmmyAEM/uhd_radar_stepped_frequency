#!/home/emmyalondraem/reusum2026/uhd_radar_code_2026/uhd_radar_2026/Python_scripting/.venv/bin/python
import loopback_testing as lt
import argparse
import glob
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('postprocessing')
sys.path.append('../data/bandwidth_Span')
import processing as pr

from ruamel.yaml import YAML as ym

# Check if a YAML file was provided as a command line argument
parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", nargs='?', default='config/default.yaml',
        help='Path to YAML configuration file')
parser.add_argument("--combine", "-c", action="store_true",
        help='Combine all plots into a single figure/window instead of opening one window per plot '
             '(with --dir, this puts the range/freq plots and the per-step FFT grid side by side in '
             'one window instead of showing them one at a time)')
parser.add_argument("--dir", "-d", default=None,
        help="Directory containing multiple '*_config.yaml' files from a stepped-frequency "
             "acquisition (see run.py --stepped). When given, every subchirp step found in the "
             "directory is overlaid on the same plots (sorted by RF0.freq), and yaml_file is "
             "ignored.")
parser.add_argument("--range-offset", type=float, default=0.0,
        help="With --dir: constant distance in meters subtracted from every range-profile sample, "
             "to correct for fixed hardware delay in the FMCW/deramp range calculation (which has no "
             "built-in calibration). Tune this until the measured peak lines up with your known coax "
             "length -- the console prints both after each run so you can iterate.")
parser.add_argument("--total-span", type=float, default=1.0,
        help="With --dir: compress the freq-vs-time x-axis so the whole multi-step sequence spans "
             "this many seconds, keeping each step's short chirp ramp visible (default: 1.0). Pass 0 "
             "to disable compression and show true real-world acquisition timing instead, in which "
             "case each chirp ramp will appear as a thin vertical sliver.")
args = parser.parse_args()

if args.dir is not None:
    config_paths = sorted(
        glob.glob(os.path.join(args.dir, "*_config.yaml")),
        key=lambda p: pr.load_config(p)['RF0']['freq']
    )
    if not config_paths:
        raise ValueError(f"No '*_config.yaml' files found in directory: {args.dir}")

    prefixes = [p[:-len("_config.yaml")] for p in config_paths]

    print(f"Found {len(prefixes)} step(s) in {args.dir}:")
    for prefix in prefixes:
        freq = pr.load_config(prefix)['RF0']['freq']
        print(f"\t{freq/1e6:.2f} MHz -> {prefix}")

    total_span = args.total_span if args.total_span > 0 else None
    lt.main_multi(prefixes, combine=args.combine, range_offset=args.range_offset,
                  total_span=total_span)
    sys.exit(0)

str_arg = str(args.yaml_file)
print("original string is:", str_arg)

# Derive the run's file prefix straight from the yaml path itself (rather than
# assuming a fixed "data/" folder depth), so it works no matter which folder
# under data/ the run's files live in.
config_suffix = "_config.yaml"
if not str_arg.endswith(config_suffix):
    raise ValueError(f"Expected a '*{config_suffix}' file, got: {str_arg}")
prefix = str_arg[:-len(config_suffix)]  # e.g. "data/bandwidth_Span/20260219_202834"
timestamp = os.path.basename(prefix)
print("timestamp is:", timestamp)

# Initialize Constants
yaml = ym()                         # Always use safe load if not dumping
with open(args.yaml_file) as stream:
   config = yaml.load(stream)
   rx_params = config["PLOT"]
   sample_rate = rx_params["sample_rate"]    # Hertz

   direct_start = rx_params["direct_start"]
   echo_start = rx_params["echo_start"]
   sig_speed = rx_params["sig_speed"]

   print("The timestamp is: ", timestamp)

   # Derive the samples file from prefix (like processing.py does elsewhere) rather
   # than trusting config["PLOT"]["rx_samps"] -- that path is recorded at capture
   # time and goes stale if the run's files are later moved (e.g. into a backup dir).
   rx_samps = prefix + "_rx_samps.bin"

   rx_sig = pr.extractSig(rx_samps)
   # Reconstruct the chirp from this run's own config rather than reading
   # orig_chirp off disk -- that path (e.g. "data/chirp.bin") is a shared
   # scratch file that later runs overwrite, so it may not match this run.
   _, tx_sig = lt.generate_chirp(config)
   print("Loaded data")

   print("rx size is first, tx is second")
   print(rx_sig.shape)
   print(tx_sig.shape)

   if args.combine:
       # Lay chirp (real/imag) alongside the main() plots (matched filter, raw
       # matched, compressed, freq-vs-time, and the two FFT coefficient plots)
       # in one shared window, instead of each plot opening its own window.
       fig, combined_axs = plt.subplots(2, 4, figsize=(20, 8))
       lt.plot_chirp(tx_sig, sample_rate, axs=combined_axs[0, :2], show=False)
       print("Plotting chirp")

       print("running main") #Chris' loopback_testing code
       lt.main(prefix, combine=True, axs=(
           combined_axs[0, 2], combined_axs[1, 0], combined_axs[1, 1],
           combined_axs[1, 2], combined_axs[0, 3], combined_axs[1, 3],
       ))

       fig.tight_layout()
       plt.show()
   else:
       lt.plot_chirp(tx_sig, sample_rate)
       print("Plotting chirp")

       print("running main") #Chris' loopback_testing code
       lt.main(prefix)