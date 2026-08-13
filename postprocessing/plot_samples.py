from signal import pause
import os
import sys
import argparse
import numpy as np
import scipy.signal as sp
import processing as pr
import matplotlib.pyplot as plt
from ruamel.yaml import YAML as ym
import os



REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_path(path):
    if not path:
        return path
    if os.path.isabs(path):
        return path
    cwd_candidate = os.path.normpath(os.path.join(os.getcwd(), path))
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    repo_candidate = os.path.normpath(os.path.join(REPO_ROOT, path))
    if os.path.exists(repo_candidate):
        return repo_candidate
    return repo_candidate


def build_plot_window(corr_sig, dir_peak, sample_rate, num_samps=2000, peak_offset=10):
    if len(corr_sig) == 0:
        raise ValueError("Correlation signal is empty.")

    start_idx = max(0, dir_peak - peak_offset)
    end_idx = min(len(corr_sig), start_idx + num_samps)
    if end_idx <= start_idx:
        raise ValueError("No samples available for plotting window.")

    window = corr_sig[start_idx:end_idx]
    x_time = (np.arange(len(window)) - (dir_peak - start_idx)) * 1e6 / sample_rate
    return x_time, window


# Check if a YAML file was provided as a command line argument
parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", nargs='?', default='config/default.yaml',
        help='Path to YAML configuration file')

args = parser.parse_args()

yaml_path = resolve_path(args.yaml_file)

# Initialize Constants
yaml = ym()                         # Always use safe load if not dumping
with open(yaml_path) as stream:
   config = yaml.load(stream)
   if "PLOT" in config:
       rx_params = config["PLOT"]
       sample_rate = rx_params["sample_rate"]    # Hertz
       rx_samps = resolve_path(rx_params["rx_samps"])          # Received data to analyze
       orig_ch = resolve_path(rx_params["orig_chirp"])         # Chirp associated with the received data
       direct_start = rx_params["direct_start"]
       echo_start = rx_params["echo_start"]
       sig_speed = rx_params["sig_speed"]
   else:
       sample_rate = config["GENERATE"]["sample_rate"]
       rx_samps = resolve_path(config["FILES"]["save_loc"])
       orig_ch = resolve_path(config["FILES"]["chirp_loc"])
       direct_start = 0
       echo_start = 1
       sig_speed = 3e8

print("--- Loaded constants from config.yaml ---")
# Read and plot RX/TX
for path, label in [(rx_samps, "RX samples"), (orig_ch, "transmitted chirp")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found: {path}")

rx_sig = pr.extractSig(rx_samps)
print("--- Plotting real samples read from %s ---" % rx_samps)
pr.plotChirpVsTime(rx_sig, 'Received Samples', sample_rate)

tx_sig = pr.extractSig(orig_ch)
print("--- Plotting transmited chirp, stored in %s ---" % orig_ch)
pr.plotChirpVsTime(tx_sig, 'Transmitted Chirp', sample_rate)

# Correlate the two chirps to determine time difference
print("--- Match filtering received chirp with transmitted chirp ---")
xcorr_sig = sp.correlate(rx_sig, tx_sig, mode='valid', method='auto')
# as finddirectpath is written right now, it must be called before taking log of the signal
# because if not, negative log values could have a greater absolute value than positive log values.
dir_peak = pr.findDirectPath(xcorr_sig, direct_start, True) 

#Test code, Only have one option uncommented--------
#Option 1:
xcorr_sig = 20 * np.log10(np.absolute(xcorr_sig))

#Option 2:
#xcorr_normalized = np.abs(xcorr_sig)
#xcorr_normalized /= np.max(xcorr_normalized)
#xcorr_sig = 20 * np.log10(xcorr_normalized + 0.00000000000000001)
#-------------------------------------------

print("--- Plotting result of match filter ---")
xcorr_samps = np.shape(xcorr_sig)[0]
print(echo_start)
xcorr_time = np.zeros(xcorr_samps)
for x in range (xcorr_samps):
    xcorr_time[x] = x * 1e6 /sample_rate

plt.figure()
plt.plot(xcorr_time, xcorr_sig)
plt.title("Output of Match Filter: Signal")
plt.xlabel('Time (us)')
plt.ylabel('Power [dB]')
plt.grid()
plt.show()

#COLIN TEST CODE----------
num_samps = 2000
x_time, plot_window = build_plot_window(xcorr_sig, dir_peak, sample_rate, num_samps=num_samps, peak_offset=10)

plt.figure()
plt.plot(x_time, plot_window)
plt.title("Output of Match Filter: Peaks")
plt.xlabel('Time (us)')
plt.ylabel('Power [dB]')
plt.grid()
#-------------------

#Unedited version of above code
# plt.figure()
# #plt.plot(range(-10,10000), xcorr_sig[dir_peak-10:dir_peak+10000])
# plt.title("Output of Match Filter: Peaks")
# plt.xlabel('Sample')
# plt.ylabel('Power [dB]')
# plt.grid()

[echo_samp, echo_dist] = pr.findEcho(xcorr_sig, sample_rate, dir_peak, echo_start, sig_speed, True)

sys.stdout.flush()
#plt.savefig("match_filter.png", dpi=150)
plt.show()