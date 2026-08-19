from signal import pause
import sys
import os
import argparse
import numpy as np
import scipy.signal as sp
import processing as pr
import matplotlib.pyplot as plt
from ruamel.yaml import YAML as ym


# uhd_radar_2026/ is the project root; run.py enforces it as the cwd for
# recording, and all "data/..." paths in config yaml files are relative to it.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check if a YAML file was provided as a command line argument
default_yaml = os.path.join(project_root, 'config', 'default.yaml')
parser = argparse.ArgumentParser()
parser.add_argument("yaml_file", nargs='?', default=default_yaml,
        help='Path to YAML configuration file (relative to the cwd or, '
             'like paths in the config files themselves, to the project root)')
args = parser.parse_args()

# A relative yaml_file is conventionally written relative to the project
# root (e.g. "config/default.yaml"), so fall back to that if it isn't found
# relative to the cwd.
yaml_file = args.yaml_file
if not os.path.isabs(yaml_file) and not os.path.exists(yaml_file):
    candidate = os.path.join(project_root, yaml_file)
    if os.path.exists(candidate):
        yaml_file = candidate

# Initialize Constants
yaml = ym()                         # Always use safe load if not dumping
with open(yaml_file) as stream:
   config = yaml.load(stream)
   rx_params = config["PLOT"]
   sample_rate = rx_params["sample_rate"]    # Hertz
   rx_samps = os.path.join(project_root, rx_params["rx_samps"])    # Received data to analyze
   orig_ch = os.path.join(project_root, rx_params["orig_chirp"])   # Chirp associated with the received data
   direct_start = rx_params["direct_start"]
   echo_start = rx_params["echo_start"]
   sig_speed = rx_params["sig_speed"]
   
print("--- Loaded constants from config.yaml ---")

# Read and plot RX/TX
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

# plt.figure()
# plt.plot(xcorr_time, xcorr_sig)
# plt.title("Output of Match Filter: Signal")
# plt.xlabel('Time (us)')
# plt.ylabel('Power [dB]')
# plt.grid()

#COLIN TEST CODE----------
numSamps = 2000
# Clamp to the bounds of xcorr_sig -- short captures (e.g. few pulses/short
# rx_duration) may not have numSamps samples past the direct path peak.
plot_start = max(dir_peak - 10, 0)
plot_end = min(dir_peak + numSamps, xcorr_samps)
xTime = (np.arange(plot_start, plot_end) - dir_peak) * 1e6 / sample_rate

plt.figure()
plt.plot(xTime, xcorr_sig[plot_start:plot_end])
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