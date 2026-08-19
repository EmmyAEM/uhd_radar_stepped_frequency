import numpy as np
import sys
sys.path.append("postprocessing")
import processing
import matplotlib
matplotlib.use("TkAgg")  # avoid Qt/Wayland plugin issues
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
sys.path.append("preprocessing")
from generate_chirp import generate_chirp


def load_data(prefix):
    """
    Load radar data from the specified file prefix.
    """
    return processing.load_radar_data(prefix)

def matched_filter(rx, tx):
    """
    Apply matched filtering to the received signal using the transmitted signal.

    Args:
        rx: Received signal.
        tx: Transmitted signal.
    """
    tx_mf = np.conjugate(tx[::-1]) # complex conjugate of time reversed tx signal for convolution
    return sp.fftconvolve(rx, tx_mf[:, None], mode='full', axes=0)

def plot_compress(rx, tx, sample_rate, ax=None, show=True, label=None, color=None):
    rx_sum = np.sum(rx, axis=1)
    n = len(rx_sum)
    tx_fft = np.fft.fft(tx, n=n)  # zero-pad chirp spectrum to match rx length
    compressed = np.fft.ifftshift(np.fft.ifft(np.fft.fft(rx_sum) * np.conjugate(tx_fft)))
    # convert to distance
    c = 299792458.0  # speed of light in vacuum (m/s)
    v = 2/3 * c
    n = np.arange(len(compressed))
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(n, 20 * np.log10(np.abs(compressed)), label=label, color=color)
    ax.set_title("Compressed Signal")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Normalized Amplitude (dB)")
    ax.grid()
    if show:
        plt.show()

def plot_chirp(tx, sample_rate, axs=None, show=True):
    """
    Plot the transmitted chirp signal.
    """
    processing.plotChirpVsTime(tx, "chirp", sample_rate, axs=axs)
    if show:
        plt.show()

def plot_freq_vs_time(signal, sample_rate, title="Frequency vs Time", label=None, ax=None, show=True, freq_offset=0.0, color=None, amplitude_threshold=0.1):
    """
    Calculate and plot the instantaneous frequency of a complex signal over time.

    Instantaneous frequency is derived from the derivative of the signal's
    unwrapped phase, so this works on any complex (I/Q) signal, e.g. a
    transmitted chirp or a received signal, without needing to know its sweep
    parameters. Pass the same `ax` for multiple signals (e.g. tx and rx) to
    overlay them on one plot for comparison.

    Args:
        signal: 1D complex signal.
        sample_rate: sampling rate in Hz.
        title: plot title.
        label: legend label for this signal's curve; omit to skip the legend.
        freq_offset: Hz added to the (baseband) instantaneous frequency before
            plotting -- pass a step's RF0.freq here to place multiple
            subchirps at their true absolute RF frequency on one shared axis.
        color: matplotlib color for this curve; omit to let matplotlib cycle.
        amplitude_threshold: fraction of the signal's peak amplitude below
            which a sample is treated as noise floor rather than signal.
            Phase-derivative frequency estimation is meaningless where
            amplitude is near zero (the phase is essentially random), and
            those samples produce spurious frequency spikes that can swamp
            an axis shared with another trace (e.g. a low-SNR RX capture
            drowning out a much cleaner TX chirp). Estimates touching a
            below-threshold sample are excluded (plotted as gaps) instead.
    """
    phase = np.unwrap(np.angle(signal))
    inst_freq = np.diff(phase) / (2.0 * np.pi) * sample_rate  # Hz
    time = np.arange(len(inst_freq)) / sample_rate * 1e6  # us

    amplitude = np.abs(signal)
    peak = amplitude.max()
    if peak > 0:
        # inst_freq[i] is derived from samples i and i+1, so drop it if either
        # contributing sample is too weak to trust its phase.
        low_amp = amplitude < (amplitude_threshold * peak)
        inst_freq = np.where(low_amp[:-1] | low_amp[1:], np.nan, inst_freq)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(time, (inst_freq + freq_offset) / 1e6, label=label, color=color)  # MHz
    ax.set_title(title)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Frequency (MHz)")
    ax.grid()
    if label is not None:
        ax.legend()
    if show:
        plt.show()

def stack(rx, n):
    """
    Coherently stack received signals.

    Args:
        rx: Received signals (2D array).
        n: Number of signals to stack.
    """
    return processing.stack(rx, n)

def plot_raw_matched(matched, sample_rate, tx, ax=None, show=True, label=None, color=None):
    # on x,y plot
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(20*np.log10(np.abs(np.sum(matched, axis=1))), label=label, color=color)
    ax.set_title("Raw Matched Filter Output")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Normalized Amplitude (dB)")
    ax.grid()
    if show:
        plt.show()

def plot_matched(matched, sample_rate, tx=10000, velocity_factor=2/3, loopback=True, coax_length=None, zero_sample_idx=158.87, ax=None, show=True, label=None, color=None, annotate=True):
    """Plot matched filter output with physical distance scaling.

    Args:
        matched: 2D array (samples x pulses) after matched filtering.
        sample_rate: sampling rate in Hz.
        tx: optional transmit chirp (1D array). If provided, its length is
            used to compute the delay offset. If not, a fallback is used.
        velocity_factor: fraction of light speed inside the medium (coax),
            default 2/3 for typical coax.
        loopback: True if the signal traverses the medium one-way (tx->rx).
            For free-space reflections set False (two-way and divide by 2).
        coax_length: optional expected coax length in meters; if provided,
            a vertical line and annotation for expected return is drawn.
    """
    c0 = 299792458.0  # speed of light in vacuum (m/s)

    range_profile = np.sum(matched, axis=1) # coherent sum across matched axis
    magnitude = np.abs(range_profile)

    # Use the provided chirp length if available, otherwise fall back to 1120
    chirp_len = len(tx) if tx is not None else 1120
    delay_offset = chirp_len - 1 + zero_sample_idx # correct for the delay caused by the matched filter, 159 comes from hardware latency and is proprotional to sampling rate (taken from Loopback test jupyter notebook)

    n = np.arange(len(magnitude))
    # time per sample converted from sample index difference
    time = (n - delay_offset) / float(sample_rate)

    v = velocity_factor * c0
    if loopback:
        ranges = time * v
    else:
        ranges = time * v / 2.0

    v = 2/3 * 299792458
    #Change this variable if your cable isn't 50 meters
    cable_length = 0.5
    expected_idx = int(cable_length / v * sample_rate + len(tx) - 1)
    print("Expected sample index for", cable_length, "m:", expected_idx)

    mask = ranges >= 0
    magnitude = magnitude[mask]
    ranges = ranges[mask]
    magnitude = 20 * np.log10(magnitude)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(ranges, magnitude, label=label, color=color)
    ax.set_title("Matched Filter")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid()

    # If expected physical length is provided, mark it on the plot
    if coax_length is not None and annotate:
        expected_time = coax_length / v
        expected_sample = expected_time * sample_rate + delay_offset
        expected_range = coax_length
        ax.axvline(expected_range, color='r', linestyle='--', label=f"expected {coax_length} m")

        # Diagnostics: find the received peak and suggest a velocity_factor correction
        peak_idx = np.argmax(magnitude)
        measured_range = ranges[peak_idx]

        # Print diagnostics to console so you can see numeric mismatch
        print(f"Matched peak index: {peak_idx}")
        print(f"Measured range at peak: {measured_range:.3f} m")
        print(f"Expected coax length: {coax_length} m")

        # Annotate peak and suggested correction on the plot
        ax.plot([measured_range], [magnitude[peak_idx]], 'ko', label=f'measured peak: {measured_range:.1f} m')

    if label is not None or (coax_length is not None and annotate):
        ax.legend()

    if show:
        plt.show()

def compress(stacked, chirp, sample_rate):
    fast_time, x = processing.pulse_compress(stacked, chirp, sample_rate)
    return fast_time, x

def main(prefix, combine=False, axs=None):
    #prefix = "../../data/20260219_202834"
    #prefix = "../../data/20260218_233213"
    slowtime, sample_rate, rx = load_data(prefix)
    _, chirp = generate_chirp(processing.load_config(prefix))
    #plot_chirp(chirp, sample_rate)
    stacked = stack(rx, rx.shape[1]) # initial stack to reduce size of rx
    matched = matched_filter(stacked, chirp)

    if combine:
        # Draw all four plots into one figure/window instead of four separate ones.
        own_fig = axs is None
        if own_fig:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        ax_matched, ax_raw, ax_compress, ax_freq = axs
        #Change your coax_length to the length of your cable
        plot_matched(matched, sample_rate, tx=chirp, velocity_factor=2/3, loopback=True, coax_length=0.5, ax=ax_matched, show=False)
        plot_raw_matched(matched, sample_rate, tx=chirp, ax=ax_raw, show=False)
        plot_compress(rx, chirp, sample_rate, ax=ax_compress, show=False)
        plot_freq_vs_time(chirp, sample_rate, title="Frequency vs Time", label="Tx Chirp", ax=ax_freq, show=False)
        plot_freq_vs_time(stacked[:, 0], sample_rate, label="Rx Signal", ax=ax_freq, show=False)
        if own_fig:
            fig.tight_layout()
            plt.show()
    else:
        #Change your coax_length to the length of your cable
        plot_matched(matched, sample_rate, tx=chirp, velocity_factor=2/3, loopback=True, coax_length=0.5)
        plot_raw_matched(matched, sample_rate, tx=chirp)
        plot_compress(rx, chirp, sample_rate)
        _, ax_freq = plt.subplots()
        plot_freq_vs_time(chirp, sample_rate, title="Frequency vs Time", label="Tx Chirp", ax=ax_freq, show=False)
        plot_freq_vs_time(stacked[:, 0], sample_rate, label="Rx Signal", ax=ax_freq, show=True)
        plot_freq_vs_time(chirp, sample_rate, title="Tx Chirp Frequency vs Time")

def main_multi(prefixes, axs=None, labels=None, coax_length=0.5):
    """
    Like main(), but overlays results from multiple runs (e.g. the subchirp
    steps of a stepped-frequency acquisition -- see run.py --stepped) onto one
    shared set of plots, so the tiled subbands can be visually compared.

    Args:
        prefixes: list of data file prefixes (one per step), e.g. sorted by
            each step's RF0.freq.
        labels: optional list of legend labels, one per prefix. Defaults to
            each step's RF0.freq in MHz, read from that step's own archived
            "_config.yaml".
        coax_length: expected coax length in meters (see plot_matched); only
            annotated once (for the first step) to avoid a duplicated marker
            per step cluttering the range-profile legend.
    """
    if labels is None:
        labels = [f"{processing.load_config(p)['RF0']['freq']/1e6:.1f} MHz" for p in prefixes]

    own_fig = axs is None
    if own_fig:
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    ax_matched, ax_raw, ax_compress, ax_freq = axs

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, (prefix, label) in enumerate(zip(prefixes, labels)):
        color = colors[i % len(colors)]
        config = processing.load_config(prefix)
        rf_freq = config['RF0']['freq']

        slowtime, sample_rate, rx = load_data(prefix)
        _, chirp = generate_chirp(config)
        stacked = stack(rx, rx.shape[1])
        matched = matched_filter(stacked, chirp)

        plot_matched(matched, sample_rate, tx=chirp, velocity_factor=2/3, loopback=True,
                     coax_length=coax_length, ax=ax_matched, show=False,
                     label=label, color=color, annotate=(i == 0))
        plot_raw_matched(matched, sample_rate, tx=chirp, ax=ax_raw, show=False,
                          label=label, color=color)
        plot_compress(rx, chirp, sample_rate, ax=ax_compress, show=False,
                      label=label, color=color)
        plot_freq_vs_time(chirp, sample_rate, title="Frequency vs Time (absolute RF)",
                           label=f"Tx {label}", ax=ax_freq, show=False,
                           freq_offset=rf_freq, color=color)
        plot_freq_vs_time(stacked[:, 0], sample_rate, label=f"Rx {label}", ax=ax_freq,
                           show=False, freq_offset=rf_freq, color=color)

    ax_matched.legend()
    ax_raw.legend()
    ax_compress.legend()
    ax_freq.legend()

    if own_fig:
        fig.tight_layout()
        plt.show()

    return axs
