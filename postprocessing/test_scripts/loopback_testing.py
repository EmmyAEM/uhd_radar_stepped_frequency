import numpy as np
import sys
import math
sys.path.append("postprocessing")
import processing
import matplotlib
matplotlib.use("TkAgg")  # avoid Qt/Wayland plugin issues
import matplotlib.pyplot as plt
import os
import scipy.signal as sp
sys.path.append("preprocessing")
from generate_chirp import generate_chirp


def load_data(prefix, max_seconds_to_load=60*100):
    """
    Load radar data from the specified file prefix.

    max_seconds_to_load caps how much of a (potentially many-GB) capture
    actually gets read into memory -- see processing.load_radar_data() for
    the exact formula, but in short: pulses_per_second is floored to at
    least 1, so for a slow pulse_rep_int (e.g. > 1s, as with a small
    num_pulses/large-rx_duration ApRES-style config), max_seconds_to_load
    effectively caps the number of *pulses* loaded, not real seconds. Lower
    it (e.g. to 10-20) if a full capture doesn't fit in available RAM.

    Returns (slowtime, sample_rate, rx, start_timestamp), where
    start_timestamp is the wall-clock Unix time (seconds) this run's radar
    program actually started transmitting/receiving (or None if it couldn't
    be found in the saved log).
    """
    return processing.load_radar_data(prefix, max_seconds_to_load=max_seconds_to_load)

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

def plot_freq_vs_time(signal, sample_rate, title="Frequency vs Time", label=None, ax=None, show=True, freq_offset=0.0, time_offset=0.0, color=None, amplitude_threshold=0.1):
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
        time_offset: us added to the time axis before plotting -- pass a
            step's real acquisition start time (relative to the first step)
            here to lay multiple subchirps out side-by-side in the order/gaps
            they were actually sent, instead of all starting at time 0.
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
    time = np.arange(len(inst_freq)) / sample_rate * 1e6 + time_offset  # us

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

def plot_fft_phase(signal, sample_rate, title="Phase of FFT Coefficients", label=None, ax=None, show=True, color=None):
    """
    Plot the complex phase of the Fourier coefficients of a signal.

    Args:
        signal: 1D complex signal (e.g. a single pulse/column of rx, or a tx
            chirp).
        sample_rate: sampling rate in Hz.
        title: plot title.
        label: legend label for this signal's curve; omit to skip the legend.
        color: matplotlib color for this curve; omit to let matplotlib cycle.
    """
    n = len(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate))
    phase = np.angle(spectrum)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(freqs / 1e6, phase, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Phase (radians)")
    ax.grid()
    if label is not None:
        ax.legend()
    if show:
        plt.show()

def plot_fft_magnitude(signal, sample_rate, title="Magnitude of FFT Coefficients", label=None, ax=None, show=True, color=None, freq_offset=0.0, amplitude_threshold=None, db=False):
    """
    Plot the absolute value (magnitude) of the Fourier coefficients of a signal.

    Args:
        signal: 1D complex signal (e.g. a single pulse/column of rx, or a tx
            chirp).
        sample_rate: sampling rate in Hz.
        title: plot title.
        label: legend label for this signal's curve; omit to skip the legend.
        color: matplotlib color for this curve; omit to let matplotlib cycle.
        freq_offset: Hz added to the (baseband) FFT bin frequencies before
            plotting -- pass a step's RF0.freq here to place a subchirp's
            spectrum at its true absolute RF frequency.
        amplitude_threshold: fraction of the spectrum's peak magnitude below
            which a bin is treated as noise floor and excluded (plotted as a
            gap) rather than drawn -- without this, bins outside the actual
            chirp bandwidth are essentially noise and can make the plot look
            like meaningless clutter. Omit (None) to plot every bin.
        db: if True, plot magnitude in dB (20*log10) instead of linear --
            usually much easier to read since a chirp's in-band energy can be
            orders of magnitude above the noise floor.
    """
    n = len(signal)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate)) + freq_offset
    magnitude = np.abs(spectrum)

    if amplitude_threshold is not None:
        peak = magnitude.max()
        if peak > 0:
            magnitude = np.where(magnitude < amplitude_threshold * peak, np.nan, magnitude)

    y = 20 * np.log10(magnitude) if db else magnitude

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(freqs / 1e6, y, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude (dB)" if db else "Magnitude")
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

def plot_fmcw_range(rx, tx_ref, sample_rate, chirp_bandwidth, chirp_duration,
                     velocity_factor=2/3, loopback=True, coax_length=None,
                     range_offset=0.0, ax=None, show=True, label=None,
                     color=None, annotate=True):
    """Plot an FMCW/deramp range profile (see processing.fmcw_range_profile()).

    Cheap alternative to plot_matched()'s pulse-compression approach for long
    continuous chirps (e.g. ApRES-style stepped-frequency captures), where
    pulse compression's full-length convolution is prohibitively expensive.
    Same coax_length calibration/annotation pattern as plot_matched().

    range_offset: see processing.fmcw_range_profile() -- constant meters
        subtracted from every range to correct for fixed hardware delay.
        Tune this until the measured/expected peaks below line up; the
        printed diagnostics show how far off they currently are.
    """
    ranges, magnitude_db = processing.fmcw_range_profile(
        rx, tx_ref, sample_rate, chirp_bandwidth, chirp_duration,
        velocity_factor=velocity_factor, loopback=loopback, window=np.blackman,
        range_offset=range_offset)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(ranges, magnitude_db, label=label, color=color)
    ax.set_title("FMCW Range Profile (Deramp)")
    ax.set_xlabel("Distance (meters)")
    ax.set_ylabel("Normalized Amplitude (dB)")
    ax.grid()

    if coax_length is not None and annotate:
        ax.axvline(coax_length, color='r', linestyle='--', label=f"expected {coax_length} m")

        peak_idx = np.argmax(magnitude_db)
        measured_range = ranges[peak_idx]

        print(f"Measured range at peak: {measured_range:.3f} m")
        print(f"Expected coax length: {coax_length} m")

        ax.plot([measured_range], [magnitude_db[peak_idx]], 'ko', label=f'measured peak: {measured_range:.1f} m')

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
    slowtime, sample_rate, rx, _ = load_data(prefix)
    _, chirp = generate_chirp(processing.load_config(prefix))
    #plot_chirp(chirp, sample_rate)
    stacked = stack(rx, rx.shape[1]) # initial stack to reduce size of rx
    matched = matched_filter(stacked, chirp)

    if combine:
        # Draw all six plots into one figure/window instead of six separate ones.
        own_fig = axs is None
        if own_fig:
            fig, axs = plt.subplots(2, 3, figsize=(20, 10))
            axs = axs.flatten()
        ax_matched, ax_raw, ax_compress, ax_freq, ax_fft_mag, ax_fft_phase = axs
        #Change your coax_length to the length of your cable
        plot_matched(matched, sample_rate, tx=chirp, velocity_factor=2/3, loopback=True, coax_length=0.5, ax=ax_matched, show=False)
        plot_raw_matched(matched, sample_rate, tx=chirp, ax=ax_raw, show=False)
        plot_compress(rx, chirp, sample_rate, ax=ax_compress, show=False)
        plot_freq_vs_time(chirp, sample_rate, title="Frequency vs Time", label="Tx Chirp", ax=ax_freq, show=False)
        plot_freq_vs_time(stacked[:, 0], sample_rate, label="Rx Signal", ax=ax_freq, show=False)
        plot_fft_magnitude(stacked[:, 0], sample_rate, title="Magnitude of Rx FFT Coefficients", label="Rx Signal", ax=ax_fft_mag, show=False)
        plot_fft_phase(stacked[:, 0], sample_rate, title="Phase of Rx FFT Coefficients", label="Rx Signal", ax=ax_fft_phase, show=False)
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
        plot_fft_magnitude(stacked[:, 0], sample_rate, title="Magnitude of Rx FFT Coefficients", label="Rx Signal")
        plot_fft_phase(stacked[:, 0], sample_rate, title="Phase of Rx FFT Coefficients", label="Rx Signal")

def _show_figures_sequentially(figs):
    """
    Show each already-populated figure in turn, blocking until the user closes
    it before opening the next one.

    plt.show() instead shows every currently-registered figure at once and
    blocks until *all* of them are closed -- fine for a single combined
    figure, but not when we've built several separate ones (e.g. main_multi's
    range/freq/phase plots with combine=False) and want to inspect/zoom one
    at a time without the others cluttering the screen.
    """
    for fig in figs:
        fig.show()
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)


def _fft_grid_shape(n):
    """Return (rows, cols) for a roughly-square grid holding n subplots."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def main_multi(prefixes, axs=None, labels=None, coax_length=0.5, total_span=1.0,
               range_offset=0.0, combine=False):
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
        coax_length: expected coax length in meters (see plot_fmcw_range); only
            annotated once (for the first step) to avoid a duplicated marker
            per step cluttering the range-profile legend.
        total_span: if given (default 1.0), the freq-vs-time x-axis is
            linearly compressed so the last step's start lands at
            `total_span` seconds instead of its true real-world offset (e.g.
            the default spans the whole sequence over 1 second). Relative
            spacing between steps' real start times is preserved (a step
            reached twice as long after its predecessor as another pair is
            still shown twice as far apart), just scaled down uniformly --
            this is what makes each ~20us chirp ramp visible at all, since
            the true inter-step gaps (tens of seconds, from process restart +
            LO retune) would otherwise squeeze every ramp down to much less
            than a pixel. Pass None to plot at true real-world scale instead
            (in which case each ramp will visually collapse to a vertical
            sliver -- useful for confirming actual acquisition timing, not
            for seeing chirp shape).
        range_offset: forwarded to plot_fmcw_range()/fmcw_range_profile() --
            constant meters subtracted from every computed range to correct
            for fixed hardware delay. Tune until the measured peak lines up
            with coax_length.
        combine: if True, lay the range/freq plots and a grid of per-step FFT
            magnitude plots side by side in one shared window. If False
            (default), give the range plot, freq plot, and FFT grid each
            their own window and show them one at a time -- each window
            blocks until you close it -- so you can zoom into one plot
            without the others open and cluttering the screen.
    """
    if labels is None:
        labels = [f"{processing.load_config(p)['RF0']['freq']/1e6:.1f} MHz" for p in prefixes]

    n_steps = len(prefixes)
    rows, cols = _fft_grid_shape(n_steps)

    own_fig = axs is None
    if own_fig:
        if combine:
            fig = plt.figure(figsize=(20, 5 * rows))
            gs = fig.add_gridspec(rows, 2 + cols)
            ax_range = fig.add_subplot(gs[:, 0])
            ax_freq = fig.add_subplot(gs[:, 1])
            fft_axes = [fig.add_subplot(gs[r, 2 + c])
                        for r in range(rows) for c in range(cols)][:n_steps]
        else:
            fig_range, ax_range = plt.subplots(figsize=(7, 5))
            fig_freq, ax_freq = plt.subplots(figsize=(7, 5))
            fig_fft, fft_axes_grid = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
            fft_axes_flat = fft_axes_grid.flatten()
            fft_axes = list(fft_axes_flat[:n_steps])
            for extra_ax in fft_axes_flat[n_steps:]:
                extra_ax.axis("off")
        axs = (ax_range, ax_freq, fft_axes)
    ax_range, ax_freq, fft_axes = axs

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Real wall-clock start time of each step (read from each step's small
    # log file only -- NOT the raw sample .bin, which can be many GB and
    # must not all be loaded into memory at once), so the freq-vs-time plot
    # lays steps out in the order/gaps they were actually sent (including
    # the oscillator retune + process restart time between steps) instead of
    # overlaying every step starting at time 0.
    start_times = [processing.get_start_timestamp(p) for p in prefixes]
    known_start_times = [t for t in start_times if t is not None]
    if len(known_start_times) < len(prefixes):
        print("WARNING: Couldn't find a start timestamp for one or more steps "
              "(missing/incomplete log); those steps will be plotted with no time offset.")
    t0 = min(known_start_times) if known_start_times else 0.0
    time_offsets_us = [
        (t - t0) * 1e6 if t is not None else 0.0
        for t in start_times
    ]

    if total_span is not None:
        true_span_us = max(time_offsets_us) if time_offsets_us else 0.0
        print(f"True acquisition span across {len(prefixes)} steps: {true_span_us/1e6:.3f} s "
              f"-- compressing to {total_span:.3f} s for display.")
        if true_span_us > 0:
            scale = (total_span * 1e6) / true_span_us
            time_offsets_us = [t * scale for t in time_offsets_us]

    for i, (prefix, label) in enumerate(zip(prefixes, labels)):
        color = colors[i % len(colors)]
        config = processing.load_config(prefix)
        rf_freq = config['RF0']['freq']
        time_offset = time_offsets_us[i]

        # Stream through one step's raw capture, accumulating a running
        # average one pulse at a time, rather than loading the whole
        # (potentially many-GB) capture into memory before averaging it.
        # Peak memory is ~one pulse's size regardless of pulse count.
        averaged_pulse, sample_rate, n_pulses_used = processing.stream_average_pulses(prefix)
        _, chirp = generate_chirp(config)
        print(f"{label}: averaged {n_pulses_used} pulses")

        # Deramp/FFT range profile -- cheap (O(N) multiply + one FFT), unlike
        # pulse-compression cross-correlation, which needs FFTs padded to
        # ~2x the signal length and is prohibitively expensive (GB-scale
        # memory) for these long continuous chirps rather than short pulses.
        plot_fmcw_range(averaged_pulse, chirp, sample_rate,
                         chirp_bandwidth=config['GENERATE']['chirp_bandwidth'],
                         chirp_duration=config['GENERATE']['chirp_length'],
                         velocity_factor=2/3, loopback=True,
                         coax_length=coax_length, range_offset=range_offset,
                         ax=ax_range, show=False,
                         label=label, color=color, annotate=(i == 0))
        freq_time_title = ("Frequency vs Time (absolute RF, compressed acquisition time)"
                           if total_span is not None else
                           "Frequency vs Time (absolute RF, real acquisition time)")
        plot_freq_vs_time(chirp, sample_rate, title=freq_time_title,
                           label=f"Tx {label}", ax=ax_freq, show=False,
                           freq_offset=rf_freq, time_offset=time_offset, color=color)
        plot_freq_vs_time(averaged_pulse, sample_rate, label=f"Rx {label}", ax=ax_freq,
                           show=False, freq_offset=rf_freq, time_offset=time_offset, color=color)
        # One subplot per step instead of overlaying every step's spectrum on
        # a shared axis -- out-of-band bins are noise (phase there is
        # essentially random), so overlaying N steps just produced visual
        # clutter. amplitude_threshold masks that noise floor out and db=True
        # keeps the in-band peak legible against it.
        plot_fft_magnitude(averaged_pulse, sample_rate, title=f"{label} Rx FFT Magnitude",
                            ax=fft_axes[i], show=False, color=color,
                            freq_offset=rf_freq, amplitude_threshold=0.05, db=True)

    if total_span is not None:
        ax_freq.set_xlabel(f"Time (us, compressed -- true span {true_span_us/1e6:.3f} s)")

    ax_range.legend(loc="upper right")
    ax_freq.legend(loc="upper right")

    if own_fig:
        if combine:
            fig.tight_layout()
            plt.show()
        else:
            for f in (fig_range, fig_freq, fig_fft):
                f.tight_layout()
            _show_figures_sequentially([fig_range, fig_freq, fig_fft])

    return axs