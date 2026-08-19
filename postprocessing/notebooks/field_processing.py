"""
Field Processing.

Script version of Field Processing.ipynb. Loads raw radar data, cleans/stacks it,
pulse compresses it, and generates all the plots documented in the notebook
(raw pulse, 1D pulse-compressed trace, 2D radargram, spectrogram, and power
spectrum). All figures are shown together at the end via plt.show().
"""

import copy
import sys

import numpy as np
import scipy.constants
import scipy.signal
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
from dask.distributed import Client

sys.path.append("..")
import processing_dask as pr
import processing as old_processing

sys.path.append("../../preprocessing/")
from generate_chirp import generate_chirp


def main():
    plt.rcParams.update({'font.size': 16})

    # -----------------------------------------------------------------------
    # Open and resave file
    # -----------------------------------------------------------------------

    client = Client()  # Note that `memory_limit` is the limit **per worker**.
    print(client)  # If you open the dashboard link, you can monitor real-time progress.

    # file path to data and configs
    prefix = "/home/emmyalondraem/reusum2026/uhd_radar_code_2026/uhd_radar_2026/data/sdr_Roble_test/data_1/20260817_200922_step0_225MHz"  # Emmy's

    # resave data as zarr for dask processing
    zarr_base_location = "/home/emmyalondraem/reusum2026/uhd_radar_code_2026/test_tmp_zarr_cache"
    zarr_path = pr.save_radar_data_to_zarr(prefix, zarr_base_location=zarr_base_location)

    # open zarr file, adjust chunk size to be 10 MB - 1 GB based on sample rate/bit depth
    raw = xr.open_zarr(zarr_path, chunks={"pulse_idx": 1000})

    # -----------------------------------------------------------------------
    # Enter processing parameters
    # -----------------------------------------------------------------------

    #zero_sample_idx = 36 # X310, fs = 20 MHz
    #zero_sample_idx = 63 # X310, fs = 50 MHz
    zero_sample_idx = 159  # B205mini, fs = 56 MHz
    #zero_sample_idx = 166 # B205mini, fs = 20 MHz

    nstack = 1  # number of pulses to stack

    modify_rx_window = False  # set to true if you want to window the reference chirp only on receive, false uses ref chirp as transmitted in config file
    rx_window = "rectangular"  # what you want to change the rx window to if modify_rx_window is true

    #dielectric_constant = 3.17 # ice (air = 1, 66% velocity coax = 2.2957)
    dielectric_constant = 2.2957  # COAX (air = 1, 66% velocity coax = 2.2957)
    sig_speed = scipy.constants.c / np.sqrt(dielectric_constant)

    # -----------------------------------------------------------------------
    # Generate reference chirp
    # -----------------------------------------------------------------------

    if modify_rx_window:
        config = copy.deepcopy(raw.config)
        config['GENERATE']['window'] = rx_window
    else:
        config = raw.config

    chirp_ts, ref_chirp = generate_chirp(config)

    # -----------------------------------------------------------------------
    # View raw pulse in time domain to check for clipping
    # -----------------------------------------------------------------------

    single_pulse_raw = raw.radar_data[{'pulse_idx': 0}].compute()

    fig1, ax1 = plt.subplots(facecolor='white', figsize=(10, 6))
    ax1.plot(single_pulse_raw.fast_time, np.real(single_pulse_raw), color='red', label='Real')
    ax1.plot(single_pulse_raw.fast_time, np.imag(single_pulse_raw), label='Imag')
    ax1.set_xlabel('Fast Time (s)')
    ax1.set_ylabel('Raw Amplitude')
    ax1.set_title('Raw Pulse (Time Domain)')
    ax1.legend()
    ax1.grid()

    # -----------------------------------------------------------------------
    # Clean and stack data
    # -----------------------------------------------------------------------

    stacked = pr.fill_errors(raw, error_fill_value=0.0)  # fill receiver errors with 0s
    stacked = pr.stack(stacked, nstack)  # stack

    # -----------------------------------------------------------------------
    # Pulse compress data
    # -----------------------------------------------------------------------

    compressed = pr.pulse_compress(
        stacked, ref_chirp,
        fs=stacked.config['GENERATE']['sample_rate'],
        zero_sample_idx=0,
        signal_speed=sig_speed,
    )

    compressed_power = xr.apply_ufunc(
        lambda x: 20 * np.log10(np.abs(x)),
        compressed,
        dask="parallelized",
    )

    # -----------------------------------------------------------------------
    # View 1D pulse compressed data
    # -----------------------------------------------------------------------

    fig2, ax2 = plt.subplots(facecolor='white', figsize=(10, 6))
    ax2.plot(compressed_power.reflection_distance, compressed_power.radar_data[0, :], label='First Pulse')
    ax2.plot(compressed_power.reflection_distance, compressed_power.radar_data[-1, :], label='Last Pulse')
    ax2.set_xlabel('Reflection Distance (m)')
    ax2.set_ylabel('Return Power (dB)')
    ax2.set_title('1D Pulse Compressed Data')
    ax2.set_xlim(-50, 200)
    ax2.set_ylim(-120, -40)
    ax2.legend()
    ax2.grid()

    # -----------------------------------------------------------------------
    # View 2D pulse compressed data (radargram)
    # -----------------------------------------------------------------------

    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6), facecolor='white')

    p = ax3.pcolormesh(
        compressed_power.slow_time,
        compressed_power.reflection_distance,
        compressed_power.radar_data.transpose(),
        shading='auto', cmap='inferno',
    )
    ax3.invert_yaxis()
    clb = fig3.colorbar(p, ax=ax3)
    clb.set_label('Return Power (dB)')
    ax3.set_xlabel('Slow Time (s)')
    ax3.set_ylabel('Distance to Reflector (m)')
    ax3.set_title('2D Pulse Compressed Data (Radargram)')
    # relevant options: ax.set_ylim=(100,-50), ax.set_xlim=(0, 1), vmin=-90, vmax=40
    ax3.set_ylim(100, -50)

    # -----------------------------------------------------------------------
    # View spectrogram of stacked data
    # -----------------------------------------------------------------------

    inpt = raw
    num_presums = raw.attrs["config"]["CHIRP"]["num_presums"]

    n = 1
    normalize = True

    pulse = pr.stack(inpt, n)[{'pulse_idx': 0}]["radar_data"].to_numpy()

    f, t, S = scipy.signal.spectrogram(
        pulse,
        fs=raw.attrs["config"]["GENERATE"]["sample_rate"],
        window='flattop',
        nperseg=128,
        noverlap=64,
        scaling='density', mode='psd',
        return_onesided=False,
    )

    if normalize:
        S /= np.max(S)

    fig4, ax4 = plt.subplots(facecolor='white', figsize=(20, 6))
    freq_mhz = (np.fft.fftshift(f) + raw.attrs['config']['RF0']['freq']) / 1e6
    pcm = ax4.pcolormesh(t, freq_mhz, 10 * np.log10(np.abs(np.fft.fftshift(S, axes=0))), shading='nearest')
    clb = fig4.colorbar(pcm, ax=ax4)
    clb.set_label('Power [dB]')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Frequency [MHz]')
    ax4.text(
        0, 1.05, prefix.split("/")[-1] + "\n" + f"n_stack * num_presums = {n * num_presums}",
        horizontalalignment='left', verticalalignment='center', transform=ax4.transAxes,
        fontdict={'size': 12},
    )
    fig4.tight_layout()

    fig4.savefig(f"orca_paper/outputs/{raw.basename}_ft_spectrogram_n{n}.png", dpi=300)

    # -----------------------------------------------------------------------
    # View Power Spectrum of All Received Data
    # -----------------------------------------------------------------------

    single_stack = pr.stack(raw, raw.radar_data.shape[1])

    data_rx_fft = da.fft.fft(raw.radar_data, axis=0) / raw.radar_data.shape[0]
    stacked_fft = da.fft.fft(stacked.radar_data, axis=0) / stacked.radar_data.shape[0]
    full_fft = da.fft.fft(single_stack.radar_data, axis=0) / single_stack.radar_data.shape[0]

    data_rx_fft_pwr = 20 * da.log10(da.abs(data_rx_fft))
    stacked_fft_pwr = 20 * da.log10(da.abs(stacked_fft))
    full_fft_pwr = 20 * da.log10(da.abs(full_fft))

    fig5, ax5 = plt.subplots(facecolor='white', figsize=(10, 6))
    freqs = np.fft.fftshift(np.fft.fftfreq(data_rx_fft_pwr.shape[0], d=1 / raw.config['GENERATE']['sample_rate']))
    ax5.plot(freqs / 1e6, np.fft.fftshift(data_rx_fft_pwr[:, 0]), label='Single Pulse')
    ax5.plot(freqs / 1e6, np.fft.fftshift(stacked_fft_pwr[:, 0]), label='Single Stack')
    ax5.plot(freqs / 1e6, np.fft.fftshift(full_fft_pwr[:, 0]), label='Full File')
    ax5.set_xlabel('Frequency [MHz]')
    ax5.set_ylabel('Power [dB]')
    ax5.set_title('Spectrum -- Power')
    ax5.grid()
    ax5.legend()

    # -----------------------------------------------------------------------
    # Show all figures
    # -----------------------------------------------------------------------

    plt.show()

    client.close()


if __name__ == "__main__":
    main()
