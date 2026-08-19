# This Program visuaizes the chirps generated without having to run the SDR model.

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import find_peaks # not yet used
from numpy import random #not yet used

##Original parameters from the yaml
chirp_type = 'linear'
sample_rate = 56e6 #[Hz] Sample rate of the SDR
chirp_bandwidth = 25e6 #[Hz] Bandwidth of the chirp
chirp_length = 20e-6 #[s] Length of the chirp
offset = 0e6 #[Hz] LO offset for the chirp
window = 'rectangular' #windowing function for the chirp
pulse_length = 20e-6 #[s] Length of the pulse, default to chirp_length if not specified
end_freq = chirp_bandwidth / 2 # Chirp goes from -BW/2 to BW/2

#chirp generation with'rectangular' windowing function
start_freq = -1 * end_freq
start_freq += offset
end_freq += offset
ts = np.arange(0, chirp_length-(1/(2*sample_rate)), 1/(sample_rate))
ts_zp = np.arange(0, (pulse_length)-(1/(2*sample_rate)), 1/(sample_rate))
ph = 2*np.pi*((start_freq)*ts + (end_freq - start_freq) * ts**2 / (2*chirp_length))
chirp_complex = np.exp(1j*ph)
chirp_complex = np.pad(chirp_complex, (int(np.floor(ts_zp.size - ts.size)/2),), 'constant')
chirp_complex = chirp_complex


fig, axs = plt.subplots(2,1)
# Time domain plot
axs[0].plot(ts*1e6, np.real(chirp_complex), label='I')
axs[0].plot(ts*1e6, np.imag(chirp_complex), label='Q')
axs[0].set_xlabel('Time [us]')
axs[0].set_ylabel('Samples')
axs[0].set_title('Time Domain')
axs[0].legend()

# Frequency domain plot
freqs = scipy.fft.fftshift(scipy.fft.fftfreq(chirp_complex.size, d=1/sample_rate))
ms = 20*np.log10(scipy.fft.fftshift(np.abs(scipy.fft.fft(chirp_complex))))
axs[1].plot(freqs/1e6, ms)
axs[1].set_xlabel('Frequency [MHz]')
axs[1].set_ylabel('Amplitude [dB]')
axs[1].set_title('Frequency Domain')
axs[1].grid()
fig.tight_layout()
plt.show()