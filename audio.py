import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Audio wave visualiser
# Adapted from: https://www.youtube.com/watch?v=AShHJdSIxkY

# Constants
CHUNK = 1024 * 4             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

# pyaudio class instance
p = pyaudio.PyAudio()

# Stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Create plot
fig, (ax_wave, ax_fft) = plt.subplots(2, figsize=(15, 8))
x_wave = np.arange(0, CHUNK)
x_fft = np.linspace(0, RATE, CHUNK)

# Initial data
line_wave, = ax_wave.plot(x_wave, np.random.rand(CHUNK), "-", lw=2)
line_fft, = ax_fft.semilogx(x_fft, np.random.rand(CHUNK), "-", lw=2)

# Set axes sizes
ax_wave.set_ylim(-2**15, 2**15)
ax_wave.set_xlim(0, CHUNK)
ax_fft.set_xlim(0, RATE / 2)

while True:
    # Read binary data
    data = stream.read(CHUNK)  

    # Convert into numpy array
    data_np = np.frombuffer(data, dtype=np.int16)

    # Update waveform graph
    line_wave.set_ydata(data_np)

    # Update fft graph
    y_fft = fft(data_np)
    # NOTE(ray): Don't understand why we multiply by 2 then divide by buckets
    # NOTE(ray): Chose scale that looked right. Tried 2**16 seemed too big for it.
    line_fft.set_ydata(np.abs(y_fft) * 2 / (2**9 * CHUNK))

    # Yield to update
    plt.pause(0.001)