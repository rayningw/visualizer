import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# Audio wave visualiser
# Adapted from: https://www.youtube.com/watch?v=AShHJdSIxkY
# Also referenced: https://realpython.com/python-scipy-fft/

# Constants
CHUNK = 1024 * 4             # samples per buffer (read)
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
x_fft = rfftfreq(CHUNK, 1 / RATE)

# Initial data
line_wave, = ax_wave.plot(x_wave, np.random.rand(len(x_wave)), "-", lw=2)
line_fft, = ax_fft.semilogx(x_fft, np.random.rand(len(x_fft)), "-", lw=2)

# Set axes sizes
ax_wave.set_ylim(-2**15, 2**15)
# NOTE(ray): Thought that it should be 2**15, same magnitude as input amplitude
# scale, but it was too big. So just chose a scale that looked right.
ax_fft.set_ylim(0, 2**10)

while True:
    # Read binary data
    data = stream.read(CHUNK, exception_on_overflow=False)

    # Convert into numpy array
    data_np = np.frombuffer(data, dtype=np.int16)

    # Update waveform graph
    line_wave.set_ydata(data_np)

    # Compute FFT
    # Use real FFT because input is real numbers
    y_fft = rfft(data_np)

    # Compute the magnitude of the values
    # Normalize by the size of the buffer
    # NOTE(ray): In the video apparently we're meant to multiply by 2 first (?)
    # NOTE(ray): In some other tutorials we're meant to square it (?)
    y_transformed = np.abs(y_fft) / CHUNK
    line_fft.set_ydata(y_transformed)

    # Yield to update
    plt.pause(0.001)