import pyaudio
import numpy as np
import matplotlib.pyplot as plt

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

# Create graph
fig, ax = plt.subplots()
# Initial data
line, = ax.plot(np.random.randn(CHUNK))
# Set axes sizes
ax.set_ylim(-2**15, 2**15)
ax.set_xlim(0, CHUNK)

while True:
    # Read binary data
    data = stream.read(CHUNK)  
    # Convert into numpy array
    data_np = np.frombuffer(data, dtype=np.int16)
    # Update graph
    line.set_ydata(data_np)
    # Yield to update
    plt.pause(0.001)