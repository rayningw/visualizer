from math import log
from scipy.fft import rfft
import numba
import numpy as np
import pyaudio
import pygame as pg

# Audio constants
CHUNK = 1024 * 1             # frames per buffer (read)
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

HIGH_POINT = 2**12           # arbitrary high point
DECAY_RATE = 1               # decay rate of previous volume (per millisecond) - set to 1 to decay completely
MAX_SPEEDUP = 10             # max speedup factor

# texture
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture).astype(dtype=np.uint32)

x_start, y_start = -2, -2  # an interesting region starts here
space_width, space_height = 4, 4  # for 4 units up and right
density_per_unit = 128  # how many pixels per unit

# resolution
res = res_width, res_height = space_width * density_per_unit, space_height * density_per_unit

# max allowed iterations
threshold = 32

# number of segments in revolution
num_segments = 360

# time for one revolution
revolution_millis = 10000
millis_per_segment = revolution_millis / num_segments

# real and imaginary axis
re = np.linspace(x_start, x_start + space_width, space_width * density_per_unit)
im = np.linspace(y_start, y_start + space_height, space_height * density_per_unit)

# we represent c as c = r*cos(a) + i*r*sin(a) = r*e^{i*a}
#radius = 0.7885
low_volume_radius = 1.5
high_volume_radius = 0.5
angles = np.linspace(0, 2*np.pi, num_segments)

# Renders a Julia set with C revolving around the origin
class JuliaRenderer:
    def __init__(self, screen, screen_array):
        self.screen = screen
        self.screen_array = screen_array

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(radius_ratio, segment_index, screen_array, smooth=False):
        # Compute the C number
        radius_delta = (low_volume_radius - high_volume_radius) * radius_ratio
        radius = low_volume_radius - radius_delta
        print("radius_ratio:", radius_ratio)
        print("radius:", radius)
        cx, cy = radius * np.cos(angles[segment_index]), radius * np.sin(angles[segment_index])
        c = complex(cx, cy)
        
        # iterations for the given threshold
        for i in numba.prange(len(re)):
            for j in range(len(im)):
                # Compute Julia set membership
                z = complex(re[i], im[j])
                num_iterations = 0
                for _i in range(threshold):
                    z = z**2 + c
                    if z.real ** 2 + z.imag ** 2 > 4:
                        break
                    num_iterations += 1

                if smooth:
                    # Fractional escape count:
                    # https://realpython.com/mandelbrot-set-python/#smoothing-out-the-banding-artifacts
                    # NOTE(ray): Don't know how to adapt speedy magnitude check here
                    escape_count = num_iterations + 1 - log(log(abs(z))) / log(2)
                    unbounded_escape_ratio = escape_count / threshold
                    escape_ratio = max(0.0, min(unbounded_escape_ratio, 1.0))
                else:
                    escape_ratio = num_iterations / threshold

                # Colour based on escape ratio
                col = int(texture_size * escape_ratio)
                screen_array[i, j] = texture_array[col, col]
        
        return screen_array

    def animate(self, radius_ratio, tick):
        segment_index = int(tick / millis_per_segment) % num_segments
        self.render(radius_ratio, segment_index, self.screen_array, smooth=True)
        pg.surfarray.blit_array(self.screen, self.screen_array)

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.screen_array = np.full((res_width, res_height, 3), [0, 0, 0], dtype=np.uint8)
        self.julia = JuliaRenderer(self.screen, self.screen_array)

        # pyaudio class instance
        self.audio = pyaudio.PyAudio()

        # Stream object to get data from microphone
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            output=True,
            frames_per_buffer=CHUNK
        )
    
    @staticmethod
    def get_frequency_data(data):
        # Convert into numpy array
        data_np = np.frombuffer(data, dtype=np.int16)

        # Compute FFT (use real-FFT because input is real numbers)
        data_fft = rfft(data_np)

        # Compute the magnitude of the values
        # Normalize by the size of the buffer
        # NOTE(ray): In the video apparently we're meant to multiply by 2 first (?)
        # NOTE(ray): In some other tutorials we're meant to square it (?)
        data_transformed = np.abs(data_fft) / CHUNK
        return data_transformed

    def read_audio(self):
        # Read binary data
        # NOTE(ray): `exception_on_overflow` flag is needed to suppress exception -
        # Unsure why it happens, maybe because we're not reading fast enough
        data = self.stream.read(CHUNK, exception_on_overflow=False)

        frequency_data = self.get_frequency_data(data)
        return frequency_data

    def run(self):
        tick = 0
        prev_audio_volume = 0
        while True:
            # Determine FPS
            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            millis_elapsed = self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')

            # Read audio data
            audio_data = self.read_audio()

            # Get current audio volume
            cur_audio_volume = np.sum(audio_data)

            # Decay previous audio volume
            decayed_audio_volume = prev_audio_volume * (1 - DECAY_RATE)**millis_elapsed
            total_audio_volume = decayed_audio_volume + cur_audio_volume

            # Copy over total audio volume
            prev_audio_volume = total_audio_volume

            # Calc volume ratio against arbitrary high point
            volume_ratio = min(total_audio_volume / HIGH_POINT, 1)

            # Progress tick with a speedup
            tick_increment = millis_elapsed * (1 + volume_ratio * MAX_SPEEDUP)
            tick += tick_increment

            print("*******")
            print("millis_elapsed:", millis_elapsed)
            print("decayed_audio_volume:", decayed_audio_volume)
            print("total_audio_volume:", total_audio_volume)
            print("volume_ratio:", volume_ratio)
            print("tick_increment:", tick_increment)
            print("tick:", tick)

            # Paint Julia set
            self.screen.fill('black')
            self.julia.animate(volume_ratio, tick)
            pg.display.flip()

if __name__ == '__main__':
    app = App()
    app.run()
