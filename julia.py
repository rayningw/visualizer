from math import log
from scipy.fft import rfft
import numba
import numpy as np
import pyaudio
import pygame as pg

pg.init()

# Audio constants
CHUNK = 1024 * 1             # frames per buffer (read)
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second

VOLUME_HIGH_POINT = 2**12    # arbitrary volume high point to calculate ratios
DECAY_RATE = 1               # decay rate of previous volume (per millisecond) - set to 1 to decay completely

MID_FREQ = 400               # start of mid-range frequencies
HIGH_FREQ = 4000             # start of high-range frequencies

# FFT output granularity
# CHUNK / 2 is the length of the FFT output
# RATE / 2 is the maximum sampleable frequency
points_per_freq = (CHUNK / 2) / (RATE / 2)
freq_per_point = 1 / points_per_freq

# lo/mid/high frequency breakpoints
mid_freq_idx = int(points_per_freq * MID_FREQ)
high_freq_idx = int(points_per_freq * HIGH_FREQ)

# texture
texture = pg.image.load('img/texture.jpg')
texture_size_x = texture.get_size()[0]
texture_size_y = texture.get_size()[1]
texture_array = pg.surfarray.array3d(texture).astype(dtype=np.uint32)

# time taken to traverse all rows of the texture map
millis_per_texture_traversal = 1000

# colours
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)

# metric labels
metrics_start_x = 32
metrics_start_y = 32
metrics_font_size = 10
metrics_spacing = 4
metric_ratio_format = "{0:.0%}"
metric_int_format = "{:.0f}"

# rendering region
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
revolution_millis = 100000
millis_per_segment = revolution_millis / num_segments

# max speedup factor
max_speedup = 100

# real and imaginary axis
re = np.linspace(x_start, x_start + space_width, space_width * density_per_unit)
im = np.linspace(y_start, y_start + space_height, space_height * density_per_unit)

# we represent c as c = r*cos(a) + i*r*sin(a) = r*e^{i*a}
#radius = 0.7885
low_volume_radius = 1.3
high_volume_radius = 0.5
angles = np.linspace(0, 2*np.pi, num_segments)

# pixel width of each frequency graph bar
freq_bar_width = 8

# Renders a bar graph
class BarGraphRenderer:
    def __init__(self, screen, graph_start_x, graph_start_y, graph_height):
        self.screen = screen
        self.graph_start_x = graph_start_x
        self.graph_start_y = graph_start_y
        self.graph_height = graph_height

    def render(self, values):
        for i in range(len(values)):
            value = values[i]
            height = value * self.graph_height
            top = self.graph_start_y + (self.graph_height - height)
            left = self.graph_start_x + i * freq_bar_width
            pg.draw.rect(self.screen, white, pg.Rect(left, top, freq_bar_width, height))

# Renders a Julia set with C revolving around the origin
class JuliaRenderer:
    def __init__(self, screen, screen_array):
        self.screen = screen
        self.screen_array = screen_array

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(radius_ratio, segment_index, texture_row, screen_array, smooth=False):
        # Compute the C number
        radius_delta = (low_volume_radius - high_volume_radius) * radius_ratio
        radius = low_volume_radius - radius_delta
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
                col = int(texture_size_x * escape_ratio)
                screen_array[i, j] = texture_array[texture_row, col]
        
        return screen_array

    def animate(self, radius_ratio, tick):
        segment_index = int(tick / millis_per_segment) % num_segments
        texture_tick = int(tick / millis_per_texture_traversal)
        texture_row = (texture_tick % texture_size_y)

        # move back and forth along the texture rows to avoid a jump
        texture_tick_odd_cycle = (texture_tick // texture_size_y) % 2 == 1
        if (texture_tick_odd_cycle):
            texture_row = texture_size_y - texture_row

        self.render(radius_ratio, segment_index, texture_row, self.screen_array, smooth=True)
        pg.surfarray.blit_array(self.screen, self.screen_array)

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.font = pg.font.SysFont('monospace', metrics_font_size)
        self.clock = pg.time.Clock()
        self.screen_array = np.full((res_width, res_height, 3), [0, 0, 0], dtype=np.uint8)
        self.julia = JuliaRenderer(self.screen, self.screen_array)
        self.graph = BarGraphRenderer(self.screen, 32, 400, 80)

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
        prev_volume = 0
        while True:
            # Determine FPS
            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            millis_elapsed = self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')

            # Read audio data
            audio_data = self.read_audio()

            # Determine breakdown of lows, mids, and highs
            low_volume = np.sum(audio_data[0 : mid_freq_idx])
            mid_volume = np.sum(audio_data[mid_freq_idx : high_freq_idx])
            high_volume = np.sum(audio_data[high_freq_idx : ])
            cur_volume = low_volume + mid_volume + high_volume

            # Decay previous audio volume
            decayed_volume = prev_volume * (1 - DECAY_RATE)**millis_elapsed
            total_volume = decayed_volume + cur_volume

            # Copy over total audio volume
            prev_volume = total_volume

            # Calc volume ratios against high point
            total_volume_ratio = min(total_volume / VOLUME_HIGH_POINT, 1)
            low_volume_ratio = min(low_volume / (VOLUME_HIGH_POINT / 3), 1)
            radius_ratio = low_volume_ratio
            mid_volume_ratio = min(mid_volume / (VOLUME_HIGH_POINT / 3), 1)
            high_volume_ratio = min(high_volume / (VOLUME_HIGH_POINT / 3), 1)

            # Progress tick with a speedup
            speedup_factor = np.average([mid_volume_ratio, high_volume_ratio]) * max_speedup
            tick_increment = millis_elapsed * (1 + speedup_factor)
            tick += tick_increment

            print("*******")
            print("millis_elapsed:", millis_elapsed)
            print("cur_volume:", cur_volume)
            print("low_volume:", low_volume)
            print("low_volume_ratio:", low_volume_ratio)
            print("mid_volume:", mid_volume)
            print("mid_volume_ratio:", mid_volume_ratio)
            print("high_volume:", high_volume)
            print("high_volume_ratio:", high_volume_ratio)
            print("decayed_volume:", decayed_volume)
            print("total_volume:", total_volume)
            print("total_volume_ratio:", total_volume_ratio)
            print("radius_ratio:", radius_ratio)
            print("speedup_factor:", speedup_factor)
            print("tick_increment:", tick_increment)
            print("tick:", tick)

            # Paint Julia set
            self.screen.fill('black')
            self.julia.animate(low_volume_ratio, tick)

            # Paint frequency bar graph
            # Frequency graphs are in logarithmic scale
            # HACK(ray): Multiply width by 2 each time
            # TODO(ray): How do you have more bars than log2(len(audio_data))??
            last_idx = 0
            bar_span = 1
            bars = []
            for i in range(int(np.log2(len(audio_data)))):
                to_idx = last_idx + bar_span
                bar_data = audio_data[last_idx : to_idx]
                bar = np.sum(bar_data) / VOLUME_HIGH_POINT
                bars.append(bar)
                last_idx = to_idx
                bar_span = bar_span * 2
            self.graph.render(bars)

            # Paint metrics
            metric_lines = [
                "millis_elapsed:    " + metric_int_format.format(millis_elapsed),
                "tick:              " + metric_int_format.format(tick),
                "low_volume_ratio:  " + metric_ratio_format.format(low_volume_ratio),
                "mid_volume_ratio:  " + metric_ratio_format.format(mid_volume_ratio),
                "high_volume_ratio: " + metric_ratio_format.format(high_volume_ratio),
                "radius_ratio:      " + metric_ratio_format.format(radius_ratio),
                "speedup_factor:    " + metric_ratio_format.format(speedup_factor),
            ]
            for idx in range(len(metric_lines)):
                line = metric_lines[idx]
                text = self.font.render(line, True, white)
                textRect = text.get_rect()
                textRect_y = metrics_start_y + (metrics_font_size + metrics_spacing) * idx
                textRect.topleft = (metrics_start_x, textRect_y)
                self.screen.blit(text, textRect)

            # Update display
            pg.display.flip()

if __name__ == '__main__':
    app = App()
    app.run()
