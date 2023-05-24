import numba
import numpy as np
import pygame as pg

# texture
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture).astype(dtype=np.uint32)

x_start, y_start = -2, -2  # an interesting region starts here
space_width, space_height = 4, 4  # for 4 units up and right
density_per_unit = 100  # how many pixels per unit

# resolution
res = res_width, res_height = space_width * density_per_unit, space_height * density_per_unit

# max allowed iterations
threshold = 20

# number of segments in revolution
num_segments = 100

# real and imaginary axis
re = np.linspace(x_start, x_start + space_width, space_width * density_per_unit)
im = np.linspace(y_start, y_start + space_height, space_height * density_per_unit)

# we represent c as c = r*cos(a) + i*r*sin(a) = r*e^{i*a}
radius = 0.7885
angles = np.linspace(0, 2*np.pi, num_segments)

# Renders a Julia set with C revolving around the origin
class JuliaRenderer:
    def __init__(self, screen, screen_array):
        self.screen = screen
        self.screen_array = screen_array

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(segment_index, screen_array):
        # Compute the C number
        cx, cy = radius * np.cos(angles[segment_index]), radius * np.sin(angles[segment_index])
        c = complex(cx, cy)
        
        # iterations for the given threshold
        for i in numba.prange(len(re)):
            for j in range(len(im)):
                # Compute Julia set membership
                z = complex(re[i], im[j])
                escape_count = 0
                for _i in range(threshold):
                    z = z**2 + c
                    if z.real ** 2 + z.imag ** 2 > 4:
                        break
                    escape_count += 1

                # Colour based on escape ratio
                escape_ratio = escape_count / threshold
                col = int(texture_size * escape_ratio)
                screen_array[i, j] = texture_array[col, col]
        
        return screen_array

    def animate(self, tick):
        segment_index = tick % num_segments
        self.render(segment_index, self.screen_array)
        pg.surfarray.blit_array(self.screen, self.screen_array)

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.screen_array = np.full((res_width, res_height, 3), [0, 0, 0], dtype=np.uint8)
        self.julia = JuliaRenderer(self.screen, self.screen_array)

    def run(self):
        tick = 0
        while True:
            self.screen.fill('black')
            self.julia.animate(tick)
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')

            tick += 1

if __name__ == '__main__':
    app = App()
    app.run()
