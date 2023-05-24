import numpy as np
import pygame as pg

# texture
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture).astype(dtype=np.uint32)

x_start, y_start = -2, -2  # an interesting region starts here
space_width, space_height = 4, 4  # for 4 units up and right
density_per_unit = 100  # how many pixels per unit

# max allowed iterations
THRESHOLD = 20

# interesting radius to revolve
RADIUS = 0.7885

# number of segments in revolution
NUM_SEGMENTS = 100

res = res_width, res_height = space_width * density_per_unit, space_height * density_per_unit

# Renders a Julia set with C revolving around the origin
class JuliaRenderer:
    def __init__(self, screen, screen_array, radius, num_segments):
        self.screen = screen
        self.screen_array = screen_array

        # real and imaginary axis
        self.re = np.linspace(x_start, x_start + space_width, space_width * density_per_unit)
        self.im = np.linspace(y_start, y_start + space_height, space_height * density_per_unit)

        # we represent c as c = r*cos(a) + i*r*sin(a) = r*e^{i*a}
        self.radius = radius
        self.angles = np.linspace(0, 2*np.pi, num_segments)

        self.num_segments = num_segments

    def julia_quadratic(self, zx, zy, cx, cy, threshold):
        """Calculates whether the number z[0] = zx + i*zy with a constant c = x + i*y
        belongs to the Julia set. In order to belong, the sequence 
        z[i + 1] = z[i]**2 + c, must not diverge after 'threshold' number of steps.
        The sequence diverges if the absolute value of z[i+1] is greater than 4.
        
        :param float zx: the x component of z[0]
        :param float zy: the y component of z[0]
        :param float cx: the x component of the constant c
        :param float cy: the y component of the constant c
        :param int threshold: the number of iterations to considered it converged
        """
        # initial conditions
        z = complex(zx, zy)
        c = complex(cx, cy)
        
        for i in range(threshold):
            z = z**2 + c
            if abs(z) > 4.:  # it diverged
                return i
            
        return threshold - 1  # it didn't diverge

    def render(self, tick, threshold):
        segment_index = tick % self.num_segments

        # Compute the C number
        cx, cy = self.radius * np.cos(self.angles[segment_index]), self.radius * np.sin(self.angles[segment_index])
        
        # iterations for the given threshold
        for i in range(len(self.re)):
            for j in range(len(self.im)):
                escape_count = self.julia_quadratic(self.re[i], self.im[j], cx, cy, threshold)
                escape_ratio = escape_count / threshold
                col = int(texture_size * escape_ratio)
                self.screen_array[i, j] = texture_array[col, col]
        
        pg.surfarray.blit_array(self.screen, self.screen_array)

class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.screen_array = np.full((res_width, res_height, 3), [0, 0, 0], dtype=np.uint8)
        self.julia = JuliaRenderer(self.screen, self.screen_array, RADIUS, NUM_SEGMENTS)

    def run(self):
        tick = 0
        while True:
            self.screen.fill('black')
            self.julia.render(tick, THRESHOLD)
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')

            tick += 1

if __name__ == '__main__':
    app = App()
    app.run()
