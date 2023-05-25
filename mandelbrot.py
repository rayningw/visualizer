from math import log
import numba
import pygame as pg
import numpy as np

# settings
res = width, height = 800, 450
offset = np.array([1.3 * width, height]) // 2
# texture
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture).astype(dtype=np.uint32)

# Original source: https://github.com/StanislavPetrovV/Mandelbrot-set-Realtime-Viewer-/blob/main/main.py
# Video: https://www.youtube.com/watch?v=B01dLzU3LkQ
# Stripped back to `numba` only and not `taichi`
class Fractal:
    def __init__(self, app):
        self.app = app
        self.screen_array = np.full((width, height, 3), [0, 0, 0], dtype=np.uint8)
        # control settings
        self.vel = 0.01
        self.max_iter = 30
        self.max_iter_limit = 5500
        self.zoom = 2.2 / height
        self.scale = 0.993
        self.dx = 0
        self.dy = 0
        # delta_time
        self.app_speed = 1 / 4000
        self.prev_time = pg.time.get_ticks()

    def delta_time(self):
        time_now = pg.time.get_ticks() - self.prev_time
        self.prev_time = time_now
        return time_now * self.app_speed

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def render(max_iter, zoom, dx, dy, screen_array, smooth=False):
        for x in numba.prange(width):
            for y in range(height):
                c = ((x - offset[0]) * zoom - dx) + 1j * ((y - offset[1]) * zoom - dy)
                z = 0
                num_iter = 0
                for i in range(max_iter):
                    z = z ** 2 + c
                    if z.real ** 2 + z.imag ** 2 > 4:
                        break
                    num_iter += 1

                if smooth:
                    # Fractional escape count:
                    # https://realpython.com/mandelbrot-set-python/#smoothing-out-the-banding-artifacts
                    # NOTE(ray): Don't know how to adapt speedy magnitude check here
                    escape_count = num_iter + 1 - log(log(abs(z))) / log(2)
                    escape_ratio = escape_count / max_iter
                    clamped_ratio = max(0.0, min(escape_ratio, 1.0))
                    col = int(texture_size * clamped_ratio)
                else:
                    col = int(texture_size * num_iter / max_iter)
                screen_array[x, y] = texture_array[col, col]
        return screen_array

    def control(self):
        pressed_key = pg.key.get_pressed()
        dt = self.delta_time()
        # movement
        if pressed_key[pg.K_a]:
            self.dx += self.vel * dt
        if pressed_key[pg.K_d]:
            self.dx -= self.vel * dt
        if pressed_key[pg.K_w]:
            self.dy += self.vel * dt
        if pressed_key[pg.K_s]:
            self.dy -= self.vel * dt

        # stable zoom and movement
        if pressed_key[pg.K_UP] or pressed_key[pg.K_DOWN]:
            inv_scale = 2 - self.scale
            if pressed_key[pg.K_UP]:
                self.zoom *= self.scale
                self.vel *= self.scale
            if pressed_key[pg.K_DOWN]:
                self.zoom *= inv_scale
                self.vel *= inv_scale

        # mandelbrot resolution
        if pressed_key[pg.K_LEFT]:
            self.max_iter -= 1
        if pressed_key[pg.K_RIGHT]:
            self.max_iter += 1
        self.max_iter = min(max(self.max_iter, 2), self.max_iter_limit)

    def update(self):
        self.control()
        self.screen_array = self.render(self.max_iter, self.zoom, self.dx, self.dy, self.screen_array)

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.screen_array)

    def run(self):
        self.update()
        self.draw()


class App:
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            [exit() for i in pg.event.get() if i.type == pg.QUIT]
            self.clock.tick()
            pg.display.set_caption(f'FPS: {self.clock.get_fps() :.2f}')


if __name__ == '__main__':
    app = App()
    app.run()
