import numpy as np
import pygame as pg

def julia_quadratic(zx, zy, cx, cy, threshold):
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

# texture
texture = pg.image.load('img/texture.jpg')
texture_size = min(texture.get_size()) - 1
texture_array = pg.surfarray.array3d(texture).astype(dtype=np.uint32)

x_start, y_start = -2, -2  # an interesting region starts here
space_width, space_height = 4, 4  # for 4 units up and right
density_per_unit = 100  # how many pixels per unit

# real and imaginary axis
re = np.linspace(x_start, x_start + space_width, space_width * density_per_unit )
im = np.linspace(y_start, y_start + space_height, space_height * density_per_unit)

threshold = 20  # max allowed iterations
frames = 100  # number of frames in the animation

# we represent c as c = r*cos(a) + i*r*sin(a) = r*e^{i*a}
r = 0.7885
a = np.linspace(0, 2*np.pi, frames)

res = res_width, res_height = space_width * density_per_unit, space_height * density_per_unit
screen = pg.display.set_mode(res, pg.SCALED)
clock = pg.time.Clock()
screen_array = np.full((res_width, res_height, 3), [0, 0, 0], dtype=np.uint8)

def animate(i):
    cx, cy = r * np.cos(a[i]), r * np.sin(a[i])  # the initial c number
    
    # iterations for the given threshold
    for i in range(len(re)):
        for j in range(len(im)):
            escape_count = julia_quadratic(re[i], im[j], cx, cy, threshold)
            escape_ratio = escape_count / threshold
            col = int(texture_size * escape_ratio)
            screen_array[i, j] = texture_array[col, col]
    
    pg.surfarray.blit_array(screen, screen_array)

i = 0
while True:
    screen.fill('black')
    animate(i % frames)
    pg.display.flip()

    [exit() for i in pg.event.get() if i.type == pg.QUIT]
    clock.tick()
    pg.display.set_caption(f'FPS: {clock.get_fps() :.2f}')

    i += 1
