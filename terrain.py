import numpy as np
from pyqtgraph import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
from opensimplex import OpenSimplex
import struct
import pyaudio

# Terrain flyover
# Based off https://github.com/markjay4k/Audio-Spectrum-Analyzer-in-Python/blob/master/terrain.py
# Video: https://www.youtube.com/watch?v=a1kxPd8Mdhw
class Terrain(object):
  def __init__(self):
    """
    Initialize the graphics window and mesh
    """

    # setup the view window
    self.app = QtWidgets.QApplication(sys.argv)
    self.window = gl.GLViewWidget()
    self.window.setGeometry(0, 110, 1920, 1080)
    self.window.show()
    self.window.setWindowTitle('Terrain')
    self.window.setCameraPosition(distance=30, elevation=8)

    # constants
    self.nsteps = 1
    self.ypoints = range(-20, 22, self.nsteps)
    self.xpoints = range(-20, 22, self.nsteps)
    self.nfaces = len(self.ypoints)

    # current camera point
    self.offset = 0

    # perlin noise object
    self.noise = OpenSimplex(0)

    # create the mesh elements
    verts, faces, colors = self.mesh()

    # create the mesh
    self.mesh1 = gl.GLMeshItem(
      vertexes=verts,
      faces=faces,
      faceColors=colors,
      smooth=False,
      drawEdges=True,
    )
    self.mesh1.setGLOptions('additive')
    self.window.addItem(self.mesh1)

  def update(self):
    """
    update the mesh and shift the noise each time
    """
    verts, faces, colors = self.mesh()
    self.mesh1.setMeshData(
      vertexes=verts, faces=faces, faceColors=colors
    )
    self.offset -= 0.18

  def mesh(self):
    verts = [
      [
        x, y, 2.5 * self.noise.noise2(x=n/5 + self.offset, y=m/5 + self.offset)
      ] for n, x in enumerate(self.xpoints) for m, y in enumerate(self.ypoints)
    ]
    verts = np.array(verts, dtype=np.float32)

    faces = []
    colors = []
    for m in range(self.nfaces - 1):
      yoff = m * self.nfaces
      for n in range(self.nfaces - 1):
        faces.append([
          # Current row
          yoff + n,
          # Next row
          self.nfaces + yoff + n,
          # Next row + 1
          self.nfaces + yoff + n + 1,
        ])
        faces.append([
          # Current row
          yoff + n,
          # Current row + 1
          yoff + n + 1,
          # Next row + 1
          self.nfaces + yoff + n + 1,
        ])
        colors.append([n / self.nfaces, 1 - n / self.nfaces, m / self.nfaces, 0.7])
        colors.append([n / self.nfaces, 1 - n / self.nfaces, m / self.nfaces, 0.8])

    faces = np.array(faces, dtype=np.uint32)
    colors = np.array(colors, dtype=np.float32)

    return verts, faces, colors

  def start(self):
    """
    get the graphics window open and setup
    """
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
      self.app.exec()

  def animation(self):
    """
    calls the update method to run in a loop
    """
    timer = QtCore.QTimer()
    timer.timeout.connect(self.update)
    timer.start(10)
    self.start()

if __name__ == '__main__':
  t = Terrain()
  t.animation()