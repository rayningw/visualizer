import numpy as np
from pyqtgraph import QtCore, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import sys
from opensimplex import OpenSimplex

class Terrain(object):
  def __init__(self):
    """
    Initialize the graphics window and mesh
    """

    # setup the view window
    self.app = QtWidgets.QApplication(sys.argv)
    self.w = gl.GLViewWidget()
    self.w.setGeometry(0, 110, 1920, 1080)
    self.w.show()
    self.w.setWindowTitle('Terrain')
    self.w.setCameraPosition(distance=30, elevation=8)

    # constants and arrays
    self.nsteps = 1
    self.ypoints = range(-20, 22, self.nsteps)
    self.xpoints = range(-20, 22, self.nsteps)
    self.nfaces = len(self.ypoints)
    self.offset = 0

    # perlin noise object
    self.tmp = OpenSimplex(0)

    # create the veritices array
    verts = [
      [
        x, y, 1.5 * self.tmp.noise2(x=n/5, y=m/5)
      ] for n, x in enumerate(self.xpoints) for m, y in enumerate(self.ypoints)
    ]
    verts = np.array(verts, dtype=np.float32)

    # create the faces and colors arrays
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

    # create the mesh item
    self.m1 = gl.GLMeshItem(
      vertexes=verts,
      faces=faces,
      faceColors=colors,
      smooth=True,
      drawEdges=True,
    )
    self.m1.setGLOptions('additive')
    self.w.addItem(self.m1)

  def update(self):
    """
    update the mesh and shift the noise each time
    """
    verts = [
      [
        x, y, 2.5 * self.tmp.noise2(x=n/5 + self.offset, y=m/5 + self.offset)
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

    self.m1.setMeshData(
      vertexes=verts, faces=faces, faceColors=colors
    )
    self.offset -= 0.18

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
    self.update()


if __name__ == '__main__':
  t = Terrain()
  t.animation()