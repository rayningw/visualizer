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

    # distance between each vertex
    # convenient to set such that len(self.ypoints) * len(self.xpoints) == power
    # of 2 in order to help FFT
    self.nsteps = 1.3

    # x and y coords of the vertices
    self.ypoints = np.arange(-20, 20 + self.nsteps, self.nsteps)
    self.xpoints = np.arange(-20, 20 + self.nsteps, self.nsteps)

    # number of triangle faces
    # NOTE(ray): have no idea why it's set this way
    self.nfaces = len(self.ypoints)

    # current camera point
    self.offset = 0

    # sampling rate
    self.RATE = 44100

    # number of samples per read
    self.CHUNK = len(self.xpoints) * len(self.ypoints)

    # audio objects
    self.p = pyaudio.PyAudio()
    self.stream = self.p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=self.RATE,
        input=True,
        output=True,
        frames_per_buffer=self.CHUNK,
    )

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

    waveform_data = self.stream.read(self.CHUNK)

    verts, faces, colors = self.mesh(waveform_data=waveform_data)
    self.mesh1.setMeshData(
      vertexes=verts, faces=faces, faceColors=colors
    )
    self.offset -= 0.05

  # NOTE(ray): For some reason the waveform needs to be passed in instead of
  # being read here
  def mesh(self, waveform_data=None):

    if waveform_data is not None:
      # convert into integer array
      waveform_data = struct.unpack(str(2 * self.CHUNK) + 'B', waveform_data)
      # something about taking every other element
      waveform_data = np.array(waveform_data, dtype='b')[::2] + 128
      # center around 0
      waveform_data = np.array(waveform_data, dtype='int32') - 128
      # reduce amplitude
      waveform_data = waveform_data * 0.04
      # reshape into 2d array
      waveform_data = waveform_data.reshape((len(self.xpoints), len(self.ypoints)))
    else:
      waveform_data = np.array([1] * 1024)
      waveform_data = waveform_data.reshape((len(self.xpoints), len(self.ypoints)))

    verts = [
      [
        x, y,
        waveform_data[x_idx][y_idx] * self.noise.noise2(x=x_idx/5 + self.offset, y=y_idx/5 + self.offset)
      ] for x_idx, x in enumerate(self.xpoints) for y_idx, y in enumerate(self.ypoints)
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