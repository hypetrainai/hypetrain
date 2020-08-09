from absl import flags
import math
import numpy as np
import pygame

FLAGS = flags.FLAGS


class Renderer:

  def __init__(self, ndisplay=np.inf, nrows=0):
    self._ndisplay = min(FLAGS.batch_size, ndisplay)
    self._nrows = nrows or int(math.sqrt(self._ndisplay))
    self._ncols = int(math.ceil(self._ndisplay / self._nrows))
    self._screen = None

  def render(self, frame):
    wait = False
    while wait:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          return
        if event.type == pygame.KEYDOWN:
          wait = False

    if not self._screen:
      pygame.init()
      height, width = frame.shape[-2:]
      self._screen = pygame.display.set_mode(
          [width * self._ncols, height * self._nrows], pygame.SCALED, depth=8)

    # [batch, channel, height, width] -> [batch, width, height, channel].
    frame = np.transpose(frame, [0, 3, 2, 1])
    if frame.shape[-1] == 1:
      frame = np.tile(frame, [1, 1, 1, 3])
    for i in range(self._ndisplay):
      start_x = (i % self._ncols) * FLAGS.image_width
      start_y = (i // self._ncols) * FLAGS.image_height
      self._screen.blit(pygame.surfarray.make_surface(frame[i]), (start_x, start_y))
    pygame.display.flip()
