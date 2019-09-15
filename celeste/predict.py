from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import pprint
import signal
import torch

import celeste_detector
import environment
import model
import speedrun
import utils

FLAGS = flags.FLAGS


class Predictor(speedrun.Trainer):

  def __init__(self, env):
    self.env = env
    self.det = celeste_detector.CelesteDetector()
    self.processed_frames = 0
    self.trajectory = []
    self.sampled_action = []
    self._generate_goal_state()

    frame_channels = 4
    extra_channels = 1 + len(utils.button_dict)
    actor_network = getattr(model, FLAGS.actor_network)
    self.actor = actor_network(frame_channels, extra_channels, out_dim=len(utils.class2button.dict))
    if FLAGS.use_cuda:
      self.actor = self.actor.cuda()
    self.actor.reset()
    self.actor.eval()

    assert FLAGS.pretrained_model_path
    logging.info('Loading pretrained model from %s' % FLAGS.pretrained_model_path)
    state = torch.load(os.path.join(FLAGS.pretrained_model_path, 'train/model_%s.tar' % FLAGS.pretrained_suffix))
    self.actor.load_state_dict(state['actor'])
    logging.info('Done!')

  def _generate_goal_state(self):
    assert FLAGS.random_goal_prob == 0.0
    return super(Predictor, self)._generate_goal_state()

  def process_frame(self, frame):
    """Returns a list of button inputs for the next N frames."""
    self._set_inputs_from_frame(frame)

    y, x = self.trajectory[-1]
    if y is None:
      # Assume death
      input('Death detected. Press enter to quit.')
      return None
    dist_to_goal = np.sqrt((y - self.goal_y)**2 + (x - self.goal_x)**2)
    if dist_to_goal < 5:
      input('Goal reached. Press enter to quit.')
      return None

    with torch.no_grad():
      softmax = self.actor.forward(self.processed_frames).detach().cpu()
    idxs, button_inputs = utils.sample_action(softmax, greedy=True)
    self.sampled_action.append(idxs)
    # Predicted button_inputs include a batch dimension.
    button_inputs = button_inputs[0]
    # Returned button_inputs should be for next N frames, but for now N==1.
    button_inputs = [button_inputs] * FLAGS.hold_buttons_for

    self.processed_frames += 1
    return button_inputs


def main(argv):
  del argv  # unused

  logging.info('\n%s', pprint.pformat(FLAGS.flag_values_dict()))

  env = environment.Environment()
  try:
    speedrun.Speedrun(env, Predictor)
  finally:
    env.cleanup()


if __name__ == '__main__':
  app.run(main)
