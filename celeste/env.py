import matplotlib.pyplot as plt
import numpy as np

import utils


class Env(object):

  def __init__(self):
    self.saved_states = {}

  def quit(self):
    """Perform cleanup before quitting program."""
    pass

  def can_savestate(self):
    """Returns whether a savestate can be performed right now."""
    return True

  def savestate(self, index):
    """Performs a savestate. Must write into self.saved_states[index]."""
    self.saved_states[index] = True

  def loadstate(self, index):
    """Performs a loadstate."""
    pass

  def reset(self):
    """Resets state for a new episode."""
    pass

  def num_actions(self):
    """Returns the size of the action space (softmax dim)."""
    raise NotImplementedError()

  def start_frame(self):
    """Performs processing at the start of a frame.

    Returns:
      (frame, action) pair, where frame is passed into get_inputs_for_frame
      and action, if not None, will bypass the network and provide the given
      action to end_frame directly.
    """
    raise NotImplementedError()

  def get_inputs_for_frame(self, frame):
    """Returns inputs for networks from the frame obtained from start_frame.

    The returned tensors are passed as-is to network.set_inputs().
    """
    raise NotImplementedError()

  def get_reward(self):
    """Returns (rewards, should_end_episode) for the current state."""
    raise NotImplementedError()

  def indices_to_actions(self, idxs):
    """Given softmax indices, return a batch of actions to be provided to end_frame."""
    raise NotImplementedError()

  def indices_to_labels(self, idxs):
    """Given softmax indices, return a batch of string action names."""
    raise NotImplementedError()

  def end_frame(self, action):
    """Performs processing at the end of a frame given a single action. (batch not yet supported)"""
    raise NotImplementedError()

  def finish_episode(self, processed_frames, frame_buffer):
    """Called at the end of an episode with the number of frames processed."""
    pass

  def _add_action_summaries_image(self, ax, frame_number, frame_buffer):
    utils.imshow(frame_buffer[frame_number], ax)
    ax.axis('off')

  def _add_action_summaries_actions(self, ax, softmax, sampled_idx):
    num_topk = 5
    topk_idxs = np.argsort(softmax)[::-1][:num_topk]
    ax.bar(np.arange(num_topk), softmax[topk_idxs], width=0.3)
    ax.set_xticks(np.arange(num_topk))
    ax.set_xticklabels(self.indices_to_labels(topk_idxs))
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Sampled: %s (%0.2f%%)' % (
        self.indices_to_labels([sampled_idx])[0],
        softmax[sampled_idx] * 100.0))

  def add_action_summaries(self, frame_number, frame_buffer, softmax, sampled_idx):
    """Called with the intermediate frame softmaxes and sampled idxs."""
    ax1_height_ratio = 3
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={
        'height_ratios' : [ax1_height_ratio, 1],
    })
    self._add_action_summaries_image(ax1, frame_number, frame_buffer)
    self._add_action_summaries_actions(ax2, softmax, sampled_idx)

    asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
    ax2.set_aspect(asp / ax1_height_ratio)
    utils.add_summary('figure', 'action/frame_%03d' % frame_number, fig)
