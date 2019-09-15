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

  def index_to_action(self, idx):
    """Given a softmax index, return an action to be provided to end_frame."""
    raise NotImplementedError()

  def end_frame(self, action):
    """Performs processing at the end of a frame given an action."""
    raise NotImplementedError()

  def finish_episode(self, processed_frames):
    """Called at the end of an episode with the number of frames processed."""
    pass

  def add_action_summaries(self, frame_number, softmax, sampled_idx):
    """Called with the intermediate frame softmaxes and sampled idxs."""
    pass
