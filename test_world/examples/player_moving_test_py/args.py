class Argument(object):
	def __init__(self):
		# Core training parameters
		self.lr =1e-2 # learning rate for Adam optimizer
		self.gamma = 0.95 # discount factor
		self.batch_size = 1024 # number of episodes to optimize at the same time
		self.min_buffer_size = self.batch_size * 25 # minimum replay buffer size to update
		self.num_units = 64 # number of units in the mlp
		# Checkpointing
		self.restore = True
		# self.save_dir = "./save_model/aiwc_maddpg" # directory in which training state and model should be saved
		# self.save_rate = 1000 # save model once every time this many episodes are completed
		self.load_dir = "./save_model/moving_quarter/moving_quarter-" # directory in which training state and model are loaded