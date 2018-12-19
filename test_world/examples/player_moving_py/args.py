class Argument(object):
	def __init__(self):
		# Environment
		self.max_episode_len = 1 # maximum episode length
		# Core training parameters
		self.lr =1e-2 # learning rate for Adam optimizer
		self.gamma = 0.95 # discount factor
		self.batch_size = 16 # number of episodes to optimize at the same time
		self.num_units = 64 # number of units in the mlp
		# Checkpointing
		self.restore = False
		self.save_dir = "./save_model/aiwc_maddpg" # directory in which training state and model should be saved
		self.save_rate = 1000 # save model once every time this many episodes are completed
		self.load_dir = "./save_model/aiwc_maddpg" # directory in which training state and model are loaded