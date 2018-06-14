import random
import numpy as np
import pickle

from operator import itemgetter
from sklearn.neural_network import MLPRegressor

class Planner(object):
	"""docstring for Planner"""
	def __init__(self, plan_size, state_regressor_filename, reward_regressor_filename, learning_mode=False, engines_qtd=3):
		self.learning_mode = learning_mode
		self.plan_size = plan_size
		self.plan = np.zeros(shape=(self.plan_size,), dtype=np.int)

		mode = 'rb'
		if self.learning_mode:
			mode = 'wb'

		self.state_regressor_file = open(state_regressor_filename, mode)
		self.reward_regressor_file = open(reward_regressor_filename, mode)

		if not self.learning_mode:
			self.state_regressor = pickle.load(self.state_regressor_file)
			self.reward_regressor = pickle.load(self.reward_regressor_file)
			self.engines = [ Engine(x, self.state_regressor, self.reward_regressor) for x in xrange(1, engines_qtd + 1) ]
		else:
			self.engines = xrange(1,engines_qtd+1)

		self.replan = True


	def setPlan(self, state, best_heuristic= lambda x, y: max(x, key=itemgetter(2))):
		self.current = 0
		self.replan = False
		predictions = []

		assert callable(best_heuristic)

		for p in xrange(self.plan_size):
			if not self.learning_mode:
				for eng in self.engines:
					predictions.append(
						(eng.id, eng.predict_next_state(state),	eng.predict_reward(state))
					)
				predictions.append(
					(
						0,
						self.state_regressor.predict(np.append(state, 0).reshape(1,-1)),
						self.reward_regressor.predict(np.append(state, 0).reshape(1,-1))
					)
				)
				best = best_heuristic(predictions, state)
				self.plan[p] = best[0]
				state = best[1]
			else: #if learning
				self.plan[p] = random.randint(0,len(self.engines))

	def step(self):
		ret = self.plan[self.current]
		self.current = (self.current + 1) % len(self.plan)
		if self.current == 0:
			self.replan = True
		return ret

	def dump(self, state_data, reward_data):
		if self.learning_mode:
			pickle.dump(state_data, self.state_regressor_file)
			pickle.dump(reward_data, self.reward_regressor_file)

	def close(self):
		if self.state_regressor_file and not self.state_regressor_file.closed:
			self.state_regressor_file.close()
		if self.reward_regressor_file and not self.reward_regressor_file.closed:
			self.reward_regressor_file.close()			

class Engine(object):
	"""docstring for Engine"""
	def __init__(self, i, state_regressor=None, reward_regressor=None, learning_mode=False):
		self.id = i
		self.state_regressor = state_regressor
		self.reward_regressor = reward_regressor
		self.learning_mode = learning_mode

	def predict_next_state(self, state, engid=None):
		if not self.learning_mode:
			if engid is None:
				engid = self.id
			arr = np.append(state, engid)
			s = arr.reshape(1,-1)
			return self.state_regressor.predict(s)

	def predict_reward(self, state, engid=None):
		if not self.learning_mode:
			if engid is None:
				engid = self.id			
			arr = np.append(state,engid)
			# print arr
			s = arr.reshape(1,-1)
			return self.reward_regressor.predict(s)

		