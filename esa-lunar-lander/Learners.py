from sklearn.neural_network import MLPRegressor
import math
import pickle
import numpy as np
import random

class ObservationLearner(object):
	def __init__(self, filename):
		print self, "Init"
		f = open(filename, 'rb')
		xs = []
		ys = []
		is_loading = True
		try:
			while is_loading:
				elem = pickle.load(f)
				x = np.array(np.append(elem['observation'],elem['action']))
				y = np.array(elem['next_state'])
				xs.append(x)
				ys.append(y)
		except EOFError:
			is_loading = False
		if not f.closed:
			f.close()
		print self, "Entries: ", len(xs)
		self.xs = xs
		self.ys = ys

	def split(self, train_rate):
		print self, "splitting"
		xys = list(zip(self.xs,self.ys))
		random.shuffle(xys)
		self.xs, self.ys = zip(*xys)

		train_size = int(math.floor(len(self.xs) * train_rate))
		train_xs, test_xs = self.xs[:train_size] , self.xs[train_size:]
		train_ys, test_ys = self.ys[:train_size] , self.ys[train_size:]

		return train_xs, train_ys, test_xs, test_ys


	def train(self, xs, ys):
		print self, 'training'
#		self.reg = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
#		               learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=False,
#		               random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9,
#		               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#		               epsilon=1e-08)
		#self.reg = MLPRegressor(hidden_layer_sizes=200, solver='adam', activation='relu', shuffle=True, epsilon=1e-08, learning_rate='adaptive', learning_rate_init=0.01, nesterovs_momentum=True, momentum=0.9)
		self.reg = MLPRegressor()
		print self, 'starting fiting'

		self.reg = self.reg.fit(xs, ys)
		print self, 'fit done'

	def test(self, xs, ys):
		print self, 'testing'
		errs = []
                print self.reg.score(xs,ys)

        def save(self, fname):
            pickle.dump(self.reg, open(fname, 'wb'))




class RewardLearner(object):
	def __init__(self, filename):
		print self, "Init"
		f = open(filename, 'rb')
		xs = []
		ys = []
		is_loading = True
		try:
			while is_loading:
				elem = pickle.load(f)
				x = np.array(np.append(elem['observation'],elem['action']))
				y = np.array(elem['reward'])
				xs.append(x)
				ys.append(y)
		except EOFError:
			is_loading = False
		if not f.closed:
			f.close()
		print self, "Entries: ", len(xs)
		# self.reg = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
		# 	beta_2=0.999, early_stopping=False, epsilon=1e-08,
		# 	hidden_layer_sizes=(100,), learning_rate='constant',
		# 	learning_rate_init=0.001, max_iter=200, momentum=0.9,
		# 	nesterovs_momentum=True, power_t=0.5, random_state=None,
		# 	shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
		# 	verbose=True, warm_start=True)
		self.reg = MLPRegressor(hidden_layer_sizes=(27,54,27,), activation='relu', alpha=0.0001,
			batch_size='auto', beta_1=0.9,
			beta_2=0.999, early_stopping=False, epsilon=1e-08, learning_rate='constant',
			learning_rate_init=0.001, max_iter=200, momentum=0.9,
			nesterovs_momentum=True, power_t=0.5, random_state=None,
			shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
			verbose=False, warm_start=False)
		self.xs = xs
		self.ys = ys

	def split(self, train_rate):
		print self, "splitting"
		xys = list(zip(self.xs,self.ys))
		random.shuffle(xys)
		self.xs, self.ys = zip(*xys)

		train_size = int(math.floor(len(self.xs) * train_rate))
		train_xs, test_xs = self.xs[:train_size] , self.xs[train_size:]
		train_ys, test_ys = self.ys[:train_size] , self.ys[train_size:]

		return train_xs, train_ys, test_xs, test_ys


	def train(self, xs, ys):
		print self, 'training'
#		self.reg = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
#		               learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=False,
#		               random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9,
#		               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#		               epsilon=1e-08)
		#self.reg = MLPRegressor(hidden_layer_sizes=200, solver='adam', activation='relu', shuffle=True, epsilon=1e-08, learning_rate='adaptive', learning_rate_init=0.01, nesterovs_momentum=True, momentum=0.9)
		# self.reg = MLPRegressor()
		print self, 'starting fiting'

		self.reg = self.reg.fit(xs, ys)
		print self, 'fit done'

	def test(self, xs, ys):
		print self, 'testing'
		errs = []
                print self.reg.score(xs,ys)

        def save(self, fname):
            pickle.dump(self.reg, open(fname, 'wb'))

if __name__ == '__main__':
	import sys
	for v in sys.argv:
		if v == 'o':
			print 'Learning physics'
			o = ObservationLearner('observation-record.pkl')
			data = o.split(0.75)
			o.train(data[0],data[1])
			o.test(data[2],data[3])
			o.save('physics.pkl')

		elif v == 'r':
			print 'Learning rewards'
			r = RewardLearner('rewards-record.pkl')
			data = r.split(0.75)
			r.train(data[0],data[1])
			r.test(data[2],data[3])
			r.save('rewards.pkl')

		elif v == 'ropt':
			r = RewardLearner('rewards-record.pkl')
			data = r.split(0.75)
			# Grid Search for Algorithm Tuning
			from sklearn.model_selection import GridSearchCV
			from sklearn.model_selection import RandomizedSearchCV
			# prepare a range of alpha values to test
			alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
			# create and fit a ridge regression model, testing each alpha
			grid = GridSearchCV(estimator=r.reg, param_grid=dict(alpha=alphas), verbose=True)
			print 'grid fit'
			grid.fit(data[0], data[1])
			print(grid)
			# summarize the results of the grid search
			print(grid.best_score_)
			print(grid.best_estimator_.alpha)

			# prepare a uniform distribution to sample for the alpha parameter
			# from scipy.stats import uniform as sp_rand
			# param_grid = {'alpha': sp_rand()}
			# # create and fit a ridge regression model, testing random alpha values
			# print 'rsearch'
			# rsearch = RandomizedSearchCV(estimator=r.reg, param_distributions=param_grid, n_iter=100, verbose=True)
			# rsearch.fit(data[0], data[1])
			# print(rsearch)
			# # summarize the results of the random parameter search
			# print(rsearch.best_score_)
			# print(rsearch.best_estimator_.alpha)