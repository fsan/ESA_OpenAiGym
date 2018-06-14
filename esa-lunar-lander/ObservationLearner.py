from sklearn.neural_network import MLPRegressor
import math
import pickle
import numpy as np
import random

class RewardLeaner(object):
	def __init__(self, filename):
		print('initing reward')
		f = open(filename, 'rb')
		xs = []
		ys = []
		is_loading = True
		try:
			while is_loading:
				elem = pickle.load(f)
				x = np.array(elem['observation'])
				# print x.shape
				#y = 
				y = np.float64(elem['total_reward'])
				xs.append(x)
				ys.append(y)
		except EOFError:
			is_loading = False
		if not f.closed:
			f.close()
		print("Entries: ", len(xs))
		self.xs = xs
		self.ys = ys

	def split(self, train_rate):
		print("splitting")
		xys = list(zip(self.xs,self.ys))
		random.shuffle(xys)
		self.xs, self.ys = zip(*xys)

		train_size = int(math.floor(len(self.xs) * train_rate))
		train_xs, test_xs = self.xs[:train_size] , self.xs[train_size:]
		train_ys, test_ys = self.ys[:train_size] , self.ys[train_size:]

		#del self.xs
		#del self.ys

		return train_xs, train_ys, test_xs, test_ys
	def train(self, xs, ys):
		print('training reward')
		self.reg = MLPRegressor()
		print('starting fiting reward')
		self.reg = self.reg.fit(xs, ys)
		print('fit done')

	def test(self, xs, ys):
		print('testing reward')
		errs = []
                print self.reg.score(xs,ys)

        def save(self, fname):
            pickle.dump(self.reg, open(fname, 'wb'))


class ObservationLearner(object):
	def __init__(self, filename):
		print('initing observation')
		f = open(filename, 'rb')
		xs = []
		ys = []
		is_loading = True
		try:
			while is_loading:
				elem = pickle.load(f)
				x = np.array(elem['previous'] + [elem['action']])
				# print x.shape
				#y = np.float64(elem['reward'])
				y = np.array(elem['next'])
				xs.append(x)
				ys.append(y)
		except EOFError:
			is_loading = False
		if not f.closed:
			f.close()
		print("Entries: ", len(xs))
		self.xs = xs
		self.ys = ys

	def split(self, train_rate):
		print("splitting")
		xys = list(zip(self.xs,self.ys))
		random.shuffle(xys)
		self.xs, self.ys = zip(*xys)

		train_size = int(math.floor(len(self.xs) * train_rate))
		train_xs, test_xs = self.xs[:train_size] , self.xs[train_size:]
		train_ys, test_ys = self.ys[:train_size] , self.ys[train_size:]

		#del self.xs
		#del self.ys

		return train_xs, train_ys, test_xs, test_ys


	def train(self, xs, ys):
		print('training')
#		self.reg = MLPRegressor(hidden_layer_sizes=(100,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
#		               learning_rate='adaptive', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=False,
#		               random_state=None, tol=0.0001, verbose=True, warm_start=False, momentum=0.9,
#		               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
#		               epsilon=1e-08)
		#self.reg = MLPRegressor(hidden_layer_sizes=200, solver='adam', activation='relu', shuffle=True, epsilon=1e-08, learning_rate='adaptive', learning_rate_init=0.01, nesterovs_momentum=True, momentum=0.9)
		self.reg = MLPRegressor()
		print('starting fiting')
		self.reg = self.reg.fit(xs, ys)
		print('fit done')

	def test(self, xs, ys):
		print('testing')
		errs = []
                print self.reg.score(xs,ys)
#		for x,y in zip(xs,ys):
#			x = np.array(x).reshape(1,-1)
#			# print x.shape
#			res = self.reg.predict(x)
#			# print res
#			err = y * y - res * res #(y-res) * (y-res)
#			# print('error: ', err)
#			errs.append(err)
#		avg = float(sum(errs)) / len(errs)
#		print('average: ', avg)

        def save(self, fname):
            pickle.dump(self.reg, open(fname, 'wb'))

o = ObservationLearner('learn.pkl')
data = o.split(0.75)
o.train(data[0],data[1])
o.test(data[2],data[3])
o.save('physics-db.pkl')

r = RewardLeaner('reward.pkl')
data = r.split(0.75)
r.train(data[0],data[1])
r.test(data[2],data[3])
r.save('reward-reg.pkl')
