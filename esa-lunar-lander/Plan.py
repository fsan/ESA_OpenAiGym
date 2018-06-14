import copy
from operator import itemgetter
import random
from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
from scipy.spatial import distance

class ActionPlan(object):
	def __init__(self, initial_plan_size=5, training=False):
		self.initial_plan_size = initial_plan_size
		self.actions = [0] * initial_plan_size
		self.current_pos = 0 
		self.in_training = training
		if not self.in_training:
			self.rew_reg = pickle.load(open('reward-reg.pkl', 'rb'))
		else:
			self.rew_reg = None

	def get(self, i):
		return self.actions[i]

	def has_next(self):
		if self.current_pos >= len(self.actions):
			return False
		return True

	def get_next(self):
		if self.current_pos >= len(self.actions):
			return None
		
		result = self.actions[self.current_pos]
		self.current_pos += 1
		return result

	def replan(self, selected, engines, observation):
		for e in engines:
			if e.id != selected:
				e.subplan(observation)

	# def get_plan_copy(self):
	# 	return copy.copy(self.actions)

	# how is a subplan made of ?
	# [ fit, fit, fit, fit ]
	# [ [fit, action], [fit, action], [fit,action] ] # lets try this one

	def merge_subplans(self, subplans):
		min_sp_size = 1000
		# print len(subplans[0])
		for s in subplans:
			if len(s) < min_sp_size:
				min_sp_size = len(s)
                #print subplans

		for x in xrange(min_sp_size):
			acts = [ e[x] for e in subplans ]
			self.dispute(x, acts)

	def dispute(self, x, acts):
		# todo: acts will contain fitness?
		# merging plans need later to consider the gains and etc etc from theory
		#print acts
		# print 'acts', acts
		if self.in_training:
			cl = [ acts[z] for z in xrange(len(acts)) ]
		else:
			# this need to actually categorize the actions from worst to best
			# TODO
			## cl = [ (distance.euclidean(tuple((acts[z][0])[0:2]),(0,0)),acts[z][1]) for z in xrange(len(acts))]
			cl = [ (self.rew_reg.predict(np.array(acts[z][0]).reshape(1,-1)), acts[z][1]) for z in xrange(len(acts))]
		#print cl
		#raw_input()
		self.actions[x] = max(cl, key=itemgetter(0))[1] #must be min ?
		#print 'x', self.actions[x]
		#raw_input()
		# raw_input()
		# getstate = lambda z: z[0]
		# getact = lambda z: z[1]

		# for i in xrange(len(acts)):
		# 	print acts[i]
		# 	print 'getstate', getstate(acts[i])
		# 	print 'getact', getact(acts[i])

		# raw_input()

		# arr = [ (distance.euclidean(tuple(getstate(acts[i])[0:2]), tuple([0,0])), getact(acts[i])) for i in xrange(len(acts))]

		# print arr
		# raw_input()


		# self.actions[x] = min(arr, key=lambda z: z[0])[0]
		# print 'self.actions[x]', self.actions[x]


class EngineAgent(object):
	def __init__(self, idz, initial_plan_size=3, training=False):
		self.id = idz
		self.in_training = training
		self.initial_plan_size = initial_plan_size
		if not training:
			self.reg = pickle.load(open('physics-db.pkl', 'rb'))
		else:
			self.reg = None
			self.rew_reg = None

	def subplan(self, initial_state):
		last_state = initial_state
		#for p in xrange(len(master_plan.actions)):
		# should we plan all the next state, or should we try to figure out only the next steps
		# and then recalculate ?
		self._subplan = [False] * self.initial_plan_size
		for p in xrange(self.initial_plan_size):
			# result_active, next_state_active = self.predict_next_state(last_state, True)
			## result_active = self.predict_next_state(last_state, self.id)[0]
			#print last_state
			active_result = self.predict_next_state(last_state, self.id)
			# result_inactive, next_state_inactive = self.predict_next_state(last_state, False)
			## result_inactive = self.predict_next_state(last_state, 0)[0]
			inactive_result = self.predict_next_state(last_state, 0)

			self._subplan[p] = self.select_best(active_result, inactive_result)

			activate = self._subplan[p][1] > 0
			# print activate

			if activate:
				last_state = active_result[0]
			else:
				last_state = inactive_result[0]

		# print subplan
		# self._subplan = subplan

	def get_subplan(self):
		return self._subplan

	# todo:
	# - learn to adjust prediction
	# - learn to choose between actions

	def predict_next_state(self, state, active):
		if self.in_training:
			return random.random(), []
		S = self.reg.predict(np.array(state + [active]).reshape(1,-1))
		#print(S)
		return S

	def select_best(self, active_result, inactive_result):
		#print active_result[0:2]
		if self.in_training:
			if active_result[0] >= inactive_result[0]:
				return (active_result[0], self.id)
			return (inactive_result[0], 0)			
		else:
			active_desired_dist = distance.euclidean(tuple(active_result[0][0:2]), tuple(np.zeros(active_result[0][0:2].shape)))
			inactive_desired_dist = distance.euclidean(tuple(inactive_result[0][0:2]), tuple(np.zeros(inactive_result[0][0:2].shape)))
			
			active_desired_yspeed = active_result[0][3] #distance.euclidean(tuple(active_result[0][2:4]), tuple(np.zeros(active_result[0][2:4].shape)))
			inactive_desired_yspeed = inactive_result[0][3] #distance.euclidean(tuple(inactive_result[0][2:4]), tuple(np.zeros(inactive_result[0][2:4].shape)))
			
			active_desired_xspeed = active_result[0][2]
			inactive_desired_xspeed = inactive_result[0][2]

			active_desired_rot = active_result[0][4]
			inactive_desired_rot = inactive_result[0][4]

			#print self.id, active_desired_xspeed, inactive_desired_xspeed
			#print self.id, active_desired_yspeed, inactive_desired_yspeed
			# print self.id, active_desired_rot

			# center motor
			if self.id == 2:
				if inactive_desired_dist < 0.1:
					if inactive_desired_yspeed < -0.05:
						return (active_result[0], self.id)
					elif inactive_desired_rot < -0.2:
						return (active_result[0], 1)
					elif inactive_desired_rot > 0.2:
						return (active_result[0], 3)						
				elif inactive_desired_dist <= 0.3:
					if inactive_desired_rot < -0.2:
						return (active_result[0], 1)
					elif inactive_desired_rot > 0.2:
						return (active_result[0], 3)						
					elif inactive_desired_yspeed < -0.05:
						return (active_result[0], self.id)

			# rot to the right is -2.7
			# rot to the left is 1.25
			if self.id == 1:
				if inactive_desired_rot < -0.4:
					return (active_result[0], self.id)

			if self.id == 3:
				if inactive_desired_rot > 0.4:
					return (active_result[0], self.id)

			if self.id == 2:
				if inactive_desired_yspeed < -0.2:
					return (active_result[0], self.id)

			# 3 moves to the right
			# 1 moves to the left
			elif self.id == 1:
				if inactive_desired_xspeed >= 0.4:
					return (active_result[0], self.id)
			elif self.id == 3:
				if inactive_desired_xspeed <= -0.4:
					return (active_result[0], self.id)

			if inactive_desired_dist <= active_desired_dist:
				return (inactive_result[0], 0)		
			return (active_result[0], self.id)
