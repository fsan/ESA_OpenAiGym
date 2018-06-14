# This is just a proof of concept demonstration for the Enhanced Subsumption Architecture
# This is the ESA-Strict version mentioned in the article.

import copy
import gym
import time
import random
import signal
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from operator import itemgetter
from threading import Thread
from EnginePlanner import Planner, Engine

from Learners import ObservationLearner, RewardLearner

DISPLAY, LEARNING, GRAPH, VERBOSE, FINAL_GRAPH = True, False, False, False, '0'

if len(sys.argv) >= 2:
	DISPLAY = (sys.argv[1] == 'true') or (sys.argv[1] == '1')
if len(sys.argv) >= 3:
	LEARNING = (sys.argv[2] == 'true') or (sys.argv[2] == '1')	
if len(sys.argv) >= 4:
	GRAPH = (sys.argv[3] == 'true') or (sys.argv[3] == '1')		
if len(sys.argv) >= 5:
	VERBOSE = (sys.argv[4] == 'true') or (sys.argv[4] == '1')	
if len(sys.argv) >= 6:
	FINAL_GRAPH = sys.argv[5]

success_landings = 0
fail_landings = 0
total_reward_history = []

def signal_handler(signal, frame):
        global running
        running = False

signal.signal(signal.SIGINT, signal_handler)


env = gym.make('LunarLander-v2')
observation = env.reset()
running = True
total_reward = 0.
step_count = 0

if GRAPH:
	rewards_points = []
	total_rewards_points = []
	strategy_history = []
	speed_history = []

	plt.ioff()
	fig, ax = plt.subplots(3,1, figsize=(7,12))
	plt.show(block=False)

if FINAL_GRAPH > 0:
	datapos_grid = np.zeros(shape=(13,22))

def update_graphs(fig, ax, canvas, strategy_history, strategies, rewards_points):
	# global fig, ax
	# global canvas
	# global strategy_history
	# global strategies
	# global rewards_points

	ax[0].clear()
	ax[1].clear()
	ax[2].clear()

	m = min(len(rewards_points), len(strategy_history))
	
	rewards_points = rewards_points[:m]
	strategy_history = strategy_history[:m]

	df = pd.DataFrame(data={'strategy':strategy_history, 'rewards':rewards_points}, index=xrange(len(strategy_history)))
	ax[0].set_ylim(-300,300)
	ax[0].plot(range(len(rewards_points)), rewards_points, 'b', alpha=0.75)
	ax[0].plot(range(len(total_rewards_points)), total_rewards_points, 'r', alpha=0.75)
	sns.factorplot(x=df.index, y='strategy', data=df, ax=ax[1], kind='swarm', legend=False)
	ax[2].plot(range(len(speed_history)), [y for (_,y,_ ) in speed_history], 'g', alpha=0.75)
	ax[2].plot(range(len(speed_history)), [x for (x,y,_ ) in speed_history], 'orange', alpha=0.75)
	ax[2].plot(range(len(speed_history)), [z for (_,_,z ) in speed_history], 'purple', alpha=0.75)
	ax[2].set_ylim(-1,+1)
	ax[2].legend(['vertical speed', 'horizontal speed', 'angular speed'],loc=9, bbox_to_anchor=(-0.33,0.66))
	fig.tight_layout()
	canvas.draw()

if GRAPH:
	backend = plt.rcParams['backend']
	figManager = matplotlib._pylab_helpers.Gcf.get_active()
	canvas = figManager.canvas

plan_size = 4
if LEARNING:
	P = Planner(plan_size, 'observation-record.pkl', 'rewards-record.pkl', LEARNING, 3)
else:
	P = Planner(plan_size, 'physics.pkl', 'rewards.pkl', LEARNING, 3)

from scipy.spatial import distance
strategies = [
	# 0 - reduce dist to center
	{'name':'reduce distance to center', 'fn':lambda x: min(x, key=lambda y: distance.euclidean((y[1][0][0],y[1][0][1]),(0,0)))},

	# 1 - reduce speed
	{'name':'reduce general speed','fn':lambda x: min(x, key=lambda y: distance.euclidean((y[1][0][2],y[1][0][3]),(0,0)))},

	# 2 - reduce angular movement (TODO WRONG)
	{'name':'reduce angular movement','fn':lambda x: min(x, key=lambda y: (y[1][0][4]))},

	# 3 - best prediction to reward
	{'name':'best prediction reward', 'fn':lambda x: max(x, key=itemgetter(2))},

	# 4 - gain altitude
	{'name':'gain altitude', 'fn':lambda x: max(x, key=lambda y: y[1][0][1] * y[2])},

	# 5 - reduce x_dist
	{'name':'reduce X distance','fn':lambda x: min(x, key=lambda y: distance.euclidean((y[1][0][0]),(0)))},

	# 6 - max pos speed
	{'name':'max positive speed','fn':lambda x: max(x, key=lambda y: y[1][0][3])},

	# 7 - max height and max speed
	{'name':'max height and max speed','fn':lambda x: max(x, key=lambda y: y[1][0][1]**2 + magn(y[1][0][2],y[1][0][3]))},

	# 8 - min rotation
	{'name':'min rotation','fn':lambda x: min(x, key=lambda y: distance.euclidean((y[1][0][5]),(0)))},

	# 9 - max rotation
	{'name':'max rotation', 'fn':lambda x: max(x, key=lambda y: distance.euclidean((y[1][0][5]),(0)))},

	# 10 - do nothing
	{'name':'do nothing', 'fn':lambda x: [0,np.zeros((1,8)),[-np.inf]]}, 

	# 11 - min rotation min speed
	{'name':'min rot min speed' ,'fn': lambda x: min(x, key=lambda y: (5. * np.abs(y[1][0][3])) + ((5. * np.abs(y[1][0][5]))+ (5. * y[1][0][4])) )}, 

	# 12 - reduce general speed and rot
	{'name':'reduce general speed and rot','fn':lambda x: min(x, key=lambda y: 0.1 * distance.euclidean((y[1][0][2],y[1][0][3]),(0,0)) + (5. * np.abs(y[1][0][5]) )+ (5. * y[1][0][4]) )},	

	# 13 - go high and center
	{'name':'go high and center', 'fn':lambda x: min(x, key=lambda y: -(y[1][0][0] + y[1][0][1]) + 0.01 * distance.euclidean((y[1][0][5]),(0)))},

	# 14 - zero mov
	{'name':'zero mov', 'fn':lambda x: min(x, key=lambda y: distance.euclidean((y[1][0][0],y[1][0][1]),(0,1)) + distance.euclidean((y[1][0][2],y[1][0][3],y[1][0][4]),(0,0,0)))},	
        
    # 15 - RAC
    {'name':'RAC','fn':lambda x: min(x, key=lambda y: 0.2 * euc(y[1][0][2],y[1][0][3]) + (1-(1./np.dot((y[1][0][2],y[1][0][3])/np.linalg.norm((y[1][0][2],y[1][0][3])),np.array((0,1))))) )},

    #16 - Slow Fall
    {'name':'Slow Fall','fn':lambda x: min(x, key=lambda y: float(step_count)/250. * euc(y[1][0][0],y[1][0][1]) + float(step_count)/50. * np.abs(y[1][0][3])/(np.abs(y[1][0][1])) + float(step_count)/100 * np.abs(y[1][0][0]) + float(step_count)/100. * np.abs(y[1][0][4]) + float(step_count)/20. * np.abs(y[1][0][5])) },

]


def euc(a,b):
	return distance.euclidean((a,b),(0,0))

def euc2(a,b,c,d):
	return distance.euclidean((a,b),(c,d))

def magn(a,b):
	return np.linalg.norm((a,b))

def selector(x, state):
	step = step_count

	if state.shape == (1,8):
		state = state[0]
	
	y_speed = state[3]
	speed = magn(state[2],state[3])
	height = state[1]
	angle = state[4]
	angle_sp = state[5]
	dist_target = distance.euclidean((state[0],state[1]),(0,0))
	x_dist = magn(state[0],0)
	left_touch_val = state[6]
	right_touch_val = state[7]
	left_touch = left_touch_val >= 0.1
	right_touch = right_touch_val >= 0.1

	if height > 0.5:
		# altitude phase
		if np.abs(angle) > 0.1:
			i = 8
		else:
			if y_speed <= -0.2:
				i = 15
			elif np.abs(angle) > 0.5:
				i = 8 
			elif x_dist > 2:
				i = 5
			else:
				i = 15
	else:
		if height > 0.3:
			# Approximation
			if speed > 0.1:
				i = 15
			elif np.abs(angle) > 0.5:
				i = 8 
			elif x_dist > 0.3:
				i = 5
			else:
				i = 3
		else:
			# Landing
			if right_touch and left_touch and speed <= 0.05:
					i = 10
			elif speed >= 0.05:
				i = 14
			else:

				if np.abs(angle) > 0.03 and speed > 0.1:
					i = 8
				elif x_dist > 0.3:
					i = 15
				elif right_touch and left_touch:
					i = 10
				elif ( speed > 0 or height <= 0.02) and (left_touch or right_touch):
					i = 10
				elif dist_target <= 0.1:
					i = 10
				else:
					i = 15


	n = strategies[i]['fn'](x)
	
	if GRAPH:
		strategy_history.append(i)

 	return n


while True:
	if P.replan:
		P.setPlan(observation, selector)

	if LEARNING:
		state_data = {'observation': copy.copy(observation)}
		reward_data = {'observation': copy.copy(observation)}

	step = P.step()

	if LEARNING:
		state_data['action'] = step
		reward_data['action'] = step

	observation, reward, done, info = env.step(step)

	if LEARNING:
		state_data['next_state'] = copy.copy(observation)
		reward_data['reward'] = reward

	total_reward += reward
	step_count += 1

	if LEARNING:
		P.dump(state_data, reward_data)

	if VERBOSE:
		print {'reward': reward, 'done':done, 'step':step_count}

	if GRAPH:
		rewards_points.append(reward)
		total_rewards_points.append(total_reward)

	if DISPLAY:
		env.render()

	if done:
		print "Finished in %d steps - total reward: %f" % (step_count, total_reward)
		total_reward_history.append(total_reward)
		
		if total_reward >= 0:
			success_landings += 1
		else:
			fail_landings += 1

		total_reward = 0
		step_count = 0

		if FINAL_GRAPH > 0:
			fpx, fpy = observation[0] , observation[1]
			afpx = int((fpx + 1) * 10.)
			afpy = 12-int((fpy + 0.2) * 10.)
			if afpx >= 22:
				afpx = 22
			elif afpx < 0:
				afpx = 0
			datapos_grid[afpy][afpx] += 1


		if GRAPH:
			update_graphs(fig, ax, canvas, strategy_history, strategies, rewards_points)
			time.sleep(1)
			rewards_points = []
			total_rewards_points = []
			strategy_history = []
			speed_history = []

		observation = env.reset()

	if not running:
		break

P.close()

print "Sucess landing: ", success_landings
print "Fail landing: ", fail_landings
print "Average Total Score: ", np.average(total_reward_history)

print 'FINAL_GRAPH', int(FINAL_GRAPH)
if int(FINAL_GRAPH) > 0:
	df = pd.DataFrame()
	df['total_reward_history'] = total_reward_history
	fig2, ax2 = plt.subplots(1,1)
	fig3, ax3 = plt.subplots(1,1)
	ax2.clear()
	ax3.clear()
	sns.factorplot(kind='box', data=df, x='total_reward_history', ax=ax2)
	sns.heatmap(datapos_grid,cmap='YlGnBu',ax=ax3)

	plt.setp( ax3.xaxis.get_majorticklabels(), rotation=70 )
	plt.setp( ax3.yaxis.get_majorticklabels(), rotation=-45 )
	ax3.set_xticklabels(
		[ str(x) for x in [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
		 -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
		 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]])
	ax3.set_yticklabels(
		[ str(x) for x in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5,
		 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2]])	
	plt.tight_layout()
	if int(FINAL_GRAPH) == 1:
		plt.show()
	elif int(FINAL_GRAPH) == 2:
		fig2.savefig('boxplot.png')
		fig3.savefig('heatmap.png')


if LEARNING:
	o = ObservationLearner('observation-record.pkl')
	data = o.split(0.75)
	o.train(data[0],data[1])
	o.test(data[2],data[3])
	o.save('physics.pkl')

	r = RewardLearner('rewards-record.pkl')
	data = r.split(0.75)
	r.train(data[0],data[1])
	r.test(data[2],data[3])
	r.save('rewards.pkl')
	
