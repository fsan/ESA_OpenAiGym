# This is a demonstration for proof of concept of the Enhanced Subsumption Architecture
# This is the Automatic version referred in the article.

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


MAGIC = True
found_count = 0

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

if GRAPH or MAGIC:
	rewards_points = []
	total_rewards_points = []
	strategy_history = []
	speed_history = []

	plt.ioff()
	fig, ax = plt.subplots(3,1, figsize=(7,12))
	plt.show(block=False)

if FINAL_GRAPH > 0:
	datapos_grid = np.zeros(shape=(13,22))

def update_graphs(fig, ax, canvas, strategy_history, strategies, rewards_points, found_count, show=True,save=True):

	ax[0].clear()
	ax[1].clear()
	ax[2].clear()

	m = min(len(rewards_points), len(strategy_history))
	
	rewards_points = rewards_points[:m]
	strategy_history = strategy_history[:m]

	if show:
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
		if not save:
			canvas.draw()
		else:
			fig.savefig('found' + str(found_count) + '.png')

if GRAPH or MAGIC:
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
	{'name':'Slow Fall', 'weight':1.1 , 'fn':lambda x:min(x, key=lambda y: np.abs(y[1][0][5])+np.abs(y[1][0][4]) + 10 * np.abs(y[1][0][0]) + np.abs(y[1][0][1]) + 0.9 * euc(y[1][0][2],y[1][0][3]) )},

	{'name':'RAC', 'weight':1.1 ,'fn':lambda x: min(x, key=lambda y: (1-(1./np.dot((y[1][0][2],y[1][0][3])/np.linalg.norm((y[1][0][2],y[1][0][3])),np.array((0,1))))) )},

	{'name':'RAV','weight':1.10 , 'fn':lambda x: min(x, key=lambda y: 0.5 * euc(y[1][0][0],y[1][0][1]) + np.abs(y[1][0][5]) + euc(y[1][0][3],y[1][0][3]))},

	{'name':'Zero Mov', 'weight':1.12, 'fn':lambda x: min(x, key=lambda y: distance.euclidean((y[1][0][0],y[1][0][1]),(0,1)) + distance.euclidean((y[1][0][2],y[1][0][3],y[1][0][5]),(0,0,0)))},		

	{'name':'Do Nothing', 'weight':1.,  'fn':lambda x: [0,np.zeros((1,8)),[-np.inf]]}, 

	{'name':'best prediction reward', 'weight':1.105, 'fn':lambda x: max(x, key=itemgetter(2))},
]


def euc(a,b):
	return distance.euclidean((a,b),(0,0))

def euc2(a,b,c,d):
	return distance.euclidean((a,b),(c,d))

def magn(a,b):
	return np.linalg.norm((a,b))

def selector(x, state):
	global plan_size
	step = step_count
	if GRAPH or MAGIC:
		global strategy_history

	if state.shape == (1,8):
		state = state[0]

	px,py = state[0],state[1]
	x_speed = state[2]
	y_speed = state[3]
	speed = magn(state[2],state[3])
	height = state[1]
	angle = state[4]
	angle_sp = state[5]
	d = euc(px,py)
	x_dist = state[0]
	left_touch_val = state[6]
	right_touch_val = state[7]
	left_touch = left_touch_val >= 0.7
	right_touch = right_touch_val >= 0.7
	touching = left_touch and right_touch

	x_limit = 0.5
	y_limit = 0.2
	a_limit = 0.5
	as_limit = 0.5
	s_limit = 0.2
	p_limit = 0.3
	stop_limit = 0.05

	if height <= y_limit and touching:#np.abs(x_dist)+y_limit <= stop_limit * 25. and touching:
		strat = {
					'name': 'Land Checker',
					'weight': 1.0,
					'fn':lambda x: [0,np.zeros((1,8)),[-np.inf]]
				}
	elif height <= y_limit:
		strat = {
			'name': 'landing',
			'weight': 1.0,
			'fn':lambda x: min(x, key=lambda y: (2./height) * np.abs((y_speed)-(2*y[1][0][3]))**2 + 80 * y[1][0][4] + (1./height) * np.abs(angle_sp-y[1][0][5]) + (1./height) * np.abs(angle))
		}				
	elif height >= y_limit and (np.abs(angle) > a_limit or np.abs(angle_sp) >= as_limit) and np.abs(angle) > p_limit:
		fn1 = lambda x: min(x, key=lambda y:
											1./np.abs(x_dist) + 
											1. * np.abs(y[1][0][2]) +	
											1.5 * np.abs(y[1][0][3]) +
											2.5 * np.cos(angle)**2 + 
											0.1 * np.sin((y[1][0][5] - np.abs(angle))*100.)
								)

		strat = {
			'name': 'Angle Stabilizer',
			'weight': 1.0,
			'fn':lambda x: min(x, key=lambda y: 1./np.abs(x_dist) + + 6 * np.abs(y[1][0][4]) +
			 				3 * distance.euclidean((y[1][0][0],y[1][0][1]),(0,1)) +
			  				6 * distance.euclidean((y[1][0][2],y[1][0][3]),(0,0)) + 
			  				3 * np.abs(y[1][0][5]) + 
			  				1./height * np.sin(y_speed)**2)
		}
	elif height >= y_limit and (np.abs(y_speed) > s_limit):
		strat = {
			'name': 'Speed Stabilizer',
			'weight': 1.0,
			'fn': lambda x: min(x, key=lambda y:
											1./np.abs(x_dist) +
											2. * np.abs(y[1][0][2]) +
											2. * np.abs(y[1][0][3])
								)
		}
	elif height >= y_limit and np.abs(x_dist) <= x_limit:
		strat = min(strategies, key=lambda c: meta(c,x))

	elif np.abs(x_dist) > x_limit:
		strat = {
			'name': 'MHD',
			'weight': 1.0,
			'fn': lambda x: min(x, key=lambda y:
											2 * np.sin(x_dist) + 
											np.abs(y_speed) +
											0.5 * np.abs(angle)
								)
		}
	else:
		strat = min(strategies, key=lambda c: meta(c,x))


	n = strat['fn'](x)
	
	if GRAPH or MAGIC:
		strategy_history.append(strat['name'])
		speed_history.append((state[2],state[3],angle_sp))

	return n


def meta(c,x):
	buf = c['fn'](x)
	pw = c['weight']

	pstate = buf[1][0]
	cstate = x[0][1][0]

	px, py = pstate[0], pstate[1]
	d = euc(px,py)
	y_speed = pstate[3]
	height = pstate[1]
	cheight = cstate[1]
	touching = (cstate[6] >= 0.7) and (cstate[7] >= 0.7)
	hw = (cheight)/(80.)

	dw = (d**2)*height*60.	

	if (c['name'] == 'do nothing') and touching:
		result = np.inf
	else:
		result = (hw * y_speed) + dw * d + (0.4 + cheight) * np.abs(px)

	return -result / pw

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

	if GRAPH or MAGIC:
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
			(fig, ax, canvas, strategy_history, strategies, rewards_points, found_count)
			time.sleep(1)
			rewards_points = []
			total_rewards_points = []
			strategy_history = []
			speed_history = []

		elif MAGIC:
			if len(total_rewards_points) > 0 and total_rewards_points[-1] >= 0:
				STOP = True
				found_count += 1
			else:
				STOP = False
			update_graphs(fig, ax, canvas, strategy_history, strategies, rewards_points, found_count, show=STOP,save=STOP)
			rewards_points = []
			total_rewards_points = []
			strategy_history = []
			speed_history = []
			if STOP:
				print "FOUND"

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
		[ str(x) for x in np.arange(-1.0,1.1,0.1)])
	ax3.set_yticklabels(
		[ str(x) for x in np.arange(1.0,-0.3,-0.1)])	
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
	
