import logit as logit
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist
import sys
import time
import copy as cop
import os

np.random.seed(1)

attack_goal = sys.argv[1].rstrip('/')
allgoals=['labelaversion','labeltargeting','parametertargeting']
if attack_goal not in allgoals:
	raise('attack goal must be labelaversion, labeltargeting, or parametertargeting')

# used for generate training uniformly-distributed data pointsin a unit ball. Irrelevant 
# of attack
def generateUniform(n,d):
	X = np.zeros((n,d))
	for i in range(n):
		while True:
			x = np.random.random(d)*2-1
			if np.linalg.norm(x)<=1:
				X[i,:]=x
				break
			else:
				continue
	return X

# used for generate equally-spaced data points in a unit ball. Irrelevant Irrelevant 
# of attack
def generateGrid(gap,d):
	ndim = int(2.0/gap+1)
	grid = np.zeros((np.power(ndim,d),d))
	grid_dim = np.linspace(-1,1,ndim)
	g = np.meshgrid(*(grid_dim,)*d)
	for i in range(d):
		temp = g[i].reshape(-1,1)
		grid[:,i] = temp[:,0]
	return grid

# load evaluation set and training data
f = open('data/Xeval.txt', 'r')
Xeval = np.loadtxt(f)
f = open('data/yeval.txt', 'r')
yeval = np.loadtxt(f)

f = open('data/X.txt', 'r')
X = np.loadtxt(f)
f = open('data/y.txt', 'r')
y = np.loadtxt(f)
n = len(y)

# define the victim
Reg = 10
eps = float(sys.argv[2])
victim = logit.DPlogitobj(Reg, eps)

# attack parameter
method = sys.argv[3]
if method == 'shallow-SV' or method =='deep-SV':
	bdis = 'None'
elif method == 'shallow-DPV' or method =='deep-DPV':
	bdis = 'Lap'
else:
	raise('unknown attack method')
k = int(sys.argv[4])
Ta = 5000
Tcheck = 2000
alpha_check = 0.0001
alpha_attack = 0.0
eta = 1.0

Dir = 'eps'+str(eps)+'/'+attack_goal+'/'+str(k)+'/'+method+'/'
if not os.path.exists(Dir):
    os.makedirs(Dir)

if attack_goal == 'labelaversion':
	attacker = logit.attacker_obj_labelaversion(victim)
	if k == n:
		attackID = [i for i in range(n)]
	elif method == 'shallow-SV' or method == 'shallow-DPV':
		attackID = attacker.getattackID(X,y,Xeval,yeval,k,bdis)
	else:
		checkID = [i for i in range(n)]
		_,Xp_check = attacker.attack(X,y,checkID,Xeval,yeval,bdis,Tcheck,alpha_check,eta)
		norm_change = np.sum(np.abs(Xp_check-X)**2,axis=1)/2
		Order = np.argsort(norm_change)
		attackID = Order[n-k:]
	start_time = time.time()
	Xp_traj,Xp = attacker.attack(X,y,attackID,Xeval,yeval,bdis,Ta,alpha_attack,eta)
	end_time = time.time()
	runtime = end_time - start_time

elif attack_goal == 'labeltargeting':
	attacker = logit.attacker_obj_labeltargeting(victim)
	if k == n:
		attackID = [i for i in range(n)]
	elif method == 'shallow-SV' or method == 'shallow-DPV':
		attackID = attacker.getattackID(X,y,Xeval,yeval,k,bdis)
	else:
		checkID = [i for i in range(n)]
		_,Xp_check = attacker.attack(X,y,checkID,Xeval,yeval,bdis,Tcheck,alpha_check,eta)
		norm_change = np.sum(np.abs(Xp_check-X)**2,axis=1)/2
		Order = np.argsort(norm_change)
		attackID = Order[n-k:]

	start_time = time.time()
	Xp_traj,Xp = attacker.attack(X,y,attackID,Xeval,yeval,bdis,Ta,alpha_attack,eta)
	end_time = time.time()
	runtime = end_time - start_time

elif attack_goal == 'parametertargeting':
	attacker = logit.attacker_obj_parametertargeting(victim)
	theta_star = attacker.victim.logitobj(Xeval,yeval,'None')
	if k == n:
		attackID = [i for i in range(n)]
	elif method == 'shallow-SV' or method == 'shallow-DPV':
		attackID = attacker.getattackID(X,y,theta_star,k,bdis)
	else:
		checkID = [i for i in range(n)]
		_,Xp_check = attacker.attack(X,y,checkID,theta_star,bdis,Tcheck,alpha_check,eta)
		norm_change = np.sum(np.abs(Xp_check-X)**2,axis=1)/2
		Order = np.argsort(norm_change)
		attackID = Order[n-k:]

	start_time = time.time()
	Xp_traj,Xp = attacker.attack(X,y,attackID,theta_star,bdis,Ta,alpha_attack,eta)
	end_time = time.time()
	runtime = end_time - start_time
else:
	raise('unknown attack goal')

savedata = 1
gapT = 20
N = Ta/gapT+1
if savedata:
	for i in range(1,N):
		t = i*gapT
		f = open(Dir+str(t)+'.txt', 'w')
		np.savetxt(f, Xp_traj[t])

	f = open(Dir+'X'+'.txt', 'w')
	np.savetxt(f, X)
	f = open(Dir+'y'+'.txt', 'w')
	np.savetxt(f, y)
	f = open(Dir+'time'+'.txt', 'w')
	np.savetxt(f, [runtime])

	if attack_goal == 'labelaversion':
		f = open(Dir+'Xeval'+'.txt', 'w')
		np.savetxt(f, Xeval)
		f = open(Dir+'yeval'+'.txt', 'w')
		np.savetxt(f, yeval)
	elif attack_goal == 'labeltargeting':
		f = open(Dir+'Xeval'+'.txt', 'w')
		np.savetxt(f, Xeval)
		f = open(Dir+'yeval'+'.txt', 'w')
		np.savetxt(f, yeval)
	elif attack_goal == 'parametertargeting':
		f = open(Dir+'theta_star'+'.txt', 'w')
		np.savetxt(f, theta_star)