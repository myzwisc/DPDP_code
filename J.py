import logit as logit
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

attack_goal = sys.argv[1].rstrip('/')
k = int(sys.argv[4])
method = sys.argv[3]

Reg = 10	# additional second order term
noise_mode = 'Lap'
Ta = 2000
gapT = 20
N = Ta/gapT+1

EPS = sys.argv[2]
eps = float(sys.argv[2])

victim = logit.DPlogitobj(Reg, eps)

Dir = 'eps'+EPS+'/'+attack_goal+'/'+str(k)+'/'+method+'/'

f = open(Dir+'X.txt', 'r')
X = np.loadtxt(f)
f = open(Dir+'y.txt', 'r')
y = np.loadtxt(f)

Xp_traj = {}
for i in range(1,N):
	t = i*gapT
	f = open(Dir+str(t)+'.txt', 'r')
	Xp_traj[t] = np.loadtxt(f)

eval_dis = 'Lap'
Te = 1000
zero_one = 0

cost = [0 for i in range(N)]
std = [0 for i in range(N)]

if attack_goal == 'labelaversion':
	attacker = logit.attacker_obj_labelaversion(victim)
	f = open(Dir+'Xeval.txt', 'r')
	Xeval = np.loadtxt(f)
	f = open(Dir+'yeval.txt', 'r')
	yeval = np.loadtxt(f)
	cost[0],std[0] = attacker.getCost(X, y, Xeval, yeval,eval_dis, zero_one, Te)
	for i in range(1,N):
		t = i*gapT
		print(t)
		cost[i],std[i] = attacker.getCost(Xp_traj[t], y, Xeval, yeval,eval_dis, zero_one, Te)

elif attack_goal == 'labeltargeting':
	attacker = logit.attacker_obj_labeltargeting(victim)
	f = open(Dir+'Xeval.txt', 'r')
	Xeval = np.loadtxt(f)
	f = open(Dir+'yeval.txt', 'r')
	yeval = np.loadtxt(f)
	cost[0],std[0] = attacker.getCost(X, y, Xeval, yeval,eval_dis, zero_one, Te)
	for i in range(1,N):
		t = i*gapT
		print(t)
		cost[i],std[i] = attacker.getCost(Xp_traj[t], y, Xeval, yeval,eval_dis, zero_one, Te)

elif attack_goal == 'parametertargeting':
	attacker = logit.attacker_obj_parametertargeting(victim)
	f = open(Dir+'theta_star.txt', 'r')
	theta_star = np.loadtxt(f)
	cost[0],std[0] = attacker.getCost(X, y, theta_star, eval_dis, Te)
	for i in range(1,N):
		t = i*gapT
		print(t)
		cost[i],std[i] = attacker.getCost(Xp_traj[t], y, theta_star, eval_dis, Te)

outDir = 'J/'+attack_goal+'/'
if not os.path.exists(outDir):
	os.makedirs(outDir)
f = open(outDir + 'cost.txt', 'w')
np.savetxt(f,cost)
f.close()

f = open(outDir + 'std.txt', 'w')
np.savetxt(f,std)
f.close()

fig, ax = plt.subplots()
x = [0]+[i*gapT for i in range(1,N)]
plt.plot(x, cost, linestyle='-', color='forestgreen', label = r'$\hat J(\tilde D)$')
error = np.array([2*std[i] for i in range(N)])
plt.fill_between(x, cost-error, cost+error, alpha =0.3,color='forestgreen')
plt.legend(fontsize=12)

plt.grid()
plt.title(attack_goal+' attack')
plt.tight_layout()
plt.show()