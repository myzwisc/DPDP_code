import logit as logit
import numpy as np
import sys
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

attack_goal = sys.argv[1].rstrip('/')
k = int(sys.argv[4])
method = sys.argv[3]

Reg = 10	# additional second order term
noise_mode = 'Lap'
Ta = 5000
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

if attack_goal == 'labelaversion':

	f = open(Dir+'Xeval.txt', 'r')
	Xeval = np.loadtxt(f)
	f = open(Dir+'yeval.txt', 'r')
	yeval = np.loadtxt(f)
	Xp = Xp_traj[Ta]

elif attack_goal == 'labeltargeting':

	f = open(Dir+'Xeval.txt', 'r')
	Xeval = np.loadtxt(f)
	f = open(Dir+'yeval.txt', 'r')
	yeval = np.loadtxt(f)
	Xp = Xp_traj[Ta]

elif attack_goal == 'parametertargeting':
	attacker = logit.attacker_obj_parametertargeting(victim)
	f = open(Dir+'theta_star.txt', 'r')
	theta_star = np.loadtxt(f)
	f = open(Dir+'Xeval.txt', 'r')
	Xeval = np.loadtxt(f)
	f = open(Dir+'yeval.txt', 'r')
	yeval = np.loadtxt(f)
	Xp = Xp_traj[Ta]
	cost,_= attacker.getCost(Xeval,yeval,theta_star,eval_dis,Te)
	print('cost of eval data:', cost)
	cost,_= attacker.getCost(Xp,y,theta_star,eval_dis,Te)
	print('cost of poisoned data:', cost)

plt.figure()
n = len(y)
I_neg = []
I_pos = []
for i in range(n):
	if y[i]==-1:
		I_neg.append(i)
	else:
		I_pos.append(i)
plt.plot(X[I_pos,0],X[I_pos,1],'.',color='b',alpha=0.2)
plt.plot(X[I_neg,0],X[I_neg,1],'.',color='r',alpha=0.2)

plt.plot(Xp[I_pos,0],Xp[I_pos,1],'.',color='b')
plt.plot(Xp[I_neg,0],Xp[I_neg,1],'.',color='r')

for i in range(n):
	trajx = [X[i,0]]+[Xp_traj[j*gapT][i,0] for j in range(1,N)]
	trajy = [X[i,1]]+[Xp_traj[j*gapT][i,1] for j in range(1,N)]
	if y[i]==1:
		plt.plot(trajx,trajy,color='b', alpha=0.1)
	else:
		plt.plot(trajx,trajy,color='r', alpha=0.1)

plt.xlim([-1.1,1.1])
plt.ylim([-1.1,1.1])
plt.tick_params(axis='both', labelsize=20)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.show()
