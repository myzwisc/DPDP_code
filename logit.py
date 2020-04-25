import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import copy as cop

# -----------------------------
# Define differential private leaner
class DPlogitobj():
	"""docstring for self.victim.logitobj"""
	def __init__(self, Reg, eps):
		self.Reg = Reg
		self.eps = eps
		self.lam = 1.0/4
		self.zeta = 1.0
		self.checkReg()

	def checkReg(self):
		if self.Reg < 2*self.lam/self.eps:
			print('warning: regularization weight not set properly')

	def logitobj(self, X, y, DPtype):
		n, d = X.shape
		while True:
			if DPtype=='Lap':
				loc = np.array([0]*d)
				scale = 2 * self.zeta/self.eps
				b = np.random.laplace(loc=loc,scale=scale)
			elif DPtype == 'Gau':
				delta = 1.0/n
				mean = np.array([0]*d)
				cov = np.eye(d)*np.power(self.zeta,2)*(8*np.log(2.0/delta)+4*self.eps)/np.power(self.eps,2)
				b = np.random.multivariate_normal(mean, cov)
			elif DPtype == 'None':
				b = np.array([0]*d)
			else:
				raise ValueError('unknown mode!')
			# run dp learner
			v = cp.Variable(d)
			X_signed = np.multiply(X,-np.reshape(y,(n,1)))
			objective = cp.Minimize(cp.sum(cp.logistic(X_signed*v))+self.Reg*cp.sum_squares(v)/2+b*v)
			prob = cp.Problem(objective)
			try:
				result = prob.solve()
				break
			except:
				try:
					result = prob.solve(solver='SCS')
					print('using a different solver now...')
					break
				except:
					print('both solvers failed, retrying...')
					continue
		theta = v.value
		return theta

class attacker_obj_labelaversion(object):
	def __init__(self, victim):
		self.victim = victim
		self.eps = victim.eps
		self.lam = victim.lam
		self.zeta = victim.zeta
		self.Reg = victim.Reg

	def getCost(self,X,y,Xval,yval,DPtype,zero_one,T):
		cost_hat = [0] * T
		nval = len(yval)
		for t in range(T):
			theta = self.victim.logitobj(X, y, DPtype)
			C = 0
			for i in range(nval):
				si = theta.dot(Xval[i,:])*yval[i]
				if si<=0 and zero_one:
					C = C - 1
				else:
					C = C - np.log(1+np.exp(-si))
			cost_hat[t] = float(C)/nval
		cost = np.mean(cost_hat)
		std = np.std(cost_hat) / np.sqrt(T)
		return cost, std

	def getattackID(self,X,y,Xval,yval,k,DPtype):
		ID = []
		if DPtype == 'None':
			T = 1
		else:
			T = 1000
		n, d = X.shape
		nval = len(yval)
		Grad_avg = np.zeros((n,d))
		score = [0 for i in range(n)]

		for t in range(T):
			theta = self.victim.logitobj(X,y,DPtype)
			deri_theta = [0 for i in range(d)]

			sval = [0] * nval
			for i in range(nval):
				sval[i] = np.exp(-yval[i]*Xval[i,:].dot(theta))
			for i in range(nval):
				deri_theta = deri_theta + (sval[i]/(1+sval[i]))*yval[i]*Xval[i,:]/nval

			s = [0] * n
			for j in range(n):
				xj = np.transpose(X[j,:])
				yj = y[j]
				s[j] = np.exp(yj*theta.dot(xj))
			S = np.diag([s[j]/np.power(1+s[j],2) for j in range(n)])
			temp1 = np.linalg.inv(self.Reg*np.eye(d)+np.transpose(X).dot(S).dot(X))

			Grad = np.zeros((n,d))
			for j in range(n):
				xj = X[j,:]
				yj = y[j]
				sj = s[j]
				temp2 = (yj/(1+sj))*np.eye(d)-sj*np.outer(theta, xj)/np.power(1+sj,2)
				Grad[j,:] = temp2.dot(temp1).dot(deri_theta)
			Grad_avg = Grad_avg+Grad/T
		for j in range(n):
			score[j] = np.linalg.norm(Grad_avg[j,:])
		ID = np.argsort(score)[n-k:]
		return ID

	def attack(self,X,y,attackID,Xval,yval,DPtype,Ta,alpha,eta):
		n, d = X.shape
		print('Start running label-aversion attack...')

		nval = len(yval)

		Xp_traj = {}
		Xp_traj[0] = X
		Xt = cop.deepcopy(X)
		for t in range(Ta):
			print(t)
			# theta = self.victim.logitobj(Xt, y, 'None')
			# loss = 0
			# for i in range(nval):
			# 	loss = loss - np.log(1+np.exp(-yval[i]*Xval[i,:].dot(theta)))/nval
			# print(loss,t)

			theta = self.victim.logitobj(Xt, y, DPtype)
			
			sval = [0] * nval
			deri_theta = [0 for i in range(d)]
			for i in range(nval):
				sval[i] = np.exp(-yval[i]*Xval[i,:].dot(theta))
			for i in range(nval):
				deri_theta = deri_theta + (sval[i]/(1+sval[i]))*yval[i]*Xval[i,:]/nval
			
			s = [0] * n
			for j in range(n):
				xj = Xt[j,:]
				yj = y[j]
				s[j] = np.exp(yj*theta.dot(xj))
			S = np.diag([s[j]/np.power(1+s[j],2) for j in range(n)])
			temp1 = np.linalg.inv(self.Reg*np.eye(d)+np.transpose(Xt).dot(S).dot(Xt))

			Grad = np.zeros((n,d))
			for j in attackID:
				xj = np.transpose(Xt[j,:])
				yj = y[j]
				sj = s[j]
				temp2 = (yj/(1+sj))*np.eye(d)-sj*np.outer(theta, xj)/np.power(1+sj,2)
				Grad[j,:] = temp2.dot(temp1).dot(deri_theta)+alpha*(xj-X[j,:])

			Xt = Xt - eta * Grad
			rownorm = np.sum(np.abs(Xt)**2,axis=1)**0.5
			scale = [[1] if rownorm[i]<1 else [rownorm[i]] for i in range(n)]
			Xt = np.divide(Xt, scale)
			Xp_traj[t+1]=Xt
		return Xp_traj,Xt

class attacker_obj_labeltargeting(object):
	def __init__(self, victim):
		self.victim = victim
		self.eps = victim.eps
		self.lam = victim.lam
		self.zeta = victim.zeta
		self.Reg = victim.Reg

	def getCost(self,X,y,Xtar,ytar,DPtype,zero_one,T):
		cost_hat = [0] * T
		ntar = len(ytar)
		for t in range(T):
			theta = self.victim.logitobj(X, y, DPtype)
			C = 0
			for i in range(ntar):
				si = theta.dot(Xtar[i,:])*ytar[i]
				if si<=0 and zero_one:
					C = C+ 1
				else:
					C = C + np.log(1+np.exp(-si))
			cost_hat[t] = float(C)/ntar
		cost = np.mean(cost_hat)
		std = np.std(cost_hat) / np.sqrt(T)
		return cost, std

	def getattackID(self,X,y,Xtar,ytar,k,DPtype):
		ID = []
		if DPtype == 'None':
			T = 1
		else:
			T = 1000
		n, d = X.shape
		ntar = len(ytar)
		Grad_avg = np.zeros((n,d))
		score = [0 for i in range(n)]

		for t in range(T):
			theta = self.victim.logitobj(X,y,DPtype)
			deri_theta = [0 for i in range(d)]

			star = [0] * ntar
			deri_theta = [0 for i in range(d)]
			for i in range(ntar):
				star[i] = np.exp(-ytar[i]*Xtar[i,:].dot(theta))
			for i in range(ntar):
				deri_theta = deri_theta - (star[i]/(1+star[i]))*ytar[i]*Xtar[i,:]/ntar
			
			s = [0] * n
			for j in range(n):
				xj = X[j,:]
				yj = y[j]
				s[j] = np.exp(yj*theta.dot(xj))
			S = np.diag([s[j]/np.power(1+s[j],2) for j in range(n)])
			temp1 = np.linalg.inv(self.Reg*np.eye(d)+np.transpose(X).dot(S).dot(X))

			Grad = np.zeros((n,d))
			for j in range(n):
				xj = X[j,:]
				yj = y[j]
				sj = s[j]
				temp2 = (yj/(1+sj))*np.eye(d)-sj*np.outer(theta, xj)/np.power(1+sj,2)
				Grad[j,:] = temp2.dot(temp1).dot(deri_theta)
			Grad_avg = Grad_avg+Grad/T
		for j in range(n):
			score[j] = np.linalg.norm(Grad_avg[j,:])
		ID = np.argsort(score)[n-k:]
		return ID

	def attack(self,X,y,attackID,Xtar,ytar,DPtype,Ta,alpha,eta):
			n, d = X.shape
			print('Start running label-targeting attack...')
			ntar = len(ytar)

			Xp_traj = {}
			Xp_traj[0] = X
			Xt = cop.deepcopy(X)
			for t in range(Ta):
				print(t)
				# theta = self.victim.logitobj(Xt, y, 'None')
				# loss = 0
				# for i in range(ntar):
				# 	loss = loss + np.log(1+np.exp(-ytar[i]*Xtar[i,:].dot(theta)))/ntar
				# print(loss,t)

				theta = self.victim.logitobj(Xt, y, DPtype)

				star = [0] * ntar
				deri_theta = [0 for i in range(d)]
				for i in range(ntar):
					star[i] = np.exp(-ytar[i]*Xtar[i,:].dot(theta))
				for i in range(ntar):
					deri_theta = deri_theta - (star[i]/(1+star[i]))*ytar[i]*Xtar[i,:]/ntar
				
				s = [0] * n
				for j in range(n):
					xj = Xt[j,:]
					yj = y[j]
					s[j] = np.exp(yj*theta.dot(xj))
				S = np.diag([s[j]/np.power(1+s[j],2) for j in range(n)])
				temp1 = np.linalg.inv(self.Reg*np.eye(d)+np.transpose(Xt).dot(S).dot(Xt))

				Grad = np.zeros((n,d))
				for j in attackID:
					xj = np.transpose(Xt[j,:])
					yj = y[j]
					sj = s[j]
					temp2 = (yj/(1+sj))*np.eye(d)-sj*np.outer(theta, xj)/np.power(1+sj,2)
					Grad[j,:] = temp2.dot(temp1).dot(deri_theta)+alpha*(xj-X[j,:])

				Xt = Xt - eta * Grad
				rownorm = np.sum(np.abs(Xt)**2,axis=1)**0.5
				scale = [[1] if rownorm[i]<1 else [rownorm[i]] for i in range(n)]
				Xt = np.divide(Xt, scale)
				Xp_traj[t+1]=Xt
			return Xp_traj,Xt

class attacker_obj_parametertargeting(object):
	"""docstring for attacker_obj_parametertargeting"""
	def __init__(self, victim):
		self.victim = victim
		self.eps = victim.eps
		self.lam = victim.lam
		self.zeta = victim.zeta
		self.Reg = victim.Reg

	def getCost(self, X, y, theta_star, DPtype, T):
		cost_hat = [0 for t in range(T)]
		for t in range(T):
			theta = self.victim.logitobj(X, y, DPtype)
			cost_hat[t] = np.power(np.linalg.norm(theta - theta_star),2) / 2
		cost = np.mean(cost_hat)
		std = np.std(cost_hat)/np.sqrt(T)
		return cost, std

	def getattackID(self,X,y,theta_star,k,DPtype):
		ID = []
		if DPtype == 'None':
			T = 1
		else:
			T = 1000
		n, d = X.shape
		Grad_avg = np.zeros((n,d))
		score = [0 for i in range(n)]

		for t in range(T):
			theta = self.victim.logitobj(X,y,DPtype)
			diff = [theta[i]-theta_star[i] for i in range(d)]
			s = [0] * n
			for j in range(n):
				xj = np.transpose(X[j,:])
				yj = y[j]
				s[j] = np.exp(yj*theta.dot(xj))
			S = np.diag([s[j]/np.power(1+s[j],2) for j in range(n)])
			temp1 = np.linalg.inv(self.Reg*np.eye(d)+np.transpose(X).dot(S).dot(X))

			Grad = np.zeros((n,d))
			for j in range(n):
				xj = X[j,:]
				yj = y[j]
				sj = s[j]
				temp2 = (yj/(1+sj))*np.eye(d)-sj*np.outer(theta, xj)/np.power(1+sj,2)
				Grad[j,:] = temp2.dot(temp1).dot(diff)
			Grad_avg = Grad_avg+Grad/T
		for j in range(n):
			score[j] = np.linalg.norm(Grad_avg[j,:])
		ID = np.argsort(score)[n-k:]
		return ID

	def attack(self,X,y,attackID,theta_star,DPtype,Ta,alpha,eta):
		n, d = X.shape
		print('Start running parameter-targeting attack...')

		Xp_traj = {}
		Xp_traj[0] = X
		Xt = cop.deepcopy(X)

		for t in range(Ta):
			print(t)

			theta = self.victim.logitobj(Xt,y,DPtype)

			diff = [theta[i]-theta_star[i] for i in range(d)]
			s = [0] * n
			for j in range(n):
				xj = np.transpose(Xt[j,:])
				yj = y[j]
				s[j] = np.exp(yj*theta.dot(xj))
			S = np.diag([s[j]/np.power(1+s[j],2) for j in range(n)])
			temp1 = np.linalg.inv(self.Reg*np.eye(d)+np.transpose(Xt).dot(S).dot(Xt))

			Grad = np.zeros((n,d))
			for j in attackID:
				xj = Xt[j,:]
				yj = y[j]
				sj = s[j]
				temp2 = (yj/(1+sj))*np.eye(d)-sj*np.outer(theta, xj)/np.power(1+sj,2)
				Grad[j,:] = temp2.dot(temp1).dot(diff)+alpha*(xj-X[j,:])

			Xt = Xt - eta * Grad
			rownorm = np.sum(np.abs(Xt)**2,axis=1)**0.5
			scale = [[1] if rownorm[i]<1 else [rownorm[i]] for i in range(n)]
			Xt = np.divide(Xt, scale)
			
			Xp_traj[t+1]=Xt
		return Xp_traj,Xt