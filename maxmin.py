from cvxpy import *
import cvxpy as cp
from math import sqrt
import numpy as np
import pdb
from matplotlib import pyplot as plt
#pdb.pm()

def maxmin(V,x):
	utilities = V*x
	u_min = utilities.sum(0).min()
	return(u_min)

def uniform_allocation(V):
	(T,n) = V.shape
	x_uniform = V*0+1.0/n
	return(x_uniform)

def stupid_example(n):
	V=np.zeros([n**2,n])
	for i in range(0,n):
		V[i*n:(i+1)*n,i] = np.ones(n)*1.0/n
	return(V)

def uniform(n,T):
	V = np.ones([T,n])/T
	return(V)

def example1(n):
	V=np.zeros([n,n])
	for i in range(0,n):
		V[i,:] = np.ones(n)*(1-1/sqrt(n))/(n-1)
		V[i,i] = 1/sqrt(n)
	return(V)

def random_instance(n,T):
	V = np.random.uniform(0,1,[T,n])
	#V = (V>0.8).astype(float)
	for i in range(0,n):
		V[:,i] = V[:,i]*1.0/np.sum(V[:,i])
	return(V)


def solve_opt(V,u):
	(n,m) = V.shape
	x = Variable(V.shape)
	#temp1 = sum((multiply(V,x)),0)
	#temp = min(sum((multiply(V,x)),0))
	objective = Maximize(min(u+sum((multiply(V,x)),0)));
	constraints = [ sum(x,1) <= 1]
	constraints.append(x>=0)
	problem = Problem(objective,constraints)	
	problem.solve()
	#print(problem.status)
	#print("optimal solution")
	#print(np.round(x.value,2))
	return(x.value,problem.value)
def solve_dual(V):
	(T,n) = V.shape
	y = Variable(T)
	la = Variable(n)
	objective = Minimize(sum(y))
	constraints = [sum(la)>=1.0]
	for t in range(T):
		for i in range(n):
			constraints.append(y[t]>=la[i]*V[t,i])
	problem = Problem(objective,constraints)
	problem.solve()
	return(la.value,problem.value,y.value)

def optimistpessimist(V,alpha):
	# alpha decides the mix between optimist and pessimist
	x_final = V*0
	(T,n) = V.shape
	u_current = np.zeros(n)
	for t in range(T):
		# construct 2 scenarios
		v_t = V[t]
		v_rem = V[(t+1):].sum(0)
		V_optimist = resolve(v_t,v_rem)
		V_pessimist = clash(v_t,v_rem)
		x_optimist,optimist_value = solve_opt(V_optimist,u_current)
		x_pessimist,pessimist_value = solve_opt(V_pessimist,u_current)
		x_t = alpha*x_optimist[0] + (1-alpha)*x_pessimist[0]
		delta_u = x_t*v_t
		u_current+=delta_u
		x_final[t] = x_t
		#pdb.set_trace()
	return(x_final,u_current.min())



def evaluate_policy(policy,V):
	#V = instance(m)
	(T,n) = V.shape
	opt_value = solve_opt(V,np.zeros(n))[1]
	x,val = policy(V)
	#print(x)
	policy_value = maxmin(V,x)
	#print("greedy solution:")
	#print(np.round(x.value,2))
	print("greedy/optimal=")
	print(policy_value/opt_value)
	return(policy_value)
 
def apprx_est(V,u):
	(n,m)=V.shape
	p = np.zeros(n)
	for t in range(0,n):
		p[t] = np.max([V[t,j]/u[j] for j in range(0,m)])
	#print('p',p)
	return(sum(p)/m)
def clash(v_t, v_rem):
	V = np.array([v_t,v_rem])
	return(V)
def resolve(v_t,v_rem):
	n = len(v_t)
	V = np.zeros((n+1,n))
	V[0,:] = v_t
	for t in range(n):
		temp = v_t*0
		temp[t] = v_rem[t]
		V[t+1,:] = temp
	return(V)



policy_1 = lambda V: optimistpessimist(V,0.5)
policy_2 = lambda V: optimistpessimist(V,0.0)
policy_3 = lambda V: optimistpessimist(V,1.0)
#V = np.array([[0.66,0.33],[1-0.66,1-0.33]]) # the example we tested by hand
V = random_instance(10,10)
val1 = evaluate_policy(policy_1,V)
val2 = evaluate_policy(policy_2,V)
val3 = evaluate_policy(policy_3,V)

print(val2)
print(val3)
print(val1)
print((val2+val3)/2.0)

