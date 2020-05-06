# -*- coding: utf-8 -*-
"""ilqr_driving.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qT76HgUYXI1VvK6eZWvxeWHVWcHHn_i9

# Iterative Linear Quadratic Regulator

## Notebook Setup 
The following cell will install Drake, checkout the underactuated repository, and set up the path (only if necessary).
- On Google's Colaboratory, this **will take approximately two minutes** on the first time it runs (to provision the machine), but should only need to reinstall once every 12 hours.  Colab will ask you to "Reset all runtimes"; say no to save yourself the reinstall.
- On Binder, the machines should already be provisioned by the time you can run this; it should return (almost) instantly.

More details are available [here](http://underactuated.mit.edu/drake.html).
"""


import pydrake

# python libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# pydrake imports
from pydrake.all import (Variable, SymbolicVectorSystem, DiagramBuilder,
                         LogOutput, Simulator, ConstantVectorSource,
                         MathematicalProgram, Solve, SnoptSolver, PiecewisePolynomial)
import pydrake.symbolic as sym

"""## Iterative Linear Quadratic Regulator Derivation

In this exercise we will derive the iterative Linear Quadratic Regulator (iLQR) solving the following optimization problem.

\begin{align*} \min_{\mathbf{u}[\cdot]} \quad & \ell_f(\mathbf{x}[N]) +
      \sum_{n=0}^{N-1} \ell(\mathbf{x}[n],\mathbf{u}[n]) \\ \text{subject to} \quad & \mathbf{x}[n+1] = {\bf
      f}(\mathbf{x}[n], \mathbf{u}[n]), \quad \forall n\in[0, N-1] \\ & \mathbf{x}[0] = \mathbf{x}_0
\end{align*}

After completing this exercise you will be able to write your own MPC solver from scratch without any proprietary or third-party software (with the exception of auto-differentiation). You will derive all necessary equations yourself.
While the iLQR algorithm will be capable of solving general model predictive control problems in the form described above, we will apply it to the control of a vehicle. 

### Vehicle Control Problem
Before we start the actual derivation of iLQR we will take a look at the vehicle dynamics and cost functions. The vehicle has the following continuous time dynamics and is controlled by longitudinal acceleration and steering velocity.
"""

n_x = 5
n_u = 2
def car_continuous_dynamics(x, u):
    # x = [x position, y position, heading, speed, steering angle] 
    # u = [acceleration, steering velocity]
    m = sym if x.dtype == object else np # Check type for autodiff
    heading = x[2]
    v = x[3]
    steer = x[4]
    x_d = np.array([
        v*m.cos(heading),
        v*m.sin(heading),
        v*m.tan(steer),
        u[0],
        u[1]        
    ])
    return x_d

"""Note that while the vehicle dynamics are in continuous time, our problem formulation is in discrete time. Define the general discrete time dynamics $\bf f$ with a simple [Euler integrator](https://en.wikipedia.org/wiki/Euler_method) in the next cell."""

def discrete_dynamics(x, u):
    dt = 0.1
    # TODO: Fill in the Euler integrator below and return the next state
    x_next = x + dt*car_continuous_dynamics(x, u)
    return x_next

"""Given an initial state $\mathbf{x}_0$ and a guess of a control trajectory $\mathbf{u}[0:N-1]$ we roll out the state trajectory $x[0:N]$ until the time horizon $N$. Please complete the rollout function."""

def rollout(x0, u_trj):
    x_trj = np.zeros((u_trj.shape[0]+1, x0.shape[0]))
    # TODO: Define the rollout here and return the state trajectory x_trj: [N, number of states]
    for i in range(u_trj.shape[0]+1):
      if i == 0:
        x_trj[i,:] = x0
      else:
        x_trj[i,:] = discrete_dynamics(x_trj[i-1,:], u_trj[i-1,:])     
    return x_trj


v_target = 0.01

eps = 1e-6 # The derivative of sqrt(x) at x=0 is undefined. Avoid by subtle smoothing
def cost_stage(x, u, des):
    m = sym if x.dtype == object else np # Check type for autodiff
    c_trj = (x[0]-des[0])**2 + (x[1]-des[1])**2 + eps
    c_speed = (x[3]-v_target)**2
    c_control= (u[0]**2 + u[1]**2)*0.1
    return c_trj + c_speed + c_control

def cost_final(x,des):
    m = sym if x.dtype == object else np # Check type for autodiff
    c_trj = (x[0]-des[0])**2 + (x[1]-des[1])**2 + eps
    c_speed = (x[3]-v_target)**2
    return c_trj + c_speed

def cost_trj(x_trj, u_trj,des):
    total = 0.0
    # TODO: Sum up all costs
    for i in range(u_trj.shape[0]):
      total += cost_stage(x_trj[i,:], u_trj[i,:], des)
    total += cost_final(x_trj[-1,:],des)
    return total
    
class derivatives():
    def __init__(self, discrete_dynamics, cost_stage, cost_final, n_x, n_u, des):
        self.x_sym = np.array([sym.Variable("x_{}".format(i)) for i in range(n_x)])
        self.u_sym = np.array([sym.Variable("u_{}".format(i)) for i in range(n_u)])
        x = self.x_sym
        u = self.u_sym
        
        l = cost_stage(x, u, des)
        self.l_x = sym.Jacobian([l], x).ravel()
        self.l_u = sym.Jacobian([l], u).ravel()
        self.l_xx = sym.Jacobian(self.l_x, x)
        self.l_ux = sym.Jacobian(self.l_u, x)
        self.l_uu = sym.Jacobian(self.l_u, u)
        
        l_final = cost_final(x,des)
        self.l_final_x = sym.Jacobian([l_final], x).ravel()
        self.l_final_xx = sym.Jacobian(self.l_final_x, x)
        
        f = discrete_dynamics(x, u)
        self.f_x = sym.Jacobian(f, x)
        self.f_u = sym.Jacobian(f, u)
    
    def stage(self, x, u):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        env.update({self.u_sym[i]: u[i] for i in range(u.shape[0])})
        
        l_x = sym.Evaluate(self.l_x, env).ravel()
        l_u = sym.Evaluate(self.l_u, env).ravel()
        l_xx = sym.Evaluate(self.l_xx, env)
        l_ux = sym.Evaluate(self.l_ux, env)
        l_uu = sym.Evaluate(self.l_uu, env)
        
        f_x = sym.Evaluate(self.f_x, env)
        f_u = sym.Evaluate(self.f_u, env)

        return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u
    
    def final(self, x):
        env = {self.x_sym[i]: x[i] for i in range(x.shape[0])}
        
        l_final_x = sym.Evaluate(self.l_final_x, env).ravel()
        l_final_xx = sym.Evaluate(self.l_final_xx, env)
        
        return l_final_x, l_final_xx
        

def Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    # TODO: Define the Q-terms here
    Q_x = np.zeros(l_x.shape)
    Q_x = l_x + V_x.dot(f_x)
    Q_u = np.zeros(l_u.shape)
    Q_u = l_u + V_x.dot(f_u)
    Q_xx = np.zeros(l_xx.shape)
    Q_xx = l_xx + f_x.T.dot(V_xx).dot(f_x)
    Q_ux = np.zeros(l_ux.shape)
    Q_ux = l_ux + f_u.T.dot(V_xx).dot(f_x)
    Q_uu = np.zeros(l_uu.shape)
    Q_uu = l_uu + f_u.T.dot(V_xx).dot(f_u)
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

def gains(Q_uu, Q_u, Q_ux):
    Q_uu_inv = np.linalg.inv(Q_uu)
    # TOD: Implement the feedforward gain k and feedback gain K.
    k = np.zeros(Q_u.shape)
    k = -Q_uu_inv.dot(Q_u)
    K = np.zeros(Q_ux.shape)
    K = -Q_uu_inv.dot(Q_ux)
    return k, K


def V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    # TODO: Implement V_x and V_xx, hint: use the A.dot(B) function for matrix multiplcation.
    V_x = np.zeros(Q_x.shape)
    V_x = Q_x + Q_u.dot(K) + k.dot(Q_ux) + k.dot(Q_uu).dot(K)
    V_xx = np.zeros(Q_xx.shape)
    V_xx = Q_xx + 2*K.T.dot(Q_ux) + K.T.dot(Q_uu).dot(K)
    return V_x, V_xx


def expected_cost_reduction(Q_u, Q_uu, k):
    return -Q_u.T.dot(k) - 0.5 * k.T.dot(Q_uu.dot(k))


def forward_pass(x_trj, u_trj, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0,:] = x_trj[0,:]
    u_trj_new = np.zeros(u_trj.shape)
    # TODO: Implement the forward pass here
    for n in range(u_trj.shape[0]):
        u_trj_new[n,:] = u_trj[n,:] + k_trj[n,:] + K_trj[n,:].dot((x_trj_new[n,:] - x_trj[n,:]))  # Apply feedback law
        x_trj_new[n+1,:] = discrete_dynamics(x_trj_new[n,:], u_trj_new[n,:]) # Apply dynamics
    return x_trj_new, u_trj_new


def backward_pass(x_trj, u_trj, regu):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    # TODO: Set terminal boundary condition here (V_x, V_xx)
    V_x = np.zeros((x_trj.shape[1],))
    V_x = derivs.final(x_trj[-1,:])[0]
    V_xx = np.zeros((x_trj.shape[1],x_trj.shape[1]))
    V_xx = derivs.final(x_trj[-1,:])[1]
    for n in range(u_trj.shape[0]-1, -1, -1):
        # TODO: First compute derivatives, then the Q-terms 
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = derivs.stage(x_trj[n,:], u_trj[n,:])
        Q_x = np.zeros((x_trj.shape[1],))
        Q_u = np.zeros((u_trj.shape[1],))
        Q_xx = np.zeros((x_trj.shape[1], x_trj.shape[1]))
        Q_ux = np.zeros((u_trj.shape[1], x_trj.shape[1]))
        Q_uu = np.zeros((u_trj.shape[1], u_trj.shape[1]))
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
        k, K = gains(Q_uu_regu, Q_u, Q_ux)
        k_trj[n,:] = k
        K_trj[n,:,:] = K
        V_x, V_xx = V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
        expected_cost_redu += expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu


def run_ilqr(des, x0, N, max_iter=50, regu_init=100):
    # First forward rollout
    u_trj = np.random.randn(N-1, n_u)*0.0001
    x_trj = rollout(x0, u_trj)
    total_cost = cost_trj(x_trj, u_trj, des)
    regu = regu_init
    max_regu = 10000
    min_regu = 0.01
    
    # Setup traces
    cost_trace = [total_cost]
    expected_cost_redu_trace = []
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]
    
    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = backward_pass(x_trj, u_trj, regu)
        x_trj_new, u_trj_new = forward_pass(x_trj, u_trj, k_trj, K_trj)
        # Evaluate new trajectory
        total_cost = cost_trj(x_trj_new, u_trj_new,des)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu)
        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        if expected_cost_redu <= 1e-6:
            break
            
    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace

# Setup problem and call iLQR
N = 100
max_iter=50
regu_init=100

des = [(0, 0), (0.7913369 , 0.70519748), (8.71058056, 4.34059217), (11.23839671,  6.45805982), (14.76180605, 10.00567686),(15, 15)]
des.reverse()
print("des=",des)
f_trj = []
for i in range(1, len(des)):
    if i == 1:
        start = des[0]
    else:
        start = start_new
    x0 = np.array([start[0],start[1], -2.5, 0.0, 0.0])
    derivs = derivatives(discrete_dynamics, cost_stage, cost_final, n_x, n_u, des[i])
    x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace = run_ilqr(des[i],x0, N, max_iter, regu_init)
    start_new = (x_trj[-1][0], x_trj[-1][1])
    if i == 1:
        f_trj = x_trj
    else:
        f_trj = np.vstack((f_trj, x_trj))

plt.figure(figsize=(9.5,8))
ax = plt.gca()

# Plot resulting trajecotry of car
plt.plot(f_trj[:,0], f_trj[:,1], linewidth=1)
w = 0.2
h = 0.1

# Plot rectangles
for n in range(f_trj.shape[0]):
    rect = mpl.patches.Rectangle((-w/2,-h/2), w, h, fill=False)
    t = mpl.transforms.Affine2D().rotate_deg_around(0, 0, 
            np.rad2deg(f_trj[n,2])).translate(f_trj[n,0], f_trj[n,1]) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
ax.set_aspect(1)
plt.ylim((-1,16))
plt.xlim((-1,16))
plt.tight_layout()
plt.show()