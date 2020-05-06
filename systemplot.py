

# python libraries


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

from matplotlib import rcParams
rcParams['figure.figsize'] = (8, 5)

start = np.array([15, 15]) # Start location
goal = np.array([0, 0]) # Goal location

n_obstacles = 30
obstacles = []
np.random.seed(6)
for i in range(n_obstacles):
    obstacles.append(
        (np.random.randint(2,15,1)[0],np.random.randint(2,15,1)[0],1.5*np.random.random_sample())
    )

bounds = np.array([0, 15]) # Bounds in both x and y

def plot_scene(obstacle_list, start, goal):
    ax = plt.gca()
    for o in obstacle_list:
        circle = plt.Circle((o[0], o[1]), o[2], color='k')
        ax.add_artist(circle)
    plt.axis([bounds[0]-0.5, bounds[1]+0.5, bounds[0]-0.5, bounds[1]+0.5])
    plt.plot(start[0], start[1], "xr", markersize=10)
    plt.plot(goal[0], goal[1], "xb", markersize=10)
    plt.legend(('start', 'goal'), loc='upper left')
    plt.gca().set_aspect('equal')
    
# plt.figure()
# plot_scene(obstacles, start, goal)
# plt.tight_layout()

class RRT:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None

    def __init__(self, start, goal, obstacle_list, bounds, 
                 max_extend_length=3.0, path_resolution=0.5, 
                 goal_sample_rate=0.05, max_iter=100):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.bounds = bounds
        self.max_extend_length = max_extend_length
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.node_list = [self.start]
        for i in range(self.max_iter):
           # modify here: 
            # 1) Create a random node (rnd_node) inside 
            # the bounded environment
            rnd_node = self.get_random_node()
            # 2) Find nearest node (nearest_node)
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)
            # 3) Get new node (new_node) by connecting
            # rnd_node and nearest_node. Hint: steer
            new_node = self.steer(nearest_node, rnd_node, max_extend_length=np.inf)
            # 4) If the path between new_node and the
            # nearest node is not in collision, add it to the node_list
            if not self.collision(new_node, nearest_node, self.obstacle_list):
              self.node_list.append(new_node)
            # Don't need to modify beyond here
            # If the new_node is very close to the goal, connect it
            # directly to the goal and return the final path
            if self.dist_to_goal(self.node_list[-1].p) <= self.max_extend_length:
                final_node = self.steer(self.node_list[-1], self.goal, self.max_extend_length)
                if not self.collision(final_node, self.node_list[-1], self.obstacle_list):
                    return self.final_path(len(self.node_list) - 1)
        return None  # cannot find path

    def steer(self, from_node, to_node, max_extend_length=np.inf):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length
        """
        new_node = self.Node(to_node.p)
        d = from_node.p - to_node.p
        dist = np.linalg.norm(d)
        if dist > max_extend_length:
            # rescale the path to the maximum extend_length
            new_node.p  = from_node.p - d / dist * max_extend_length
        new_node.parent = from_node
        return new_node

    def dist_to_goal(self, p):
        """Distance from p to goal"""
        return np.linalg.norm(p - self.goal.p)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            # Sample random point inside boundaries
            rnd = self.Node(np.random.rand(2)*(self.bounds[1]-self.bounds[0]) + self.bounds[0])
        else:  
            # Select goal point
            rnd = self.Node(self.goal.p)
        return rnd
    
    @staticmethod
    def get_nearest_node(node_list, node):
        """Find the nearest node in node_list to node"""
        dlist = [np.sum(np.square((node.p - n.p))) for n in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]
    
    @staticmethod
    def collision(node1, node2, obstacle_list):
        """Check whether the path connecting node1 and node2 
        is in collision with anyting from the obstacle_list
        """
        p1 = node2.p
        p2 = node1.p 
        for o in obstacle_list:
            center_circle = o[0:2]
            radius = o[2]
            d12 = p2 - p1 # the directional vector from p1 to p2
            # defines the line v(t) := p1 + d12*t going through p1=v(0) and p2=v(1)
            d1c = center_circle - p1 # the directional vector from circle to p1
            # t is where the line v(t) and the circle are closest
            # Do not divide by zero if node1.p and node2.p are the same.
            # In that case this will still check for collisions with circles
            t = d12.dot(d1c) / (d12.dot(d12) + 1e-7)
            t = max(0, min(t, 1)) # Our line segment is bounded 0<=t<=1
            d = p1 + d12*t # The point where the line segment and circle are closest
            is_collide = np.sum(np.square(center_circle-d)) < radius**2
            if is_collide:
                return True # is in collision
        return False # is not in collision
    
    def final_path(self, goal_ind):
        """Compute the final path from the goal node to the start node"""
        path = [self.goal.p]
        node = self.node_list[goal_ind]
        # modify here: Generate the final path from the goal node to the start node.
        # We will check that path[0] == goal and path[-1] == start
        while node is not self.start:
          path.append(node.p)
          node = node.parent
        path.append(node.p)
        return path

    def draw_graph(self):
        for node in self.node_list:
            if node.parent:
                plt.plot([node.p[0], node.parent.p[0]], [node.p[1], node.parent.p[1]], "-g")

class RRTStar(RRT):
    
    class Node(RRT.Node):
        def __init__(self, p):
            super().__init__(p)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, bounds,
                 max_extend_length=5.0,
                 path_resolution=0.5,
                 goal_sample_rate=0.0,
                 max_iter=200,
                 connect_circle_dist=50.0
                 ):
        super().__init__(start, goal, obstacle_list, bounds, max_extend_length,
                         path_resolution, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.goal = self.Node(goal)

    def plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Create a random node inside the bounded environment
            rnd = self.get_random_node()
            # Find nearest node
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            # Get new node by connecting rnd_node and nearest_node
            new_node = self.steer(nearest_node, rnd, self.max_extend_length)
            # If path between new_node and nearest node is not in collision:
            if not self.collision(new_node, nearest_node, self.obstacle_list):
                near_inds = self.near_nodes_inds(new_node)
                # Connect the new node to the best parent in near_inds
                new_node = self.choose_parent(new_node, near_inds)
                self.node_list.append(new_node)
                # Rewire the nodes in the proximity of new_node if it improves their costs
                self.rewire(new_node, near_inds)
        last_index, min_cost = self.best_goal_node_index()
        if last_index:
            return self.final_path(last_index), min_cost
        return None, min_cost

    def choose_parent(self, new_node, near_inds):
        """Set new_node.parent to the lowest resulting cost parent in near_inds and
        new_node.cost to the corresponding minimal cost
        """
        min_cost = np.inf
        best_near_node = None
        # modify here: Go through all near nodes and evaluate them as potential parent nodes by
        # 1) checking whether a connection would result in a collision,
        # 2) evaluating the cost of the new_node if it had that near node as a parent,
        # 3) picking the parent resulting in the lowest cost and updating
        #    the cost of the new_node to the minimum cost.
        
        for i in near_inds:
          candidate_node = self.node_list[i]
          if not self.collision(new_node, candidate_node, self.obstacle_list):
            temp_cost = self.new_cost(candidate_node, new_node)
            if temp_cost < min_cost:
              min_cost = temp_cost
              best_near_node = candidate_node
        # Don't need to modify beyond here
        new_node.cost = min_cost
        new_node.parent = best_near_node
        return new_node
    
    def rewire(self, new_node, near_inds):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        # modify here: Go through all near nodes and check whether rewiring them
        # to the new_node would: 
        # A) Not cause a collision and
        # B) reduce their own cost.
        # If A and B are true, update the cost and parent properties of the node.
        for i in near_inds:
          candidate_node = self.node_list[i]
          if not self.collision(new_node, candidate_node, self.obstacle_list):
            temp_cost = self.new_cost(new_node, candidate_node)
            if temp_cost < candidate_node.cost:
              candidate_node.parent = new_node
              candidate_node.cost = temp_cost
        # Don't need to modify beyond here
        self.propagate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        """Find the lowest cost node to the goal"""
        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            # Has to be in close proximity to the goal
            if self.dist_to_goal(node.p) <= self.max_extend_length:
                # Connection between node and goal needs to be collision free
                if not self.collision(self.goal, node, self.obstacle_list):
                    # The final path length
                    cost = node.cost + self.dist_to_goal(node.p) 
                    if node.cost + self.dist_to_goal(node.p) < min_cost:
                        # Found better goal node!
                        min_cost = cost
                        best_goal_node_idx = i
        return best_goal_node_idx, min_cost

    def near_nodes_inds(self, new_node):
        """Find the nodes in close proximity to new_node"""
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        dlist = [np.sum(np.square((node.p - new_node.p))) for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return near_inds

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        d = np.linalg.norm(from_node.p - to_node.p)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

np.random.seed(7)
rrt_star = RRTStar(start=start,
          goal=goal,
          bounds=bounds,
          obstacle_list=obstacles)
path_rrt_star, min_cost = rrt_star.plan()
print('Minimum cost: {}'.format(min_cost))

# Check the cost
def path_cost(path):
    return sum(np.linalg.norm(path[i] - path[i + 1]) for i in range(len(path) - 1))

if path_rrt_star:
    print('Length of the found path: {}'.format(path_cost(path_rrt_star)))


# -*- coding: utf-8 -*-
"""ilqr_driving.ipynb
"""


"""## Iterative Linear Quadratic Regulator Derivation

iLQR algorithm will be capable of solving general model predictive control problems in the form described above, we will apply it to the control of a vehicle. 

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
des=[]
for i in path_rrt_star:
    des.append(i)

des.reverse()
print(des)
f_trj = []
for i in range(1, len(des)):
    if i == 1:
        start = des[0]
        x0 = np.array([start[0],start[1], -1.5, 0.0, 0.0])
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


# Plot RRT* trajectory
plt.figure(figsize=(9.5,8))
ax = plt.gca()
plot_scene(obstacles, start, goal)
rrt_star.draw_graph()
if path_rrt_star is None:
    print("No viable path found")
else:
    plt.plot([x for (x, y) in path_rrt_star], [y for (x, y) in path_rrt_star], '-r')
plt.tight_layout()

# Plot resulting trajecotry of car
plt.plot(f_trj[:,0], f_trj[:,1], linewidth=2)
w = 0.3
h = 0.15

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