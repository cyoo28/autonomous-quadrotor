import numpy as np

from proj1_3.code.graph_search import graph_search
from scipy.sparse.linalg import lsmr

import scipy
class WorldTraj(object):
    def __init__(self, world, start, goal):
        self.resolution = np.array([0.1,0.1, 0.1])
        self.margin = 0.3
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        self.n_segments = self.path.shape[0]-1 # total segments
        self.n_points = self.n_segments+1  # total points
        self.points = np.zeros((self.n_points,3)) # shape=(n_pts,3)
        self.points = self.path

        self.t = np.zeros((self.n_points,))
        for i in range(1,self.n_points):
            self.t[i] = np.linalg.norm(self.points[i-1]-self.points[i])/0.5
        # print(self.t)
        # Mini Snap
        self.n_order = 8
        A = np.zeros((self.n_segments*self.n_order,self.n_segments*self.n_order))
        b_x = np.zeros((self.n_segments*self.n_order,1))
        b_y = np.zeros((self.n_segments*self.n_order,1))
        b_z = np.zeros((self.n_segments*self.n_order,1))

        # x position constraint, total 2*n_segment rows
        for i in range(self.n_segments):
            # t_start = self.t[i]
            t_end = self.t[i+1]
            A[2*i,self.n_order*i:self.n_order*i+8] = np.array([0,0,0,0,0,0,0,1])
            A[2*i+1,self.n_order*i:self.n_order*i+8] = np.array([pow(t_end,7),pow(t_end,6),pow(t_end,5),pow(t_end,4),
                                                       pow(t_end,3),pow(t_end,2),pow(t_end,1),1])
            b_x[2*i,:] = self.points[i,0]
            b_x[2*i+1,:] = self.points[i+1,0]
            b_y[2 * i, :] = self.points[i, 1]
            b_y[2 * i + 1, :] = self.points[i + 1, 1]
            b_z[2 * i, :] = self.points[i, 2]
            b_z[2 * i + 1, :] = self.points[i + 1, 2]

        # start and goal position derivative constraint, total 2*3 rows, just for mini snap here
        A[2*self.n_segments,0:8] = np.array([0,0,0,0,0,0,1,0])

        A[2*self.n_segments + 1,0:8] = np.array([0,0,0,0,0,2,0,0])

        A[2 * self.n_segments + 2, 0:8] = np.array([0,0,0,0,6, 0, 0, 0])

        A[2 * self.n_segments + 3, -8:] = np.array([7*pow(self.t[-1],6),6*pow(self.t[-1],5),5*pow(self.t[-1],4),
                                                    4*pow(self.t[-1],3),3*pow(self.t[-1],2),2*pow(self.t[-1],1),1,0])

        A[2 * self.n_segments + 4, -8:] = np.array([42*pow(self.t[-1],5),30*pow(self.t[-1],4),
                                                    20 * pow(self.t[-1], 3), 12 * pow(self.t[-1], 2),
                                                    6 * pow(self.t[-1], 1), 2, 0, 0])

        A[2 * self.n_segments + 5, -8:] = np.array([210 * pow(self.t[-1], 4), 120 * pow(self.t[-1], 3),
                                                    60 * pow(self.t[-1], 2), 24 * pow(self.t[-1], 1),
                                                    6, 0, 0, 0])


        #  segment derivative continuity constraint, total 4*(n_segment-1) rows, 32 rows here
        for i in range(self.n_segments-1):  # (0,8)

            # velocity
            A[2 * self.n_segments+6 + 6*i, self.n_order*i :self.n_order*i+8] = np.array([7*pow(self.t[i+1],6),
                                                                               6 * pow(self.t[i+1], 5),
                                                                               5 * pow(self.t[i+1], 4),
                                                                               4 * pow(self.t[i+1], 3),
                                                                               3 * pow(self.t[i+1], 2),
                                                                               2 * pow(self.t[i+1], 1), 1, 0])
            A[2 * self.n_segments+6 + 6*i, self.n_order*(i+1):self.n_order*(i+1) + 8] = -np.array([0,0,0,0,0,0,1,0])

            # acc
            A[2 * self.n_segments + 6 + 6*i+1, self.n_order*i:self.n_order*i + 8] = np.array([42*pow(self.t[i+1],5),
                                                                                    30 * pow(self.t[i+1], 4),
                                                                                    20 * pow(self.t[i + 1], 3),
                                                                                    12 * pow(self.t[i + 1], 2),
                                                                                    6 * pow(self.t[i + 1], 1),
                                                                                    2, 0, 0])
            A[2*self.n_segments + 6 + 6*i + 1, self.n_order*(i+1):self.n_order*(i+1) + 8] = -np.array([0,0,0,0,0,2,0,0])

            # jerk
            A[2 * self.n_segments + 6 + 6 * i + 2, self.n_order * i:self.n_order * i + 8] = np.array([210 * pow(self.t[i+1], 4),
                                                                                            120 * pow(self.t[i+1], 3),
                                                                                            60 * pow(self.t[i + 1], 2),
                                                                                            24 * pow(self.t[i + 1], 1),
                                                                                            6, 0, 0, 0])
            A[2 * self.n_segments + 6 + 6*i + 2, self.n_order*(i + 1):self.n_order*(i + 1) + 8] = -np.array([0,0,0,0,6,0,0,0])

            # snap
            A[2 * self.n_segments + 6 + 6 * i + 3, self.n_order * i:self.n_order * i + 8] = np.array([840 * pow(self.t[i+1], 3),
                                                                                            360 * pow(self.t[i + 1], 2),
                                                                                            120 * pow(self.t[i + 1], 1),
                                                                                            24,0, 0, 0, 0])
            A[2 * self.n_segments + 6 + 6 * i + 3, self.n_order * (i + 1):self.n_order * (i + 1) + 8] = -np.array([0,0,0,24,0,0,0,0])

            #
            A[2 * self.n_segments + 6 + 6 * i + 4, self.n_order * i:self.n_order * i + 8] = np.array([2520 * pow(self.t[i+1], 2),
                                                                                            720 * pow(self.t[i+1], 1),
                                                                                            120,0, 0, 0, 0, 0])
            A[2 * self.n_segments + 6 + 6 * i + 4, self.n_order * (i+1):self.n_order * (i+1) + 8] = -np.array([0,0,120, 0, 0, 0, 0, 0])
            #
            A[2 * self.n_segments + 6 + 6 * i + 5, self.n_order * i:self.n_order * i + 8] = np.array([5040 * pow(self.t[i + 1], 1),720,0, 0, 0, 0, 0, 0])
            A[2 * self.n_segments + 6 + 6 * i + 5, self.n_order * (i+1):self.n_order * (i+1) + 8] = -np.array([0,720,0, 0, 0, 0, 0, 0])

        self.coef_x = lsmr(A,b_x)[0].reshape(-1,1)
        self.coef_y = lsmr(A,b_y)[0].reshape(-1,1)
        self.coef_z = lsmr(A,b_z)[0].reshape(-1,1)

    def update(self, t):
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        time_sum = np.cumsum(self.t)
        time_index = np.where(time_sum<=t)[0][-1]
        if time_index == self.points.shape[0]-1 :
            # x_dot = np.zeros((3,))
            x = self.points[-1,:].reshape(3,)
        else:
            coef_x = self.coef_x[time_index*8:time_index*8+8]
            #print("coef_x",coef_x)
            coef_y = self.coef_y[time_index*8:time_index*8+8]
            coef_z = self.coef_z[time_index*8:time_index*8+8]

            coef = np.hstack((coef_x,coef_y,coef_z))
            # print("coef",coef.shape)
            new_t = t-time_sum[time_index]
            x = np.array([pow(new_t,7),pow(new_t,6),pow(new_t,5),pow(new_t,4),pow(new_t,3),pow(new_t,2),new_t,1])@coef
            #print("x",x.shape)
            x_dot =np.array([7*pow(new_t,6),6*pow(new_t,5),5*pow(new_t,4),4*pow(new_t,3),3*pow(new_t,2),2*new_t,1,0])@coef
            x_ddot =np.array([42*pow(new_t,5),30*pow(new_t,4),20*pow(new_t,3),12*pow(new_t,2),6*new_t,2,0,0])@coef
            x_dddot =np.array([210*pow(new_t,4),120*pow(new_t,3),60*pow(new_t,2),24*new_t,6,0,0,0])@coef
            x_ddddot =np.array([840*pow(new_t,3),360*pow(new_t,2),120*new_t,24,0,0,0,0])@coef
        # print("result",x,x_dot)
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        # print(flat_output)
        return flat_output
