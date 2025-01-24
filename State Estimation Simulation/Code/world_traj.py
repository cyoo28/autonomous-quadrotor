import numpy as np
import scipy

from .graph_search import graph_search

class WorldTraj(object):
    def on_Line(self, end1, end2, point):
        # Function to remove points on a line within a threshold
        if np.linalg.norm(np.cross(end2-end1,point-end1)) > 0.01:
            return False
        if np.dot(end2-end1,point-end1) < 0 or np.dot(end2-end1,point-end1) > np.linalg.norm(end2-end1):
            return False
        return True

    def DouglasPeucker(self, points, epsilon):
        # Function to remove redundant points
        dmax = 0
        index = 0
        start = points[0]
        end = points[-1]
        for i in range(len(points)-1):
            d = np.linalg.norm(np.cross(end-start,points[i+1]-start))/np.linalg.norm(end-start)
            if d > dmax:
                index = i+1
                dmax = d
        if dmax > epsilon:
            recResults1 = self.DouglasPeucker(points[0:index],epsilon)
            recResults2 = self.DouglasPeucker(points[index:],epsilon)
            results = np.vstack((recResults1,recResults2))
        else:
            results = np.array([points[0],points[-1]]).reshape(2,3)
        return results

    def __init__(self, world, start, goal):
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.64
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        self.points = np.zeros((1,3)) # shape=(n_pts,3)
        #self.points = self.DouglasPeucker(self.path, 0.1)
        self.points = self.path
        self.vel = 3.42
        self.times = np.zeros((self.points.shape[0],))
        # Defining times for each segment of the trajectory based on distance traveled and somewhat arbitrary "velocity"
        for i in range(1,self.points.shape[0]):
            self.times[i] = np.linalg.norm(self.points[i - 1] - self.points[i]) / self.vel

        # Setting up empty matrices to solve for Ax = b
        # Same A for x, y, and z directions
        # Different b for x, y, and z directions
        A = np.zeros((8*(self.points.shape[0]-1), 8*(self.points.shape[0]-1)))
        bx = np.zeros((8*(self.points.shape[0]-1), 1))
        by = np.zeros((8*(self.points.shape[0]-1), 1))
        bz = np.zeros((8*(self.points.shape[0]-1), 1))

        # Position constraints for every point on the path
        # Provides constraint for x, y, and z positions
        # End of current trajectory at t=times[i+1] and start of next trajectory at t=0
        for i in range(self.points.shape[0]-1):
            A[2*i,   8*i:8*i+8] = np.array([0, 0, 0, 0, 0, 0, 0, 1])
            A[2*i+1, 8*i:8*i+8] = np.array([self.times[i+1]**7, self.times[i+1]**6, self.times[i+1]**5,
                                            self.times[i+1]**4, self.times[i+1]**3, self.times[i+1]**2,
                                            self.times[i+1], 1])

            bx[2*i, :] = self.points[i, 0]
            by[2*i, :] = self.points[i, 1]
            bz[2*i, :] = self.points[i, 2]

            bx[2*i+1, :] = self.points[i+1, 0]
            by[2*i+1, :] = self.points[i+1, 1]
            bz[2*i+1, :] = self.points[i+1, 2]

        # Constraints for the beginning and the end points of the path
        # Provides constraint for velocity, acceleration, and jerk
        A[2*self.points.shape[0]+1, -8:] = np.array([7*self.times[-1]**6, 6*self.times[-1]**5, 5*self.times[-1]**4,
                                                     4*self.times[-1]**3, 3*self.times[-1]**2, 2*self.times[-1], 1, 0])
        A[2*self.points.shape[0]+2, -8:] = np.array([42*self.times[-1]**5, 30*self.times[-1]**4, 20*self.times[-1]**3,
                                                     12*self.times[-1]**2, 6*self.times[-1], 2, 0, 0])
        A[2*self.points.shape[0]+3, -8:] = np.array([210*self.times[-1]**4, 120*self.times[-1]**3, 60*self.times[-1]**2,
                                                     24*self.times[-1], 6, 0, 0, 0])

        A[2*self.points.shape[0]-2, 0:8] = np.array([0, 0, 0, 0, 0, 0, 1, 0])
        A[2*self.points.shape[0]-1, 0:8] = np.array([0, 0, 0, 0, 0, 2, 0, 0])
        A[2*self.points.shape[0],   0:8] = np.array([0, 0, 0, 0, 6, 0, 0, 0])

        # Constraints for the intermediate points of the path
        # Continuity constraints for velocity, acceleration, jerk, and snap
        # Additional constraints added by taking the first and second derivative of snap
        for i in range(self.points.shape[0]-2):
            A[2*self.points.shape[0]+4+6*i, 8*i:8*(i+1)] = np.array([7*self.times[i+1]**6, 6*self.times[i+1]**5,
                                                                     5*self.times[i+1]**4, 4*self.times[i+1]**3,
                                                                     3*self.times[i+1]**2, 2*self.times[i+1], 1, 0])
            A[2*self.points.shape[0]+5+6*i, 8*i:8*(i+1)] = np.array([42*self.times[i+1]**5, 30*self.times[i+1]**4,
                                                                     20*self.times[i+1]**3, 12*self.times[i+1]**2,
                                                                     6*self.times[i+1], 2, 0, 0])
            A[2*self.points.shape[0]+6+6*i, 8*i:8*(i+1)] = np.array([210*self.times[i+1]**4, 120*self.times[i+1]**3,
                                                                     60*self.times[i+1]**2, 24*self.times[i+1],
                                                                     6, 0, 0, 0])
            A[2*self.points.shape[0]+7+6*i, 8*i:8*(i+1)] = np.array([840*self.times[i+1]**3, 360*self.times[i+1]**2,
                                                                     120*self.times[i+1], 4, 0, 0, 0, 0])
            A[2*self.points.shape[0]+8+6*i, 8*i:8*(i+1)] = np.array([2520*self.times[i+1]**2, 720*self.times[i+1],
                                                                     120, 0, 0, 0, 0, 0])
            A[2*self.points.shape[0]+9+6*i, 8*i:8*(i+1)] = np.array([5040 * self.times[i + 1], 720, 0, 0, 0, 0, 0, 0])

            A[2*self.points.shape[0]+4+6*i, 8*(i+1):8*(i+2)] = -np.array([0, 0, 0, 0, 0, 0, 1, 0])
            A[2*self.points.shape[0]+5+6*i, 8*(i+1):8*(i+2)] = -np.array([0, 0, 0, 0, 0, 2, 0, 0])
            A[2*self.points.shape[0]+6+6*i, 8*(i+1):8*(i+2)] = -np.array([0, 0, 0, 0, 6, 0, 0, 0])
            A[2*self.points.shape[0]+7+6*i, 8*(i+1):8*(i+2)] = -np.array([0, 0, 0, 24, 0, 0, 0, 0])
            A[2*self.points.shape[0]+8+6*i, 8*(i+1):8*(i+2)] = -np.array([0, 0, 120, 0, 0, 0, 0, 0])
            A[2*self.points.shape[0]+9+6*i, 8*(i+1):8*(i+2)] = -np.array([0, 720, 0, 0, 0, 0, 0, 0])

        # Solve for "x" for x, y, and z
        self.coef_x = scipy.sparse.linalg.lsmr(A,bx)[0].reshape(-1,1)
        self.coef_y = scipy.sparse.linalg.lsmr(A,by)[0].reshape(-1,1)
        self.coef_z = scipy.sparse.linalg.lsmr(A,bz)[0].reshape(-1,1)

    def update(self, t):
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        if t >= sum(self.times):
            x = self.points[-1,:].reshape(3,)
        else:
            time_seg = np.where(np.cumsum(self.times) <= t)[0][-1]

            x_c = self.coef_x[time_seg*8:time_seg*8+8]
            y_c = self.coef_y[time_seg*8:time_seg*8+8]
            z_c = self.coef_z[time_seg*8:time_seg*8+8]
            xyz_c = np.hstack((x_c, y_c, z_c))
            # For min jerk x**6=0 so x is 5th degree
            # So for min snap:
            # x**8=0 so x is 7th degree polynomial
            x = np.array([(t-sum(self.times[:time_seg]))**7, (t-sum(self.times[:time_seg]))**6, (t-sum(self.times[:time_seg]))**5,
                          (t-sum(self.times[:time_seg]))**4, (t-sum(self.times[:time_seg]))**3, (t-sum(self.times[:time_seg]))**2,
                          (t-sum(self.times[:time_seg])), 1])@xyz_c
            # x_dot is derivative of x and so on
            x_dot = np.array([7*(t-sum(self.times[:time_seg]))**6, 6*(t-sum(self.times[:time_seg]))**5,
                              5*(t-sum(self.times[:time_seg]))**4, 4*(t-sum(self.times[:time_seg]))**3,
                              3*(t-sum(self.times[:time_seg]))**2, 2*(t-sum(self.times[:time_seg])), 1, 0])@xyz_c
            x_ddot = np.array([42*(t-sum(self.times[:time_seg]))**5, 30*(t-sum(self.times[:time_seg]))**4,
                               20*(t-sum(self.times[:time_seg]))**3, 12*(t-sum(self.times[:time_seg]))**2,
                               6*(t-sum(self.times[:time_seg])), 2, 0, 0])@xyz_c
            x_dddot = np.array([210*(t-sum(self.times[:time_seg]))**4, 120*(t-sum(self.times[:time_seg]))**3,
                                60*(t-sum(self.times[:time_seg]))**2, 24*(t-sum(self.times[:time_seg])), 6, 0, 0, 0])@xyz_c
            x_ddddot = np.array([840*(t-sum(self.times[:time_seg]))**3, 360*(t-sum(self.times[:time_seg]))**2,
                                 120*(t-sum(self.times[:time_seg])), 24, 0, 0, 0, 0])@xyz_c

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
