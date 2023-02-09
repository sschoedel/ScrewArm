import numpy as np


def as_skew_symmetric(vec):
    a, b, c = vec
    return np.array([[0, -c, b],
                     [c, 0, -a],
                     [-b, a, 0]])

def as_xi_skew_symmetric(xi_bar):
    v, w = xi_bar[0:3], xi_bar[3:6]
    w_ss = as_skew_symmetric(w)
    res = np.hstack([w_ss, v.reshape(3,1)])
    return np.vstack([res, [0, 0, 0, 1]])
    
def as_exponential(xi, theta):
    v, w = xi[0:3], xi[3:6]
    
    w_ss = as_skew_symmetric(w)
    
    e_w_ss_theta = np.eye(3) + w_ss * np.sin(theta) + w_ss @ w_ss * (1 - np.cos(theta))
    
    top_right = (np.eye(3) - e_w_ss_theta) @ (w_ss @ v) + w.reshape((3,1)) @ w.reshape((1,3)) @ v * theta
    
    threebyfour = np.hstack([e_w_ss_theta, top_right.reshape(3,1)])
    res = np.vstack([threebyfour, [0,0,0,1]])
    return res

def compute_xi_bar(w, q):
    return np.append(-np.cross(w, q), w)


class BarrettArm():
    tool_offset = np.array([0, 0, 0.12])
    
    g_s0_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 1.346],
                       [0, 0, 0, 1]])
    g_s1_0 = np.array([[0, 0, -1, 0.61],
                       [1, 0, 0, 0.72],
                       [0, -1, 0, 1.346],
                       [0, 0, 0, 1]])
    g_s2_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 1.346],
                       [0, 0, 0, 1]])
    g_s3_0 = np.array([[0, 0, -1, 0.61],
                       [1, 0, 0, 0.72],
                       [0, -1, 0, 1.896],
                       [0, 0, 0, 1]])
    g_s4_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 1.896],
                       [0, 0, 0, 1]])
    g_s5_0 = np.array([[0, 0, -1, 0.61],
                       [1, 0, 0, 0.72],
                       [0, -1, 0, 2.196],
                       [0, 0, 0, 1]])
    g_s6_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 2.196],
                       [0, 0, 0, 1]])
    
    g_s7_0 = np.array([[0, -1, 0, 0.61],
                       [1, 0, 0, 0.72],
                       [0, 0, 1, 2.256],
                       [0, 0, 0, 1]])
    
    g_s0 = 0
    g_s1 = 0
    g_s2 = 0
    g_s3 = 0
    g_s4 = 0
    g_s5 = 0
    g_s6 = 0
    g_s7 = 0
    
    xi_0 = compute_xi_bar([0, 0, 1], [0.61, 0.72, 1.346])
    xi_1 = compute_xi_bar([-1, 0, 0], [0.61, 0.72, 1.346])
    xi_2 = compute_xi_bar([0, 0, 1], [0.61, 0.72, 1.346])
    xi_3 = compute_xi_bar([-1, 0, 0], [0.61, 0.765, 1.896])
    xi_4 = compute_xi_bar([0, 0, 1], [0.61, 0.72, 1.896])
    xi_5 = compute_xi_bar([-1, 0, 0], [0.61, 0.72, 2.196])
    xi_6 = compute_xi_bar([0, 0, 1], [0.61, 0.72, 2.196])
    xi_7 = compute_xi_bar([0, 0, 1], [0.61, 0.72, 2.256])
    
    xi_list = [xi_0, xi_1, xi_2, xi_3, xi_4, xi_5, xi_6, xi_7]
    
    def end_effector_location(self, thetas):
        '''
        thetas: list or np.array of scalars with theta_0 first and theta_6 last
        '''
        thetas = np.append(thetas, 0) # Add theta = 0 for joint 7
        tool_offset_homogeneous = np.append(self.tool_offset, 1)
        return as_exponential(self.xi_0, thetas[0]) @ as_exponential(self.xi_1, thetas[1]) @ \
                as_exponential(self.xi_2, thetas[2]) @ as_exponential(self.xi_3, thetas[3]) @ \
                as_exponential(self.xi_4, thetas[4]) @ as_exponential(self.xi_5, thetas[5]) @ \
                as_exponential(self.xi_6, thetas[6]) @ as_exponential(self.xi_7, thetas[7]) @ \
                    self.g_s7_0 @ tool_offset_homogeneous
    
    def get_joint_positions(self, thetas):
        '''
        thetas: list or np.array of scalars with theta_0 first and theta_6 last
        
        returns: 3x8 matrix of column vectors corresponding to each joint 
                    position's (x,y,z) in the spatial/global frame
        '''
        self.compute_all_joint_poses(thetas)
        
        origin_homo = np.array([0, 0, 0, 1])
        
        # Create matrix where each column vector corresponds to an [x, y, z, 1] joint position
        joint_positions_homo = np.vstack([self.g_s0 @ origin_homo,
                                          self.g_s1 @ origin_homo,
                                          self.g_s2 @ origin_homo,
                                          self.g_s3 @ origin_homo,
                                          self.g_s4 @ origin_homo,
                                          self.g_s5 @ origin_homo,
                                          self.g_s6 @ origin_homo,
                                          self.g_s7 @ origin_homo,
                                          self.g_s7 @ np.append(self.tool_offset, 1)]).T
        return joint_positions_homo[:-1,:]

    def get_ee_position(self, thetas):
        thetas = np.append(thetas, 0)
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        res = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ e_matrices[7] @ self.g_s7_0 @ np.append(self.tool_offset, 1)
        return res[:-1]
        
    def compute_all_joint_poses(self, thetas):
        '''
        thetas: list or np.array of scalars with theta_0 first and theta_6 last
        '''
        thetas = np.append(thetas, 0) # Add theta = 0 for joint 7
        e_matrices = [as_exponential(xi, theta) for xi, theta in zip(self.xi_list, thetas)]
        self.g_s0 = e_matrices[0] @ self.g_s0_0
        self.g_s1 = e_matrices[0] @ e_matrices[1] @ self.g_s1_0
        self.g_s2 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ self.g_s2_0
        self.g_s3 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ self.g_s3_0
        self.g_s4 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ self.g_s4_0
        self.g_s5 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ self.g_s5_0
        self.g_s6 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ self.g_s6_0
        self.g_s7 = e_matrices[0] @ e_matrices[1] @ e_matrices[2] @ e_matrices[3] @ e_matrices[4] @ e_matrices[5] @ e_matrices[6] @ e_matrices[7] @ self.g_s7_0
    
    def __str__(self):
        print("Arm xi column vectors:")
        print("--------------------------------------------")
        for i, xi in enumerate(self.xi_list):
            print(f"xi_{i}: {xi}")
        return ""

# arm = BarrettArm()
# thetas = [1,0,0,0,0,1.2,np.pi/4,np.pi/2]
# print(arm.get_joint_positions(thetas))
# print(arm.end_effector_location(thetas))