import numpy as np

tau = 2*np.pi
VF_DIFF = 1.0
VS_DIFF = 1.0
HEADING_DIFF = 2.0
N_PFND = 40
N_PFNV = 20
N_HDELTAB = 19

# Define connection weight parameters
wPDH = 1.0
wPVH = 1.0

PFNVR_HDELB_RAW = np.genfromtxt('PFNVR_HDELB.csv', delimiter=',', dtype = 'float64')
PFNVL_HDELB_RAW = np.genfromtxt('PFNVL_HDELB.csv', delimiter=',', dtype = 'float64')
PFNDR_HDELB_RAW = np.genfromtxt('PFNDR_HDELB.csv', delimiter=',', dtype = 'float64')
PFNDL_HDELB_RAW = np.genfromtxt('PFNDL_HDELB.csv', delimiter=',', dtype = 'float64')
HDELB_HDELB_RAW = np.genfromtxt('HDELB_HDELB.csv', delimiter=',', dtype = 'float64')

EPGL_PFNDL = np.genfromtxt('EPGL_PFNDL.csv', delimiter=',', dtype = 'float64')
EPGR_PFNDR = np.genfromtxt('EPGR_PFNDR.csv', delimiter=',', dtype = 'float64')
EPGL_PFNVL = np.genfromtxt('EPGL_PFNVL.csv', delimiter=',', dtype = 'float64')
EPGR_PFNVR = np.genfromtxt('EPGR_PFNVR.csv', delimiter=',', dtype = 'float64')

PFNVR_HDELB_RAW = np.array([np.append(PFNVR_HDELB_RAW[i, 1:], PFNVR_HDELB_RAW[:, 0][i]) for i in range(len(PFNVR_HDELB_RAW[:, 0]))])
PFNVL_HDELB_RAW = np.array([np.append(PFNVL_HDELB_RAW[i, 1:], PFNVL_HDELB_RAW[:, 0][i]) for i in range(len(PFNVL_HDELB_RAW[:, 0]))])
PFNDR_HDELB_RAW = np.array([np.hstack((PFNDR_HDELB_RAW[i, 0], PFNDR_HDELB_RAW[i, 2:], PFNDR_HDELB_RAW[i, 1])) for i in range(len(PFNDR_HDELB_RAW[:, 0]))])
PFNDL_HDELB_RAW = np.array([np.hstack((PFNDL_HDELB_RAW[i, 0], PFNDL_HDELB_RAW[i, 2:], PFNDL_HDELB_RAW[i, 1])) for i in range(len(PFNDR_HDELB_RAW[:, 0]))])

PFNVR_HDELB = np.divide(PFNVR_HDELB_RAW, np.sum(PFNVR_HDELB_RAW, 1)[:,None])
PFNVL_HDELB = np.divide(PFNVL_HDELB_RAW, np.sum(PFNVL_HDELB_RAW, 1)[:,None])
PFNDR_HDELB = 6.0*np.divide(PFNDR_HDELB_RAW, np.sum(PFNDR_HDELB_RAW, 1)[:,None])
PFNDL_HDELB = 6.0*np.divide(PFNDL_HDELB_RAW, np.sum(PFNDL_HDELB_RAW, 1)[:,None])
HDELB_HDELB = np.divide(HDELB_HDELB_RAW, np.sum(HDELB_HDELB_RAW, 1)[:,None])


def set_PFNV_weight(new_wPVH):
    globals()['wPVH'] = new_wPVH


def set_PFND_weight(new_wPDH):
    globals()['wPDH'] = new_wPDH


def ReLU(x):
    return np.maximum(x, 0)


def get_travel_direction(movement_info):

    #vf, vs, heading = np.round(movement_info, 3)
    vf, vs, heading = movement_info

    if vf == 0.0 and vs == 0.0:
        return wrap_to_pi(np.radians(heading), rads=True)

    fly_centric_angle = np.arctan2(vs, vf)
    return fly_centric_angle

# Calculates a PVA from an ordered hDeltaB output
def calc_PVA(z):
    encoding_angles = np.linspace(2*np.pi, 0, 14, endpoint = False)
    encoding_xs = np.cos(encoding_angles)
    encoding_ys = np.sin(encoding_angles)

    decoded_x = np.dot(z, encoding_xs)
    decoded_y = np.dot(z, encoding_ys)

    decoded_angle = np.arctan2(decoded_y, decoded_x)
    return decoded_angle


def wrap_to_pi(angle, rads = True):

    if rads == True:
        while angle < -np.pi:
            angle = angle + tau

        while angle > np.pi:
            angle = angle - tau
    else:
        while angle < -180:
            angle = angle + 360.0

        while angle > 180:
            angle = angle - 360.0

    return angle


def shift_peak_by_angle(peak, goal_angle, rads = True):

    if rads == False:
        goal_angle = wrap_to_pi(np.radians(goal_angle))
    else:
        goal_angle = wrap_to_pi(goal_angle)

    n_points = len(peak)

    shift = int(np.round(goal_angle/tau * n_points, 0))

    return np.roll(peak, shift)


def downsample(peak, n_desired_points):

    downsampled_indices = np.linspace(0, len(peak), n_desired_points, endpoint=False, dtype='int32')
    downsampled_bump = peak[downsampled_indices]

    return downsampled_bump


def create_bump(discretized_angles, shape_param = 1.0, bump_type = 'von_mises'):

    if bump_type == 'von_mises':
        kappa = shape_param
        return np.exp(kappa * np.cos(discretized_angles))/(2*np.pi * sp.iv(0, kappa))
    elif bump_type == 'sin_squared':
        return shape_param*(np.sin(0.5*discretized_angles))**2
    elif bump_type == 'cos_squared':
        return shape_param*(np.cos(0.5*discretized_angles))**2


def discretize_angles(N_discretize, n_neurons_in_bump=0):

    discrete_angles_r = np.linspace(-np.pi, np.pi, N_discretize, endpoint=False)
    discrete_angles_l = np.linspace(np.pi, -np.pi, N_discretize, endpoint=False)

    if n_neurons_in_bump:
        discrete_angles_l = downsample(discrete_angles_l, n_neurons_in_bump)
        discrete_angles_r = downsample(discrete_angles_r, n_neurons_in_bump)

    return discrete_angles_l, discrete_angles_r

def v_eq(v):

    return 5.0*ReLU(v) + 1.0

def pref_angle_eq(movement_info, preferred_angle):

    vf, vs, heading = movement_info
    trav_dir = get_travel_direction(movement_info)
    move_mag = np.linalg.norm([vf, vs])

    angle_diff = wrap_to_pi(trav_dir-np.radians(preferred_angle))
    projection = move_mag*np.cos(angle_diff)
    scaling = v_eq(projection)

    return scaling


def create_PFND(bump_params, movement_info, rads=True):

    N_discretize = 8
    vf, vs, heading = movement_info

    # get the allocentric travel direction
    trav_dir = get_travel_direction(movement_info)

    pfnd_scaling_l = pref_angle_eq(movement_info, -31.0)
    pfnd_scaling_r = pref_angle_eq(movement_info, 31.0)

    discrete_angles_l, discrete_angles_r = discretize_angles(N_discretize)

    pb_template_l = create_bump(discrete_angles_l, bump_params, bump_type='cos_squared')
    pb_template_r = create_bump(discrete_angles_r, bump_params, bump_type='cos_squared')

    # For left PB, angles increase with decreasing array index
    shifted_pb_l = shift_peak_by_angle(pb_template_l, heading, rads=rads)
    shifted_pb_r = shift_peak_by_angle(pb_template_r, -heading, rads=rads)

    shifted_pfnd_l = np.matmul(EPGL_PFNDL, shifted_pb_l)
    shifted_pfnd_r = np.matmul(EPGR_PFNDR, shifted_pb_r)

    PFND_L = pfnd_scaling_l*shifted_pfnd_l
    PFND_R = pfnd_scaling_r*shifted_pfnd_r

    return PFND_L, PFND_R


def create_PFNV(bump_params, movement_info, rads=True):

    N_discretize = 8
    vf, vs, heading = movement_info

    pfnv_scaling_l = pref_angle_eq(movement_info, 137.0)
    pfnv_scaling_r = pref_angle_eq(movement_info, -137.0)

    discrete_angles_l, discrete_angles_r = discretize_angles(N_discretize)

    pb_template_l = create_bump(discrete_angles_l, bump_params, bump_type='cos_squared')
    pb_template_r = create_bump(discrete_angles_r, bump_params, bump_type='cos_squared')

    # For left PB, angles increase with decreasing array index
    shifted_pb_l = shift_peak_by_angle(pb_template_l, heading, rads=rads)
    shifted_pb_r = shift_peak_by_angle(pb_template_r, -heading, rads=rads)

    shifted_pfnv_l = np.matmul(EPGL_PFNVL, shifted_pb_l)
    shifted_pfnv_r = np.matmul(EPGR_PFNVR, shifted_pb_r)

    PFNV_L = pfnv_scaling_l*shifted_pfnv_l
    PFNV_R = pfnv_scaling_r*shifted_pfnv_r

    return PFNV_L, PFNV_R


def max_activity(neurons, number, neuron_type):
    # returns the index and the names of the highest amplitude #number neurons
    # given a list of neuron activities, the neuron type

    ind = np.argpartition(neurons, -number)[-number:]

    return ind, neuron_lookup[neuron_type][ind]


def make_fixed_allocentric_direction_inputs(allocentric_travel_dir, num_inputs,
                                            min_angle_sweep = -60.0, max_angle_sweep = 60.0):
    # From allocentric_travel_dir in degrees and num_inputs
    # returns a number of movement inputs

    # define the width of angular sweep
    min_angle_sweep = allocentric_travel_dir + min_angle_sweep
    max_angle_sweep = allocentric_travel_dir + max_angle_sweep

    # create a list of headings
    degree_headings = np.linspace(min_angle_sweep, max_angle_sweep, num_inputs, endpoint=True)
    radian_headings = [wrap_to_pi(np.radians(degs)) for degs in degree_headings ]

    # get forward vector vf assuming heading h and egocentric angle 0
    vf_vecs = [[np.cos(h), np.sin(h)] for h in radian_headings]

    # rotation matrix
    rot_mat = [[0.0, -1.0], [1.0, 0.0]]

    # get side vector vs by rotating vf 90 degrees clockwise with respect to the fly
    # but 90 degrees counter-clockwise with respect to an observer underneath
    # the fly looking at the unit circle
    vs_vecs = [np.matmul(rot_mat, vf_vec) for vf_vec in vf_vecs]

    # get unit vector for the world-centric direction of fly travel
    a = np.cos(np.radians(allocentric_travel_dir))
    b = np.sin(np.radians(allocentric_travel_dir))

    # project the forward and side vectors onto the vector of the
    # world-centric direction of travel to get the fly's forward
    # and side velocities
    vf_mag = [np.dot(vf, [a, b]) for vf in vf_vecs]
    vs_mag = [np.dot(vs, [a, b]) for vs in vs_vecs]

    return np.transpose(np.vstack((vf_mag, vs_mag, np.degrees(radian_headings))))


def make_motion_values(num_values, initial_values=[0.0, 0.0, 0.0]):

    vf_diff_values = np.hstack((initial_values[0], VF_DIFF*np.random.poisson(size=num_values)))
    vs_diff_values = np.hstack((initial_values[1], VS_DIFF*np.random.poisson(size=num_values)))
    heading_diff_values = np.hstack((initial_values[2], tu.HEADING_DIFF*np.random.poisson(size=num_values)))

    pos_neg = lambda cut_off : [1.0 if np.random.random() < cut_off else -1.0 for x in range(num_values+1)]

    vf_values = np.cumsum(vf_diff_values*pos_neg(0.8))
    vs_values = np.cumsum(vs_diff_values*pos_neg(0.5))
    heading_values = np.cumsum(heading_diff_values*pos_neg(0.5))

    return np.vstack((vf_values, vs_values, heading_values))

def run_hdeltab_simulation(move_inputs, num_timesteps, bump_param = 0.25, deltaT = 0.0001, tau_HDELB = 0.005, noise_level = 0.0):

    noise_level = noise_level
    noise = noise_level * np.random.normal(0, 1, N_HDELTAB)

    pfndl_history = []
    pfndr_history = []
    pfnvl_history = []
    pfnvr_history = []
    hdeltab_history = []
    pfnd_cont_history = []
    pfnv_cont_history = []

    current_HDELTAB = np.zeros(19)

    for move_input in move_inputs:

        pfndl_over_time = []
        pfndr_over_time = []
        pfnvl_over_time = []
        pfnvr_over_time = []
        hdeltab_over_time = []
        pfnd_cont_over_time = []
        pfnv_cont_over_time = []

        PFND_L, PFND_R = create_PFND(bump_param, move_input, rads=False)
        PFNV_L, PFNV_R = create_PFNV(bump_param, move_input, rads=False)

        for timestep in range(num_timesteps):

            current_hdelb_p1 = wPVH*np.matmul(PFNVR_HDELB, PFNV_R) + wPVH*np.matmul(PFNVL_HDELB, PFNV_L)
            current_hdelb_p2 = wPDH*np.matmul(PFNDR_HDELB, PFND_R) + wPDH*np.matmul(PFNDL_HDELB, PFND_L)
            current_HDELTAB = current_HDELTAB + (-current_HDELTAB + ReLU(current_hdelb_p1 + current_hdelb_p2 + noise))*(deltaT/tau_HDELB)

            pfndl_over_time.append(PFND_L)
            pfndr_over_time.append(PFND_R)
            pfnvl_over_time.append(PFNV_L)
            pfnvr_over_time.append(PFNV_R)
            hdeltab_over_time.append(current_HDELTAB)
            pfnd_cont_over_time.append(current_hdelb_p2)
            pfnv_cont_over_time.append(current_hdelb_p1)

        pfndl_history.append(pfndl_over_time)
        pfndr_history.append(pfndr_over_time)
        pfnvl_history.append(pfnvl_over_time)
        pfnvr_history.append(pfnvr_over_time)
        hdeltab_history.append(hdeltab_over_time)
        pfnd_cont_history.append(pfnd_cont_over_time)
        pfnv_cont_history.append(pfnv_cont_over_time)

    return move_inputs, pfndl_history, pfndr_history, pfnvl_history, pfnvr_history, hdeltab_history, pfnd_cont_history, pfnv_cont_history
