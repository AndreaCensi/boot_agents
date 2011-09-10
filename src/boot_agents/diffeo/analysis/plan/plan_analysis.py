from optparse import OptionParser

from vehicles import instance_vehicle, instance_world
from vehicles import VehicleSimulation
from geometry.yaml import from_yaml
from geometry.poses import SE2_from_SE3, translation_angle_from_SE2
from geometry.manifolds import SE2, SE3
import numpy as np
import contracts
from boot_agents.diffeo.diffeo_agent_2d import popcode
from contracts import contract
from boot_agents.diffeo.diffeo_basic import diffeo_apply

def plan_analysis(global_options, data, args):
    contracts.disable_all()
    usage = ""
    parser = OptionParser(usage=usage)
    parser.disable_interspersed_args()
    parser.add_option("--world", dest='id_world', default="stochastic_box_10",
                       help="Vehicles world to use")
    (options, args) = parser.parse_args(args)
    if args:
        raise Exception('Extra args')
    
    id_robot = data['id_robot']
#    id_agent = data['id_agent']
    pub = data['publisher']
        
    vehicle = instance_vehicle(id_robot)
    world = instance_world(options.id_world)
    
    sim = VehicleSimulation(vehicle, world)
    
    
    FORWARD = [1, 1]
    BACKWARD = [-1, -1]
    LEFT = [-1, +1]
    RIGHT = [+1, -1]
    
    FORWARD = np.array([0, +1])
    BACKWARD = np.array([0, -1])
    LEFT = np.array([+0.3, 0])
    RIGHT = np.array([-0.3, 0])
    FWD_L = np.array([1, +1])
    FWD_R = np.array([1, -1])
    BWD_R = np.array([-1, -1])
    BWD_L = np.array([+1, -1])
    
    T1 = 1
    T2 = 2
    T1 = 1
    dt = 0.02
    def commutator(cmd1, cmd2, T):
        return [(cmd1, T), (cmd2, T), (-cmd1, T), (-cmd2, T)]
    
    examples = {
                'forwd1': {'trajectory': [(FORWARD, T1)]},
##                'forwd2': {'trajectory': [(FORWARD, T2)]},
#                'left1': {'trajectory': [(LEFT, T1)]},
#                'right1': {'trajectory': [(RIGHT, T1)]},
#                'back1': {'trajectory': [(BACKWARD, T1)]},
#                'turnl2': {'trajectory': [(LEFT, T2)]},

#                'l3': {'trajectory': [([0.3, 0], T1)]},
#                'l4': {'trajectory': [([0.4, 0], T1)]},
#                'l5': {'trajectory': [([0.5, 0], T1)]},
#                'l6': {'trajectory': [([0.6, 0], T1)]},
#                'l7': {'trajectory': [([0.7, 0], T1)]},
#                'l8': {'trajectory': [([0.8, 0], T1)]},
#                'l9': {'trajectory': [([0.9, 0], T1)]},
#                'l10': {'trajectory': [([1.0, 0], T1)]},
#                'l1': {'trajectory': [([1, 0], T1)]},
#                          
#                'sidel1': {'trajectory': [(LEFT, T1),
#                                          (FORWARD, T1),
#                                          (RIGHT, T1 * 2),
#                                          (BACKWARD, T1),
#                                          (LEFT, T1) ]},
#                'sidel2': {'trajectory': commutator(FWD_L, LEFT, T1) } 
#                'sidel2': {'trajectory': commutator(FORWARD, LEFT, T1) },
#                'sidel3': {'trajectory': commutator(LEFT, FORWARD, T1) }
#                
    }
    
        
    
    
    for name, scenario in examples.items():
        while True:
            try:
                scenario_compute_inputs(scenario, sim, dt=dt)
                break
            except ValueError as e:
                print(e)
        
    actions = data['actions']
    for name, scenario in examples.items():
        scenario_solve(scenario, actions)

    
    for name, scenario in examples.items():
        S = pub.section(name)
        scenario_display(scenario, S, sim)

def scenario_solve(scenario, actions):
    exploration = {}
    M0 = scenario['M0']
    for action in actions:
        code = '%s' % action
        res = diffeo_apply(action.diffeo.d, M0)
        exploration[code] = res# {'actions' = }
    scenario['exploration'] = exploration
    
    
@contract(y0='array(>=0,<=1)')
def sensels2map(y0):
    y0 = np.maximum(y0, 0)
    y0 = np.minimum(y0, 1)
    y = popcode(y0, 180)
    from scipy.misc import imresize #@UnresolvedImport
    y = imresize(y, (90, 90))
    y = np.array(y, dtype='float32')
    return y

def scenario_display(scenario, S, sim):
    y0 = scenario['y0']
    q0 = scenario['pose0']
    y1 = scenario['y1']
    q1 = scenario['pose1']
    delta = scenario['delta']
    print(SE2.friendly(SE2_from_SE3(q0)))
    print(SE2.friendly(SE2_from_SE3(q1)))
    print('increment: %s' % SE2.friendly(SE2_from_SE3(delta)))
    
    
    with S.plot('data') as pylab:
        pylab.plot(y0, label='y0')
        pylab.plot(y1, label='y1')
        pylab.axis((0, 180, -0.04, 1.04))
    
    with S.plot('world') as pylab:
        show_sensor_data(pylab, scenario['sim0']['vehicle'], col='r')
        show_sensor_data(pylab, scenario['sim1']['vehicle'], col='g')
        pylab.axis('equal')
    
#    for pose in scenario['poses']:
#        delta = pose_diff(q0, pose)
#        print('increment: %s' % SE2.friendly(SE2_from_SE3(delta)))
#        
    S.array_as_image('M0', scenario['M0'])
    S.array_as_image('M1', scenario['M1'])
    
    with S.plot('poses') as pylab:
#        for pose in scenario['poses']:
##            print pose
#            draw_axes(pylab, SE2_from_SE3(pose))
#        
        for pose in scenario['poses']:
            draw_axes(pylab, SE2_from_SE3(pose), 'k', 'k', size=0.5)
        for pose in scenario['poses_important']:
            draw_axes(pylab, SE2_from_SE3(pose), 'k', 'k', size=1)
        draw_axes(pylab, SE2_from_SE3(q0), [0.3, 0, 0], [0, 0.3, 0], size=5)
        draw_axes(pylab, SE2_from_SE3(q1), 'r', 'g', size=5)
        

#        plot_sensor(pylab, sim.vehicle, q0, y0, 'g')
#        plot_sensor(pylab, sim.vehicle, q1, y1, 'b')
        pylab.axis('equal')
        
#    print(scenario['commands'])

    Se = S.section('exploration')
    for name, M1est in scenario['exploration'].items():
        Si = Se.section(name)
        Si.array_as_image('M1est', M1est)
        Si.array_as_image('diff', M1est - scenario['M1'])
        
        
def draw_axes(pylab, pose, cx='r', cy='g', size=1, L=0.3):
    t, th = translation_angle_from_SE2(pose)
    
    tx = [t[0] + L * np.cos(th),
          t[1] + L * np.sin(th)]
    ty = [t[0] + L * -np.sin(th),
          t[1] + L * np.cos(th)]
    
    pylab.plot([t[0], tx[0]],
               [t[1], tx[1]], '-', color=cx, linewidth=size)
    pylab.plot([t[0], ty[0]],
               [t[1], ty[1]], '-', color=cy, linewidth=size)

def plot_sensor(pylab, vehicle, pose, readings, color):
    show_sensor_data(pylab, vehicle.to_yaml())

def pose_diff(a, b):
    return SE3.multiply(SE3.inverse(a), b)
    
def scenario_compute_inputs(scenario, sim, dt=0.1):
    sim.new_episode()
    scenario['y0'] = sim.compute_observations()
    scenario['pose0'] = sim.vehicle.get_pose()
    scenario['sim0'] = sim.to_yaml()
    
    scenario['poses'] = []
    scenario['poses_important'] = []
    scenario['commands'] = []

    scenario['poses_important'].append(sim.vehicle.get_pose())

    for command, time in scenario['trajectory']:
        num = time / dt
        for _ in range(int(num)):
            scenario['poses'].append(sim.vehicle.get_pose())
            sim.simulate(command, dt) # XXX: smaller steps?
            if sim.vehicle_collided:
                raise ValueError('Collision; must restart.')
            scenario['commands'].append(command)
        scenario['poses'].append(sim.vehicle.get_pose())
        scenario['poses_important'].append(sim.vehicle.get_pose())
        
    scenario['y1'] = sim.compute_observations()
    scenario['pose1'] = sim.vehicle.get_pose()
    scenario['sim1'] = sim.to_yaml()    
    scenario['delta'] = pose_diff(scenario['pose0'], scenario['pose1'])

    scenario['M0'] = sensels2map(scenario['y0'])
    scenario['M1'] = sensels2map(scenario['y1'])
    

def show_sensor_data(pylab, vehicle, robot_pose=None, col='r'):
    if robot_pose is None:
        robot_pose = from_yaml(vehicle['pose'])
    for attached in vehicle['sensors']:
        sensor_pose = from_yaml(attached['current_pose'])
        sensor_t, sensor_theta = \
            translation_angle_from_SE2(SE2_from_SE3(sensor_pose))
        print('robot: %s' % SE2.friendly(SE2_from_SE3(robot_pose)))
        print(' sens: %s' % SE2.friendly(SE2_from_SE3(sensor_pose)))
#        sensor_theta = -sensor_theta
        sensor = attached['sensor']
        if sensor['type'] == 'Rangefinder':
            directions = np.array(sensor['directions'])
            observations = attached['current_observations']
            readings = np.array(observations['readings'])
            valid = np.array(observations['valid'])
#            directions = directions[valid]
#            readings = readings[valid]
            x = []
            y = []
            rho_min = 0.05
            for theta_i, rho_i in zip(directions, readings):
                print('theta_i: %s' % theta_i)
                x.append(sensor_t[0] + np.cos(sensor_theta + theta_i) * rho_min)
                y.append(sensor_t[1] + np.sin(sensor_theta + theta_i) * rho_min)
                x.append(sensor_t[0] + np.cos(sensor_theta + theta_i) * rho_i)
                y.append(sensor_t[1] + np.sin(sensor_theta + theta_i) * rho_i)
                x.append(None)
                y.append(None)
            pylab.plot(x, y, color=col, markersize=0.5, zorder=2000)
        elif sensor['type'] == 'Photoreceptors':
            directions = np.array(sensor['directions'])
            observations = attached['current_observations']
            readings = np.array(observations['readings'])
            luminance = np.array(observations['luminance'])
            valid = np.array(observations['valid'])
            readings[np.logical_not(valid)] = 0.6
            rho_min = 0.5
            for theta_i, rho_i, lum in zip(directions, readings, luminance):
                x = []
                y = []
                x.append(sensor_t[0] + np.cos(sensor_theta + theta_i) * rho_min)
                y.append(sensor_t[1] + np.sin(sensor_theta + theta_i) * rho_min)
                x.append(sensor_t[0] + np.cos(sensor_theta + theta_i) * rho_i)
                y.append(sensor_t[1] + np.sin(sensor_theta + theta_i) * rho_i)
                pylab.plot(x, y, color=(lum, lum, lum), markersize=0.5, zorder=2000)
        else:
            print('Unknown sensor type %r' % sensor['type'])
    
