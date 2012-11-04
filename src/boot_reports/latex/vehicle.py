from . import get_resources_dir, load_report_robot
from bootstrapping_olympics.utils import system_run
from latex_gen import latex_fragment
import os
import sys


def bvvehicleimage(id_set, id_robot, width_em=3.5):
    with latex_fragment(sys.stdout, graphics_path=get_resources_dir()) as frag:
        create_robot_figure(frag, id_set, id_robot, width_em, write_world=False)

def create_robot_figure(frag, id_set, id_robot, width=3.5, write_world=False):
    width_mine = width * 2
    with frag.minipage_bottom('%sem' % width_mine) as m:
        create_robot_figure_inside(m, id_set, id_robot, width=width,
                                   write_world=write_world)

def create_robot_figure2(frag, id_set, id_robot, width=3.5):
    width_mine = width * 2
    with frag.minipage_bottom('%sem' % width_mine) as m:
        m.tex('\\centering\n')
        create_robot_figure_inside(m, id_set, id_robot, width,
                                      write_world=False)
        m.parbreak()
        m.medskip()
        m.tex('\\footnotesize\\bvid{%s}' % id_robot)


def create_robot_figure_inside(frag, id_set, id_robot, width=3.5, write_world=True):
    report = load_report_robot(id_set, id_robot)
    
    id_world = report['environment/id_world'].raw_data
    #id_vehicle = report['vehicle/id_vehicle'].raw_data
    
    has_nuisances = report.has_child('nuisances')
    
    if has_nuisances:
        obs_nuisances = report['nuisances/observations'].raw_data
        for obs_id in obs_nuisances:
            frag.tex('$\\bvid{%s} \\cdot$' % obs_id)
    
    # TODO: check vehicle
    vehicle = report['vehicle/body_data_compact']
    
    raiselen = '%.2fem' % (-width / 2.0 + 0.5) # 0.5 align middle
    width = '%.2fem' % width
    frag.tex('\\raisebox{%s}{%%\n' % raiselen)
    gid = '%s-%s-body_data_compact' % (id_set, id_robot)
    frag.graphics_data(vehicle.raw_data, vehicle.mime, width=width, gid=gid)
    frag.tex('}%\n')
    
    filename = os.path.join(get_resources_dir(), '%s.pdf' % gid)
    system_run(".", 'pdfcrop %s %s' % (filename, filename))

    if has_nuisances:
        cmd_nuisances = report['nuisances/commands'].raw_data
        for cmd_id in cmd_nuisances:
            frag.tex('$ \\cdot \\bvid{%s}$' % cmd_id)
    
    if write_world:
        frag.tex('$ \\leftrightarrow \\bvid{%s} $' % id_world)
    
    
