from taichi_utils import *
import numpy as np
from init_conditions import *
case = 0
dim = 3

@ti.kernel
def no_mask(boundary_mask:ti.template(), boundary_vel:ti.template(), boundary_normal_vel:ti.template(),  _t: float):
    pass


if case == 0:
    # encoder hyperparameters
    use_midpoint_vel = True
    use_APIC = True
    save_frame_each_step = False
    reinit_particle_pos = True
    projFT = False
    use_hessian = True

    res_x = 127
    res_y = 63
    res_z = 127
    dx = 1.0/res_x
    inv_dx = res_x

    visualize_dt = 0.05
    reinit_every = 20
    reinit_every_grad_m = 1
    CFL = 0.5
    from_frame = 0
    total_frames = 150

    limiter = 1
    particles_per_cell = 8
    total_particles_num_ratio = 1
    cell_max_particle_num_ratio = 1.6
    total_steps = 100
    set_boundary = no_mask
    use_noslip = True
    alpha = 0.3
    interp_by_grid = True

    x_wall_v = -0.1
    y_wall_v = 0.0
    z_wall_v = 0.0
    normalize_detFT = False
    mesh_name = "plesiosaur_four"
    motion_seq_name = "plesiosaur_fourlimb_animation"
    exp_name = "3D_plesiosaur_" +str(CFL) + "_" + str(reinit_every) + "_" + str(reinit_every_grad_m) + "_alpha_" + str(alpha)