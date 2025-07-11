#
from hyperparameters import *
from taichi_utils import *
from init_conditions import *
from io_utils import *
import sys
import matplotlib.pyplot as plt
import igl
from my_mesh_interpolation import *
from my_lbvh import *
import torch
import time
# from poisson_solver_permenant_stream import PoissonSolver
# from amgpcg_pybind_taichi import PoissonSolver
from amgpcg_pybind import AMGPCGTorch

#
half_dx = dx * 0.5
upper_boundary = res_y * dx
lower_boundary = 0.
right_boundary = res_x * dx
left_boundary = 0.
back_boundary = res_z * dx
front_boundary = 0.
tile_size = 8

ti.init(arch=ti.cuda, device_memory_GB = 8, debug=False)
particles_per_cell_axis = 2
dist_between_neighbor = dx / particles_per_cell_axis

center_boundary_mask = ti.field(ti.u8, shape=(res_x, res_y, res_z))
center_boundary_mask_extend = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z + 1))
face_x_boundary_mask = ti.field(ti.u8, shape=(res_x+1, res_y, res_z))
face_y_boundary_mask = ti.field(ti.u8, shape=(res_x, res_y+1, res_z))
face_z_boundary_mask = ti.field(ti.u8, shape=(res_x, res_y, res_z + 1))

edge_x_boundary_mask = ti.field(ti.u8, shape=(res_x, res_y+1, res_z+1))
edge_y_boundary_mask = ti.field(ti.u8, shape=(res_x+1, res_y, res_z+1))
edge_z_boundary_mask = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z))

edge_x_boundary_mask_extend = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))
edge_y_boundary_mask_extend = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))
edge_z_boundary_mask_extend = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))

reverse_mask_extend = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))
affected_by_solid_penalty_w_x = ti.field(ti.u8, shape=(res_x, res_y+1, res_z+1))
affected_by_solid_penalty_w_y = ti.field(ti.u8, shape=(res_x+1, res_y, res_z+1))
affected_by_solid_penalty_w_z = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z))

bv_x = ti.field(float, shape=(res_x+1, res_y, res_z))
bv_y = ti.field(float, shape=(res_x, res_y+1, res_z))
bv_z = ti.field(float, shape=(res_x, res_y, res_z+1))

bv_x_tan = ti.field(float, shape=(res_x+1, res_y, res_z))
bv_y_tan = ti.field(float, shape=(res_x, res_y+1, res_z))
bv_z_tan = ti.field(float, shape=(res_x, res_y, res_z+1))

surf_face_mask_x = ti.field(ti.u8, shape=(res_x+1, res_y, res_z))
surf_face_mask_y = ti.field(ti.u8, shape=(res_x, res_y+1, res_z))
surf_face_mask_z = ti.field(ti.u8, shape=(res_x, res_y, res_z + 1))
surf_stream_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
surf_stream_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
surf_stream_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))

stream_x_extend = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z + 1))
stream_y_extend = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z + 1))
stream_z_extend = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z + 1))


p_ext = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z + 1))
one_sixth = 1. / 6

boundary_vel = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
boundary_vel_tan = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))

mesh_prev_v, plesiosaur_faces = igl.read_triangle_mesh(f'./{motion_seq_name}/{mesh_name}{0}.obj')
vn, fn = mesh_prev_v.shape[0], plesiosaur_faces.shape[0]
ti_vertices = ti.Vector.field(3, ti.f32, shape = (vn))
ti_new_vertices = ti.Vector.field(3, ti.f32, shape = (vn))
mesh_vel = ti.Vector.field(3, ti.f32, shape = (vn))

ti_faces_0 = ti.Vector.field(3, int, shape = (fn))
changes = ti.field(ti.i32, shape=())
surf_occupancy = ti.field(ti.u8, shape = (res_x, res_y, res_z))
out_occupancy = ti.field(ti.u8, shape = (res_x, res_y, res_z))
in_occupancy = ti.field(ti.u8, shape = (res_x, res_y, res_z))
in_occupancy_ext = ti.field(ti.u8, shape=(res_x + 1, res_y + 1, res_z + 1))
max_unknowns_nx = (res_x + 1) * (res_y) * (res_z)
max_unknowns_ny = (res_x) * (res_y + 1) * (res_z)
max_unknowns_nz = (res_x) * (res_y) * (res_z + 1)

surf_face_fraction_x = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z))
surf_face_fraction_y = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z))
surf_face_fraction_z = ti.field(ti.f32, shape=(res_x, res_y, res_z + 1))

surf_face_fraction_x_ext = ti.field(ti.f32, shape=(res_x + 2, res_y + 1, res_z + 1))
surf_face_fraction_y_ext = ti.field(ti.f32, shape=(res_x + 1, res_y + 2, res_z + 1))
surf_face_fraction_z_ext = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z + 2))

nx2ijk = ti.Vector.field(3, dtype=ti.i32, shape=max_unknowns_nx)
ny2ijk = ti.Vector.field(3, dtype=ti.i32, shape=max_unknowns_ny)
nz2ijk = ti.Vector.field(3, dtype=ti.i32, shape=max_unknowns_nz)
nx = ti.field(ti.i32, shape=())
ny = ti.field(ti.i32, shape=())
nz = ti.field(ti.i32, shape=())


tile_dim_vec = [(res_x+1)//8, (res_y+1)//8, (res_z+1)//8]

X = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
X_x = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z))
X_y = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z))
X_z = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z + 1))
X_x_e = ti.Vector.field(3, ti.f32, shape=(res_x, res_y+ 1, res_z+ 1))
X_y_e = ti.Vector.field(3, ti.f32, shape=(res_x+ 1, res_y, res_z+ 1))
X_z_e = ti.Vector.field(3, ti.f32, shape=(res_x+ 1, res_y+ 1, res_z))
center_coords_func(X, dx)
x_coords_func(X_x, dx)
y_coords_func(X_y, dx)
z_coords_func(X_z, dx)
x_coords_func_edge(X_x_e, dx)
y_coords_func_edge(X_y_e, dx)
z_coords_func_edge(X_z_e, dx)

num_extra_particles = 0
initial_particle_num = res_x * res_y * res_z * particles_per_cell
particle_num = initial_particle_num * total_particles_num_ratio + num_extra_particles
particles_active = ti.field(int, shape=particle_num)
particles_pos = ti.Vector.field(3, float, shape=particle_num)
particles_mid_w = ti.Vector.field(3, float, shape=particle_num)
particles_init_w = ti.Vector.field(3, float, shape=particle_num)

# back flow map
T_x_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
T_y_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
T_z_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z
T_x_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
T_y_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
T_z_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z



F_x_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
F_y_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
F_z_init = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z
F_x_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_x
F_y_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_y
F_z_grad_m = ti.Vector.field(3, float, shape=particle_num) # d_psi / d_z

gradF_j0_particles = ti.Matrix.field(3, 3, float, shape=particle_num) # d_psi / d_x
gradF_j1_particles = ti.Matrix.field(3, 3, float, shape=particle_num) # d_psi / d_y
gradF_j2_particles = ti.Matrix.field(3, 3, float, shape=particle_num) # d_psi / d_z

psi_x = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z))  # x coordinate
psi_y = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z))  # y coordinate
psi_z = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z + 1))  # z coordinate


# fwrd flow map
F_x_tmp = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x + 1, res_y, res_z))  # d_phi / d_x
F_y_tmp = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x, res_y + 1, res_z))  # d_phi / d_y
F_z_tmp = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=(res_x, res_y, res_z + 1))  # d_phi / d_z

# velocity storage
u = ti.Vector.field(3, float, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, float, shape=(res_x, res_y, res_z)) # curl of u
u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
penalty_u_x = ti.field(float, shape=(res_x+1, res_y, res_z))
penalty_u_y = ti.field(float, shape=(res_x, res_y+1, res_z))
penalty_u_z = ti.field(float, shape=(res_x, res_y, res_z+1))
# P2G weight storage
p2g_weight = ti.field(float, shape=(res_x, res_y, res_z))
p2g_weight_x = ti.field(float, shape=(res_x, res_y + 1, res_z + 1))
p2g_weight_y = ti.field(float, shape=(res_x + 1, res_y, res_z + 1))
p2g_weight_z = ti.field(float, shape=(res_x + 1, res_y + 1, res_z))
# APIC
init_C_x = ti.Vector.field(3, float, shape=particle_num)
init_C_y = ti.Vector.field(3, float, shape=particle_num)
init_C_z = ti.Vector.field(3, float, shape=particle_num)


# velocity storage
u = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
w = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))  # curl of u
div_w = ti.field(ti.f32, shape = (res_x+1, res_y+1, res_z+1))
u_x = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z))
u_y = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z))
u_z = ti.field(ti.f32, shape=(res_x, res_y, res_z + 1))

# some helper storage for u
tmp_u_x = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z))
tmp_u_y = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z))
tmp_u_z = ti.field(ti.f32, shape=(res_x, res_y, res_z + 1))
# vorticity storage (on face fields)
w_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
w_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
w_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))

stream_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
stream_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
stream_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))

penalty_w_x = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z + 1))
penalty_w_y = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z + 1))
penalty_w_z = ti.field(ti.f32, shape=(res_x + 1, res_y + 1, res_z))


edge_x_boundary_mask_noin = ti.field(ti.u8, shape=(res_x, res_y+1, res_z+1))
edge_y_boundary_mask_noin = ti.field(ti.u8, shape=(res_x+1, res_y, res_z+1))
edge_z_boundary_mask_noin = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z))
edge_x_boundary_mask_extend_noin = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))
edge_y_boundary_mask_extend_noin = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))
edge_z_boundary_mask_extend_noin = ti.field(ti.u8, shape=(res_x+1, res_y+1, res_z+1))


# CFL related
max_speed = ti.field(ti.f32, shape=())
min_w = ti.field(ti.f32, shape=())
num_particles_small_w = ti.field(ti.i32, shape=())

b_pretorch = ti.field(ti.f32, shape=(tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8))
a_diag_pretorch = ti.field(ti.f32, shape=(tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8))
a_x_pretorch = ti.field(ti.f32, shape=(tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8))
a_y_pretorch = ti.field(ti.f32, shape=(tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8))
a_z_pretorch = ti.field(ti.f32, shape=(tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8))
# is_dof_pretorch = ti.field(ti.u8, shape=(tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8))
b_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.float32, device="cuda:0", memory_format=torch.contiguous_format)
a_diag_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.float32, device="cuda:0", memory_format=torch.contiguous_format)
a_x_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.float32, device="cuda:0", memory_format=torch.contiguous_format)
a_y_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.float32, device="cuda:0", memory_format=torch.contiguous_format)
a_z_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.float32, device="cuda:0", memory_format=torch.contiguous_format)
is_dof_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.uint8, device="cuda:0", memory_format=torch.contiguous_format)
x_pytorch = torch.empty((tile_dim_vec[0]*8, tile_dim_vec[1]*8, tile_dim_vec[2]*8), dtype=torch.float32, device="cuda:0", memory_format=torch.contiguous_format)

edge_x_boundary_mask_noin.fill(0)
edge_y_boundary_mask_noin.fill(0)
edge_z_boundary_mask_noin.fill(0)
edge_x_boundary_mask_extend_noin.fill(0)
edge_y_boundary_mask_extend_noin.fill(0)
edge_z_boundary_mask_extend_noin.fill(0)

edge_noin_mask(edge_x_boundary_mask_noin, edge_y_boundary_mask_noin, edge_z_boundary_mask_noin)
extend_boundary_field(edge_x_boundary_mask_noin, edge_x_boundary_mask_extend_noin)
extend_boundary_field(edge_y_boundary_mask_noin, edge_y_boundary_mask_extend_noin)
extend_boundary_field(edge_z_boundary_mask_noin, edge_z_boundary_mask_extend_noin)

raw_res = [res_x+1, res_y+1, res_z+1]
bottom_smoothing = 40
verbose = False
tile_size = 8
# tile_dim = [
#     (raw_res[0] + tile_size - 1) // tile_size,
#     (raw_res[1] + tile_size - 1) // tile_size,
#     (raw_res[2] + tile_size - 1) // tile_size,
# ]

tile_dim = tile_dim_vec

amgpcg_torch = AMGPCGTorch(
    tile_dim, bottom_smoothing, verbose, False, False
)

@ti.kernel
def construct_adiag(a_diag:ti.template(), axis:int):
    dims = ti.Vector(a_diag.shape)
    # print(dims)
    for I in ti.grouped(a_diag):
        a_diag[I] = 6.0

@ti.kernel
def construct_aadj(a_adj:ti.template(), bm:ti.template(), axis:int):
    dims = ti.Vector(a_adj.shape)
    offset = ti.Vector.unit(3, axis)
    for I in ti.grouped(a_adj):
        num = -1.0
        if (I[axis] >= dims[axis] - 1) or bm[I] >= 1 or bm[I+offset] >= 1:
            num = 0.0
        # if (I[axis] <= 0) or bm[I-offset] >= 1:
        #     num = 0.0
        a_adj[I] = num

@ti.kernel
def construct_adiag_harmonic(a_diag:ti.template(), bm:ti.template()):
    dims = a_diag.shape
    for I in ti.grouped(a_diag):
        num = 6.0
        if bm[I] <= 0:
            for i in ti.static(range(3)):
                offset = ti.Vector.unit(3, i)
                if I[i] <= 0:
                    num -= 1.0
                elif bm[I-offset] >= 1:
                    num -= 1.0

                if I[i] >= dims[i] - 1:
                    num -= 1.0
                elif bm[I+offset] >= 1:
                    num -= 1.0
        a_diag[I] = num

@ti.kernel
def construct_adiag_harmonic_cutcell(a_diag:ti.template(), inner_bm:ti.template(),
                                     frx:ti.template(), fry:ti.template(), frz:ti.template()):
    dims = a_diag.shape
    for I in ti.grouped(a_diag):
        num = 0.0
        if inner_bm[I] <= 0 and (frx[I] >0 or frx[I+ti.Vector.unit(3, 0)] >0 or fry[I] >0 or fry[I+ti.Vector.unit(3, 1)]>0 or frz[I]>0 or frz[I+ti.Vector.unit(3, 2)]>0):
            for i in ti.static(range(3)):
                offset = ti.Vector.unit(3, i)
                if i==0:
                    num += (frx[I] + frx[I+offset])
                elif i==1:
                    num += (fry[I] + fry[I+offset])
                elif i==2:
                    num += (frz[I] + frz[I+offset])
        else:
            num = 6.0
        # print(num)
        a_diag[I] = num

@ti.kernel
def construct_aadj_cutcell(a_adj:ti.template(), inner_bm:ti.template(), fraction:ti.template(), axis:int):
    offset = ti.Vector.unit(3, axis)
    for I in ti.grouped(a_adj):
        a_adj[I] = - fraction[I+offset]

@ti.kernel
def setup_rhs_harmonic_cutcell(b:ti.types.ndarray(), inner_bm:ti.template(), surf_bm:ti.template(),
                        bv_x:ti.template(), bv_y:ti.template(), bv_z:ti.template(),
                       u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), 
                       frx:ti.template(), fry:ti.template(), frz:ti.template(), dx:float):
    shapex, shapey, shapez = inner_bm.shape
    for I in ti.grouped(inner_bm):
        ret = 0.0
        if inner_bm[I] <= 0:
            if ((surf_bm[I] >= 1 or I[0] == 0 or I[1] == 0 or I[2] == 0 or I[0] == shapex - 2 or I[1] == shapey-2 or I[2] == shapez-2)
                and (frx[I] >0 or frx[I+ti.Vector.unit(3, 0)] >0 or fry[I] >0 or fry[I+ti.Vector.unit(3, 1)]>0 or frz[I]>0 or frz[I+ti.Vector.unit(3, 2)]>0)):
                for i in ti.static(range(3)):
                    offset = ti.Vector.unit(3, i)
                    fr_n = choose_ax(i, I, frx, fry, frz)
                    fr_p = choose_ax(i, I+offset, frx, fry, frz)
                    ret += (fr_n * choose_ax(i, I, u_x, u_y, u_z) + (1.0 - fr_n) * choose_ax(i, I, bv_x, bv_y, bv_z))
                    ret -= (fr_p * choose_ax(i, I+offset, u_x, u_y, u_z) + (1.0 - fr_p) * choose_ax(i, I+offset, bv_x, bv_y, bv_z))
        b[I] = ret*dx

def rearrange_tensor_for_cuda(tensor, tile_dim_vec, tile_size=8, result_tensor=None):

    tensor_reshaped = tensor.view(
        tile_dim_vec[0], tile_size,
        tile_dim_vec[1], tile_size,
        tile_dim_vec[2], tile_size
    )

    tensor_reordered = tensor_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
    tensor_tiles = tensor_reordered.view(-1, tile_size ** 3)
    tensor_flat = tensor_tiles.view(-1)
    result_tensor.copy_(tensor_flat)
    
    return tensor_flat


def rearrange_tensor_from_cuda(tensor_flat, original_shape, tile_dim_vec, tile_size=8):
    tile_num = tile_dim_vec[0] * tile_dim_vec[1] * tile_dim_vec[2]
    tensor = tensor_flat.view(tile_num, tile_size ** 3)
    tensor = tensor.view(
        tile_dim_vec[0], tile_dim_vec[1], tile_dim_vec[2],
        tile_size, tile_size, tile_size
    )
    tensor = tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
    tensor = tensor.view(original_shape)
    return tensor


def rtf_solve(stream, stream_ext, w, edge_mask, edge_mask_extend, axis, amgpcg_torch):
    # prepare for lhs
    construct_adiag(a_diag_pretorch, axis)
    construct_aadj(a_x_pretorch, edge_mask_extend, 0)
    construct_aadj(a_y_pretorch, edge_mask_extend, 1)
    construct_aadj(a_z_pretorch, edge_mask_extend, 2)

    # prepare for rhs
    reverse_mask(edge_mask_extend, reverse_mask_extend)
    extend_to_pretorch(w, b_pretorch)
    scale_field(b_pretorch, dx * dx, b_pretorch)

    copy_to_external(b_pretorch, b_pytorch)
    copy_to_external(a_diag_pretorch, a_diag_pytorch)
    copy_to_external(a_x_pretorch, a_x_pytorch)
    copy_to_external(a_y_pretorch, a_y_pytorch)
    copy_to_external(a_z_pretorch, a_z_pytorch)
    copy_to_external(reverse_mask_extend, is_dof_pytorch)
    ti.sync()

    amgpcg_torch.load_coeff(
        a_diag_pytorch,
        a_x_pytorch,
        a_y_pytorch,
        a_z_pytorch,
        is_dof_pytorch
    )

    amgpcg_torch.build(6.0, -1.0)
    amgpcg_torch.load_rhs(b_pytorch)

    torch.zero_(x_pytorch)
    amgpcg_torch.solve(x_pytorch,
            pure_neumann = False)

    copy_from_external(stream_ext, x_pytorch)
    back_extend_field(stream, stream_ext)
    mtply_reversemask(stream, edge_mask, stream, 1.0)


def rtf_solve_harmonic_cutcell(bv_x, bv_y, bv_z,
                        u_x, u_y, u_z, boundary_vel, p_ext,
                        x_wall_v, y_wall_v, z_wall_v, 
                        inner_boundary_mask_extend,
                        surf_occupancy,
                        amgpcg_torch):
    # prepare for lhs
    construct_adiag_harmonic_cutcell(a_diag_pretorch, inner_boundary_mask_extend, surf_face_fraction_x_ext,
                                     surf_face_fraction_y_ext, surf_face_fraction_z_ext)
    construct_aadj_cutcell(a_x_pretorch, inner_boundary_mask_extend, surf_face_fraction_x_ext, 0)
    construct_aadj_cutcell(a_y_pretorch, inner_boundary_mask_extend, surf_face_fraction_y_ext, 1)
    construct_aadj_cutcell(a_z_pretorch, inner_boundary_mask_extend, surf_face_fraction_z_ext, 2)

    reverse_mask(inner_boundary_mask_extend, reverse_mask_extend)
    setup_rhs_harmonic_cutcell(b_pytorch, inner_boundary_mask_extend, surf_occupancy,
                               bv_x, bv_y, bv_z, u_x, u_y, u_z, 
                                surf_face_fraction_x_ext, surf_face_fraction_y_ext, 
                                surf_face_fraction_z_ext, 
                                dx)

    copy_to_external(a_diag_pretorch, a_diag_pytorch)
    copy_to_external(a_x_pretorch, a_x_pytorch)
    copy_to_external(a_y_pretorch, a_y_pytorch)
    copy_to_external(a_z_pretorch, a_z_pytorch)
    copy_to_external(reverse_mask_extend, is_dof_pytorch)

    ti.sync()

    amgpcg_torch.load_coeff(
        a_diag_pytorch,
        a_x_pytorch,
        a_y_pytorch,
        a_z_pytorch,
        is_dof_pytorch
    )

    amgpcg_torch.build(6.0, -1.0)
    amgpcg_torch.load_rhs(b_pytorch)

    torch.zero_(x_pytorch)
    amgpcg_torch.solve(x_pytorch,
            pure_neumann = True)
    
    copy_from_external(p_ext, x_pytorch)
    scale_field(p_ext, inv_dx, p_ext)
    subtract_grad_p(u_x, u_y, u_z, p_ext, inner_boundary_mask_extend, surf_face_fraction_x, surf_face_fraction_y,
                                surf_face_fraction_z, x_wall_v, y_wall_v, z_wall_v)
    enforce_surface_vel(center_boundary_mask, boundary_vel,
    u_x, u_y, u_z, x_wall_v, y_wall_v, z_wall_v)


@ti.func
def judge_inside_w_boud(i:int, j:int, boundary_mask:ti.template()):
    ret = boundary_mask[i, j] + boundary_mask[i-1, j] + boundary_mask[i, j-1] + boundary_mask[i-1, j-1]
    if ret>0 and ret<4:
        ret = 1
    else:
        ret = 0
    return ret

@ti.kernel
def mask_smoke(smoke: ti.template()):
    for I in ti.grouped(smoke):
        if center_boundary_mask[I] > 0:
            smoke[I] = ti.Vector([0.,0.,0.,1.])
        # else:
        #     smoke[I] *= 0

@ti.kernel
def mask_by_boundary(field: ti.template()):
    for I in ti.grouped(field):
        if center_boundary_mask[I] > 0:
            field[I] *= 0


@ti.kernel
def calc_max_speed(u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
    max_speed[None] = 1.0e-3  # avoid dividing by zero
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        u = 0.5 * (u_x[i, j, k] + u_x[i + 1, j, k])
        v = 0.5 * (u_y[i, j, k] + u_y[i, j + 1, k])
        w = 0.5 * (u_z[i, j, k] + u_z[i, j, k + 1])
        speed = ti.sqrt(u**2 + v**2 + w**2)
        ti.atomic_max(max_speed[None], speed)

@ti.kernel
def calc_min_w(w_x: ti.template(), w_y: ti.template(), w_z: ti.template()):
    min_w[None] = 1.0e-3  # avoid dividing by zero
    udim, vdim, wdim = w_x.shape
    for i, j, k in w_x:
        if not on4bound(i,j,k,udim,vdim,wdim):
            ti.atomic_min(min_w[None], abs(w_x[i,j,k]))

    udim, vdim, wdim = w_y.shape
    for i, j, k in w_y:
        if not on4bound(i,j,k,udim,vdim,wdim):
            ti.atomic_min(min_w[None], abs(w_y[i,j,k]))

    udim, vdim, wdim = w_z.shape
    for i, j, k in w_z:
        if not on4bound(i,j,k,udim,vdim,wdim):
            ti.atomic_min(min_w[None], abs(w_z[i,j,k]))

# set to undeformed config
@ti.kernel
def reset_to_identity(
    psi_x: ti.template(),
    psi_y: ti.template(),
    psi_z: ti.template(),
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
):
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x_e[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y_e[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z_e[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Matrix.identity(n=3, dt=ti.f32)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Matrix.identity(n=3, dt=ti.f32)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Matrix.identity(n=3, dt=ti.f32)

@ti.kernel
def limit_particles_in_boundary():
    for i in particles_pos:
        if particles_pos[i][0] < left_boundary:
            particles_pos[i][0] = left_boundary
        if particles_pos[i][0] > right_boundary:
            particles_pos[i][0] = right_boundary

        if particles_pos[i][1] < lower_boundary:
            particles_pos[i][1] = lower_boundary
        if particles_pos[i][1] > upper_boundary:
            particles_pos[i][1] = upper_boundary

        if particles_pos[i][2] < front_boundary:
            particles_pos[i][2] = front_boundary
        if particles_pos[i][2] > back_boundary:
            particles_pos[i][2] = back_boundary

@ti.kernel
def reset_to_identity_T(
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
):
    for I in ti.grouped(T_x):
        T_x[I] = ti.Matrix.identity(n=3, dt=ti.f32)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Matrix.identity(n=3, dt=ti.f32)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Matrix.identity(n=3, dt=ti.f32)

@ti.kernel
def reset_T_to_identity(T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)


@ti.func
def interp_u_MAC_grad(u_x, u_y, u_z, p, dx):
    u_x_p, grad_u_x_p = interp_grad_2(u_x, p, inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    u_y_p, grad_u_y_p = interp_grad_2(u_y, p, inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    u_z_p, grad_u_z_p = interp_grad_2(u_z, p, inv_dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p]), ti.Matrix.rows(
        [grad_u_x_p, grad_u_y_p, grad_u_z_p]
    )

@ti.func
def interp_w_MAC(w_x, w_y, w_z, p, dx):
    w_x_p = interp_2(w_x, p, inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
    w_y_p = interp_2(w_y, p, inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
    w_z_p = interp_2(w_z, p, inv_dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
    return ti.Vector([w_x_p, w_y_p, w_z_p])

@ti.kernel
def estimate_distortion(psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
                    phi_x: ti.template(), phi_y: ti.template(), phi_z: ti.template()) -> ti.f32:
    max_d = 0.0
    for i, j, k in phi_x:
        if j > 0 and j < res_y and k > 0 and k < res_z:
            back_x = interp_2_v(psi_x, phi_x[i, j, k], inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
            dist = back_x - ti.Vector([i, j, k]) * dx
            d = ti.sqrt(dist.dot(dist))
            if (d > max_d):
                max_d = d
    for i, j, k in phi_y:
        if i > 0 and i < res_x and k > 0 and k < res_z:
            back_x = interp_2_v(psi_y, phi_y[i, j, k], inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
            dist = back_x - ti.Vector([i, j, k]) * dx
            d = ti.sqrt(dist.dot(dist))
            if (d > max_d):
                max_d = d
    for i, j, k in phi_z:
        if i > 0 and i < res_x and j > 0 and j < res_y:
            back_x = interp_2_v(psi_z, phi_z[i, j, k], inv_dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
            dist = back_x - ti.Vector([i, j, k]) * dx
            d = ti.sqrt(dist.dot(dist))
            if (d > max_d):
                max_d = d
    return max_d

@ti.kernel
def RK4_grid_graduT_psiF(
    psi_x: ti.template(),
    T_x: ti.template(),
    u_x0: ti.template(),
    u_y0: ti.template(),
    u_z0: ti.template(),
    dt: ti.f32,
):

    # neg_dt = -1 * dt  # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = T_x[I] @ grad_u_at_psi  # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] - 0.5 * dt * u1  # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = T_x1 @ grad_u_at_psi  # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] - 0.5 * dt * u2  # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * dt * dT_x_dt2
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = T_x2 @ grad_u_at_psi # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] - 1.0 * dt * u3
        T_x3 = T_x[I] + 1.0 * dt * dT_x_dt3  # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = T_x3 @ grad_u_at_psi  # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] - dt * 1.0 / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + dt * 1.0 / 6 * (
            dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4
        )  # advance full



@ti.kernel
def advect_w_notrans(
    w_x0: ti.template(), w_y0: ti.template(), w_z0: ti.template(),
    w_x1: ti.template(), w_y1: ti.template(), w_z1: ti.template(),
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
    dx: ti.f32,
):
    # x velocity
    for I in ti.grouped(w_x1):
        w_at_psi = interp_w_MAC(w_x0, w_y0, w_z0, psi_x[I], dx)
        w_x1[I] = (T_x[I] @ w_at_psi)[0]
    # y velocity
    for I in ti.grouped(w_y1):
        w_at_psi = interp_w_MAC(w_x0, w_y0, w_z0, psi_y[I], dx)
        w_y1[I] = (T_y[I] @ w_at_psi)[1]
    # z velocity
    for I in ti.grouped(w_z1):
        w_at_psi = interp_w_MAC(w_x0, w_y0, w_z0, psi_z[I], dx)
        w_z1[I] = (T_z[I] @ w_at_psi)[2]


@ti.func
def determinant(mat):
    return mat.determinant()

@ti.func
def trace(mat):
    return mat.trace()

@ti.func
def inverse_transpose(mat):
    return mat.inverse().transpose()

@ti.func
def proj_FT(F):
    max_iter = 0
    if projFT:
        max_iter = 10  # Maximum number of iterations for convergence
    tol = 1e-6  # Tolerance for convergence

    lamda = 0.0
    for _ in range(max_iter):
        F_invT = inverse_transpose(F)
        F_plus_lambda_F_invT = F + lamda * F_invT
        det_F_plus_lambda_F_invT = determinant(F_plus_lambda_F_invT)
        if abs(det_F_plus_lambda_F_invT - 1.0) < tol:
            break

        trace_F_invT = trace((F.transpose() @ F).inverse())
        lamda -= (det_F_plus_lambda_F_invT - 1.0) / (trace_F_invT * det_F_plus_lambda_F_invT)
        F += lamda * F_invT
    return F

@ti.func
def normalize_det(F):
    det_F = determinant(F)
    # print(det_F)
    det_F_cub = det_F ** (1/3)
    F = F / det_F_cub

@ti.kernel
def get_particles_id_in_every_cell(cell_particles_id: ti.template(), cell_particle_num: ti.template(),
                                   particles_pos: ti.template(), distribute_idx: int):
    cell_particles_id.fill(-1)
    cell_particle_num.fill(0)
    for i in ti.ndrange(distribute_idx):
        if particles_active[i] == 1:
            cell_id = int(particles_pos[i] / dx)
            particles_index_in_cell = ti.atomic_add(cell_particle_num[cell_id], 1)
            if particles_index_in_cell < cell_max_particle_num:
                cell_particles_id[cell_id[0], cell_id[1], cell_id[2], particles_index_in_cell] = i

@ti.kernel
def compute_dT_dx(grad_T_init_x: ti.template(), grad_T_init_y: ti.template(), grad_T_init_z: ti.template(),
                          T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
                          particles_init_pos: ti.template(), cell_particles_id: ti.template(),
                          cell_particle_num: ti.template()):

    for i in grad_T_init_x:
        if particles_active[i] == 1:
            grad_T_init_x[i] = ti.Matrix.zero(float, 3, 3)
            grad_T_init_y[i] = ti.Matrix.zero(float, 3, 3)
            grad_T_init_z[i] = ti.Matrix.zero(float, 3, 3)

            base_cell_id = int(particles_init_pos[i] / dx)

            for offset in ti.static(ti.grouped(ti.ndrange(*((-1, 3),) * dim))):
                    neighbor_cell_id = base_cell_id + offset
                    for k in ti.static(ti.ndrange(0, cell_particle_num[neighbor_cell_id])):
                        neighbor_particle_id = \
                            cell_particles_id[neighbor_cell_id[0], neighbor_cell_id[1], neighbor_cell_id[2], k]
                        neighbor_particle_pos = particles_init_pos[neighbor_particle_id]
                        if particles_active[neighbor_particle_id] == 1:
                            dist_x = neighbor_particle_pos[0] - particles_init_pos[i][0]
                            dist_y = neighbor_particle_pos[1] - particles_init_pos[i][1]
                            dist_z = neighbor_particle_pos[2] - particles_init_pos[i][2]

                            dw_x = 1. / dx * dN_2(dist_x) * N_2(dist_y) * N_2(dist_z)
                            dw_y = 1. / dx * N_2(dist_x) * dN_2(dist_y) * N_2(dist_z)
                            dw_z = 1. / dx * N_2(dist_x) * N_2(dist_y) * dN_2(dist_z)
                            dw = ti.Vector([dw_x, dw_y, dw_z])

                            T = ti.Matrix.cols(
                                [T_x[neighbor_particle_id], T_y[neighbor_particle_id], T_z[neighbor_particle_id]])
                            grad_T_init_x[i] += dw.outer_product(T[0, :])
                            grad_T_init_y[i] += dw.outer_product(T[1, :])
                            grad_T_init_z[i] += dw.outer_product(T[2, :])

@ti.kernel
def update_particles_w(particles_w: ti.template(), particles_init_w: ti.template(),
                         T_x: ti.template(), T_y: ti.template(), T_z: ti.template()):
    for i in particles_w:
        if particles_active[i] == 1:
            T = ti.Matrix.cols([T_x[i], T_y[i], T_z[i]])
            particles_w[i] = T @ particles_init_w[i]


@ti.kernel
def interp_to_init_particles(particles_pos:ti.template(), particle_init_w:ti.template(), visc_field_x:ti.template(),
                             visc_field_y:ti.template(),visc_field_z:ti.template(),T_x_grad_m: ti.template(), 
                             T_y_grad_m: ti.template(), T_z_grad_m: ti.template(), T_x_init: ti.template(), 
                             T_y_init: ti.template(), T_z_init: ti.template(), dt:float, dx : float):
    for I in particles_pos:
        if particles_active[I] >= 1:
            vis_force_x = interp_2(visc_field_x, particles_pos[I], inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
            vis_force_y = interp_2(visc_field_y, particles_pos[I], inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
            vis_force_z = interp_2(visc_field_z, particles_pos[I], inv_dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
            T_grad_m = ti.Matrix.rows([T_x_grad_m[I], T_y_grad_m[I], T_z_grad_m[I]])
            T_init = ti.Matrix.rows([T_x_init[I], T_y_init[I], T_z_init[I]])
            T = T_init @ T_grad_m

            visc_force = T @ ti.Vector([vis_force_x, vis_force_y, vis_force_z])
            
            particle_init_w[I] += (visc_force) * dt

@ti.kernel
def interp_to_grid(interped_grid_x:ti.template(), interped_grid_y:ti.template(), interped_grid_z:ti.template(), 
                   visc_field_x:ti.template(), visc_field_y:ti.template(), visc_field_z:ti.template(), 
                   dt:float, dx : float):
    for I in ti.grouped(interped_grid_x):
        vis_force = interp_2(visc_field_x, X_x_e[I], inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
        interped_grid_x[I] += (vis_force) * dt

    for I in ti.grouped(interped_grid_y):
        vis_force = interp_2(visc_field_y, X_y_e[I], inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
        interped_grid_y[I] += (vis_force) * dt

    for I in ti.grouped(interped_grid_z):
        vis_force = interp_2(visc_field_z, X_z_e[I], inv_dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
        interped_grid_z[I] += (vis_force) * dt


@ti.kernel
def apply_bc_vortex_0(w_x: ti.template(), w_y: ti.template(),w_z: ti.template(),point_boundary_x: ti.template(),point_boundary_y: ti.template(),point_boundary_z: ti.template()):
    for i,j,k in w_x:
        if point_boundary_x[i, j, k] >= 1:
            w_x[i, j, k] = 0.0
    
    for i,j,k in w_y:
        if point_boundary_y[i, j, k] >= 1:
            w_y[i, j, k] = 0.0

    for i,j,k in w_z:
        if point_boundary_z[i, j, k] >= 1:
            w_z[i, j, k] = 0.0

@ti.kernel
def update_T(T_x_init: ti.template(), T_y_init: ti.template(), T_z_init: ti.template(), T_x_grad_m: ti.template(),
             T_y_grad_m: ti.template(), T_z_grad_m: ti.template()):
    for i in T_x_init:
        T_grad_m = ti.Matrix.cols([T_x_grad_m[i], T_y_grad_m[i], T_z_grad_m[i]])
        T_init = ti.Matrix.cols([T_x_init[i], T_y_init[i], T_z_init[i]])
        T = T_grad_m @ T_init
        T_x_init[i] = T[:, 0]
        T_y_init[i] = T[:, 1]
        T_z_init[i] = T[:, 2]


@ti.kernel
def update_F(F_x_init: ti.template(), F_y_init: ti.template(), F_z_init: ti.template(), F_x_grad_m: ti.template(),
             F_y_grad_m: ti.template(), F_z_grad_m: ti.template()):
    for i in F_x_init:
        F_grad_m = ti.Matrix.cols([F_x_grad_m[i], F_y_grad_m[i], F_z_grad_m[i]])
        F_init = ti.Matrix.cols([F_x_init[i], F_y_init[i], F_z_init[i]])
        F = F_init @ F_grad_m
        F_x_init[i] = F[:, 0]
        F_y_init[i] = F[:, 1]
        F_z_init[i] = F[:, 2]

@ti.func
def transpose_01(m0, m1, m2):
    m0t = ti.Matrix.rows([m0[0, :], m1[0, :], m2[0, :]])
    m1t = ti.Matrix.rows([m0[1, :], m1[1, :], m2[1, :]])
    m2t = ti.Matrix.rows([m0[2, :], m1[2, :], m2[2, :]])
    return m0t, m1t, m2t

@ti.func
def transpose_02(m0, m1, m2):
    m0t = ti.Matrix.cols([m0[:, 0], m1[:, 0], m2[:, 0]])
    m1t = ti.Matrix.cols([m0[:, 1], m1[:, 1], m2[:, 1]])
    m2t = ti.Matrix.cols([m0[:, 2], m1[:, 2], m2[:, 2]])
    return m0t, m1t, m2t


@ti.kernel
def RK4_T_F_forward_hessian(psi: ti.template(), T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
                    F_x: ti.template(), F_y: ti.template(), F_z: ti.template(),
                    F_x_init:ti.template(), F_y_init:ti.template(), F_z_init:ti.template(),
                  u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), dt: float, distribute_idx:int):
    for i in ti.ndrange(distribute_idx):
        if particles_active[i] == 1:
            F_init = ti.Matrix.rows([F_x_init[i], F_y_init[i], F_z_init[i]])
            # first
            u1, grad_u_at_psi, hessian_ux, hessian_uy, hessian_uz = interp_u_MAC_grad_grad_transpose(u_x0, u_y0, u_z0, psi[i], dx)
            grad_u_at_psi_notrans = grad_u_at_psi.transpose()
            dT_x_dt1 = grad_u_at_psi @ T_x[i]  # time derivative of T
            dT_y_dt1 = grad_u_at_psi @ T_y[i]  # time derivative of T
            dT_z_dt1 = grad_u_at_psi @ T_z[i]  # time derivative of T

            F = ti.Matrix.cols([F_x[i], F_y[i], F_z[i]])
            dF_dt1 = F @ grad_u_at_psi

            gradF_j0 = gradF_j0_particles[i]
            gradF_j1 = gradF_j1_particles[i]
            gradF_j2 = gradF_j2_particles[i]

            dgradF_j0_dt1_1 = -gradF_j0 @ grad_u_at_psi_notrans
            dgradF_j1_dt1_1 = -gradF_j1 @ grad_u_at_psi_notrans
            dgradF_j2_dt1_1 = -gradF_j2 @ grad_u_at_psi_notrans

            dgradF_j0_dt1_2 = grad_u_at_psi_notrans @ gradF_j0
            dgradF_j1_dt1_2 = grad_u_at_psi_notrans @ gradF_j1
            dgradF_j2_dt1_2 = grad_u_at_psi_notrans @ gradF_j2

            dgradF_j0_dt1_3_0ij = (hessian_ux @ (F.transpose())).transpose()
            dgradF_j1_dt1_3_1ij = (hessian_uy @ (F.transpose())).transpose()
            dgradF_j2_dt1_3_2ij = (hessian_uz @ (F.transpose())).transpose()  
            dgradF_j0_dt1_3, dgradF_j1_dt1_3, dgradF_j2_dt1_3 = transpose_01(dgradF_j0_dt1_3_0ij, dgradF_j1_dt1_3_1ij, dgradF_j2_dt1_3_2ij)
            dgradF_j0_dt1 = dgradF_j0_dt1_1 + dgradF_j0_dt1_2 + dgradF_j0_dt1_3
            dgradF_j1_dt1 = dgradF_j1_dt1_1 + dgradF_j1_dt1_2 + dgradF_j1_dt1_3
            dgradF_j2_dt1 = dgradF_j2_dt1_1 + dgradF_j2_dt1_2 + dgradF_j2_dt1_3
            
            # prepare second
            psi_x1 = psi[i] + 0.5 * dt * u1  # advance 0.5 steps
            T_x1 = T_x[i] - 0.5 * dt * dT_x_dt1
            T_y1 = T_y[i] - 0.5 * dt * dT_y_dt1
            T_z1 = T_z[i] - 0.5 * dt * dT_z_dt1
            F_1 = F + 0.5 * dt * dF_dt1
            gradF_j0_1 = gradF_j0 + 0.5 * dt * dgradF_j0_dt1
            gradF_j1_1 = gradF_j1 + 0.5 * dt * dgradF_j1_dt1
            gradF_j2_1 = gradF_j2 + 0.5 * dt * dgradF_j2_dt1

            # second
            u2, grad_u_at_psi, hessian_ux, hessian_uy, hessian_uz = interp_u_MAC_grad_grad_transpose(u_x0, u_y0, u_z0, psi_x1, dx)
            grad_u_at_psi_notrans = grad_u_at_psi.transpose()
            dT_x_dt2 = grad_u_at_psi @ T_x1  # time derivative of T
            dT_y_dt2 = grad_u_at_psi @ T_y1  # time derivative of T
            dT_z_dt2 = grad_u_at_psi @ T_z1  # time derivative of T
            dF_dt2 = F_1 @ grad_u_at_psi

            dgradF_j0_dt2_1 = -gradF_j0_1 @ grad_u_at_psi_notrans
            dgradF_j1_dt2_1 = -gradF_j1_1 @ grad_u_at_psi_notrans
            dgradF_j2_dt2_1 = -gradF_j2_1 @ grad_u_at_psi_notrans

            dgradF_j0_dt2_2 = grad_u_at_psi_notrans @ gradF_j0_1
            dgradF_j1_dt2_2 = grad_u_at_psi_notrans @ gradF_j1_1
            dgradF_j2_dt2_2 = grad_u_at_psi_notrans @ gradF_j2_1

            dgradF_j0_dt2_3_0ij = (hessian_ux @ (F_1.transpose())).transpose()
            dgradF_j1_dt2_3_1ij = (hessian_uy @ (F_1.transpose())).transpose()
            dgradF_j2_dt2_3_2ij = (hessian_uz @ (F_1.transpose())).transpose()

            dgradF_j0_dt2_3, dgradF_j1_dt2_3, dgradF_j2_dt2_3 = transpose_01(dgradF_j0_dt2_3_0ij, dgradF_j1_dt2_3_1ij, dgradF_j2_dt2_3_2ij)


            dgradF_j0_dt2 = dgradF_j0_dt2_1 + dgradF_j0_dt2_2 + dgradF_j0_dt2_3
            dgradF_j1_dt2 = dgradF_j1_dt2_1 + dgradF_j1_dt2_2 + dgradF_j1_dt2_3
            dgradF_j2_dt2 = dgradF_j2_dt2_1 + dgradF_j2_dt2_2 + dgradF_j2_dt2_3

            # prepare third
            psi_x2 = psi[i] + 0.5 * dt * u2  # advance 0.5 again
            T_x2 = T_x[i] - 0.5 * dt * dT_x_dt2
            T_y2 = T_y[i] - 0.5 * dt * dT_y_dt2
            T_z2 = T_z[i] - 0.5 * dt * dT_z_dt2
            F_2 = F + 0.5 * dt * dF_dt2
            gradF_j0_2 = gradF_j0 + 0.5 * dt * dgradF_j0_dt2
            gradF_j1_2 = gradF_j1 + 0.5 * dt * dgradF_j1_dt2
            gradF_j2_2 = gradF_j2 + 0.5 * dt * dgradF_j2_dt2

            # third
            u3, grad_u_at_psi, hessian_ux, hessian_uy, hessian_uz = interp_u_MAC_grad_grad_transpose(u_x0, u_y0, u_z0, psi_x2, dx)
            grad_u_at_psi_notrans = grad_u_at_psi.transpose()
            dT_x_dt3 = grad_u_at_psi @ T_x2 
            dT_y_dt3 = grad_u_at_psi @ T_y2 
            dT_z_dt3 = grad_u_at_psi @ T_z2 
            dF_dt3 = F_2 @ grad_u_at_psi

            dgradF_j0_dt3_1 = -gradF_j0_2 @ grad_u_at_psi_notrans
            dgradF_j1_dt3_1 = -gradF_j1_2 @ grad_u_at_psi_notrans
            dgradF_j2_dt3_1 = -gradF_j2_2 @ grad_u_at_psi_notrans

            dgradF_j0_dt3_2 = grad_u_at_psi_notrans @ gradF_j0_2
            dgradF_j1_dt3_2 = grad_u_at_psi_notrans @ gradF_j1_2
            dgradF_j2_dt3_2 = grad_u_at_psi_notrans @ gradF_j2_2

            dgradF_j0_dt3_3_0ij = (hessian_ux @ (F_2.transpose())).transpose()
            dgradF_j1_dt3_3_1ij = (hessian_uy @ (F_2.transpose())).transpose()
            dgradF_j2_dt3_3_2ij = (hessian_uz @ (F_2.transpose())).transpose()

            dgradF_j0_dt3_3, dgradF_j1_dt3_3, dgradF_j2_dt3_3 = transpose_01(dgradF_j0_dt3_3_0ij, dgradF_j1_dt3_3_1ij, dgradF_j2_dt3_3_2ij)

            dgradF_j0_dt3 = dgradF_j0_dt3_1 + dgradF_j0_dt3_2 + dgradF_j0_dt3_3
            dgradF_j1_dt3 = dgradF_j1_dt3_1 + dgradF_j1_dt3_2 + dgradF_j1_dt3_3
            dgradF_j2_dt3 = dgradF_j2_dt3_1 + dgradF_j2_dt3_2 + dgradF_j2_dt3_3

            # prepare fourth
            psi_x3 = psi[i] + 1.0 * dt * u3
            T_x3 = T_x[i] - 1.0 * dt * dT_x_dt3  # advance 1.0
            T_y3 = T_y[i] - 1.0 * dt * dT_y_dt3  # advance 1.0
            T_z3 = T_z[i] - 1.0 * dt * dT_z_dt3  # advance 1.0
            F_3 = F + 1.0 * dt * dF_dt3
            gradF_j0_3 = gradF_j0 + 1.0 * dt * dgradF_j0_dt3
            gradF_j1_3 = gradF_j1 + 1.0 * dt * dgradF_j1_dt3
            gradF_j2_3 = gradF_j2 + 1.0 * dt * dgradF_j2_dt3

            # fourth
            u4, grad_u_at_psi, hessian_ux, hessian_uy, hessian_uz = interp_u_MAC_grad_grad_transpose(u_x0, u_y0, u_z0, psi_x3, dx)
            grad_u_at_psi_notrans = grad_u_at_psi.transpose()
            dT_x_dt4 = grad_u_at_psi @ T_x3  # time derivative of T
            dT_y_dt4 = grad_u_at_psi @ T_y3  # time derivative of T
            dT_z_dt4 = grad_u_at_psi @ T_z3  # time derivative of T
            dF_dt4 = F_3 @ grad_u_at_psi
            
            dgradF_j0_dt4_1 = -gradF_j0_3 @ grad_u_at_psi_notrans
            dgradF_j1_dt4_1 = -gradF_j1_3 @ grad_u_at_psi_notrans
            dgradF_j2_dt4_1 = -gradF_j2_3 @ grad_u_at_psi_notrans

            dgradF_j0_dt4_2 = grad_u_at_psi_notrans @ gradF_j0_3
            dgradF_j1_dt4_2 = grad_u_at_psi_notrans @ gradF_j1_3
            dgradF_j2_dt4_2 = grad_u_at_psi_notrans @ gradF_j2_3

            dgradF_j0_dt4_3_0ij = (hessian_ux @ (F_3.transpose())).transpose()
            dgradF_j1_dt4_3_1ij = (hessian_uy @ (F_3.transpose())).transpose()
            dgradF_j2_dt4_3_2ij = (hessian_uz @ (F_3.transpose())).transpose()

            dgradF_j0_dt4_3, dgradF_j1_dt4_3, dgradF_j2_dt4_3 = transpose_01(dgradF_j0_dt4_3_0ij, dgradF_j1_dt4_3_1ij, dgradF_j2_dt4_3_2ij)

            dgradF_j0_dt4 = dgradF_j0_dt4_1 + dgradF_j0_dt4_2 + dgradF_j0_dt4_3
            dgradF_j1_dt4 = dgradF_j1_dt4_1 + dgradF_j1_dt4_2 + dgradF_j1_dt4_3
            dgradF_j2_dt4 = dgradF_j2_dt4_1 + dgradF_j2_dt4_2 + dgradF_j2_dt4_3

            # final advance
            psi[i] = psi[i] + dt * 1. / 6 * (u1 + 2 * u2 + 2 * u3 + u4)
            T_x[i] = T_x[i] - dt * 1. / 6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4)  # advance full
            T_y[i] = T_y[i] - dt * 1. / 6 * (dT_y_dt1 + 2 * dT_y_dt2 + 2 * dT_y_dt3 + dT_y_dt4)  # advance full
            T_z[i] = T_z[i] - dt * 1. / 6 * (dT_z_dt1 + 2 * dT_z_dt2 + 2 * dT_z_dt3 + dT_z_dt4)  # advance full
            
            gradF_j0_particles[i] = gradF_j0_particles[i] + \
                dt * 1. / 6 * (dgradF_j0_dt1 + 2 * dgradF_j0_dt2 + 2 * dgradF_j0_dt3 + dgradF_j0_dt4)
            gradF_j1_particles[i] = gradF_j1_particles[i] + \
                dt * 1. / 6 * (dgradF_j1_dt1 + 2 * dgradF_j1_dt2 + 2 * dgradF_j1_dt3 + dgradF_j1_dt4)
            gradF_j2_particles[i] = gradF_j2_particles[i] + \
                dt * 1. / 6 * (dgradF_j2_dt1 + 2 * dgradF_j2_dt2 + 2 * dgradF_j2_dt3 + dgradF_j2_dt4)

            T = ti.Matrix.rows([T_x[i], T_y[i], T_z[i]])
            F = F + dt * 1. / 6 * (dF_dt1 + 2 * dF_dt2 + 2 * dF_dt3 + dF_dt4)

            F = proj_FT(F)
            T = proj_FT(T)
            if normalize_detFT:
                normalize_det(T)
                normalize_det(F)
            F_x[i] = F[:, 0]
            F_y[i] = F[:, 1]
            F_z[i] = F[:, 2]
            T_x[i] = T[:, 0]
            T_y[i] = T[:, 1]
            T_z[i] = T[:, 2]

def stretch_T_F_and_advect_particles_hessian(particles_pos, T_x, T_y, T_z, 
                                             F_x, F_y, F_z, 
                                             F_x_init, F_y_init, F_z_init, 
                                             u_x, u_y, u_z, 
                                             dt, distribute_idx):
    # reset_to_identity_T(gradF_j0_particles, gradF_j1_particles, gradF_j2_particles)
    RK4_T_F_forward_hessian(particles_pos, T_x, T_y, T_z, 
                            F_x, F_y, F_z,
                            F_x_init, F_y_init, F_z_init, 
                            u_x, u_y, u_z, dt, distribute_idx)


@ti.kernel
def P2G_w(particles_init_w: ti.template(), particles_pos: ti.template(),
        w_x: ti.template(), w_y: ti.template(), w_z: ti.template(), 
        p2g_weight_x: ti.template(), p2g_weight_y: ti.template(), p2g_weight_z: ti.template(), 
        T_x_grad_m: ti.template(), T_y_grad_m: ti.template(), T_z_grad_m: ti.template(),
        F_x_grad_m: ti.template(), F_y_grad_m: ti.template(), F_z_grad_m: ti.template(),
        F_x_init: ti.template(), F_y_init: ti.template(), F_z_init: ti.template(), distribute_idx: int, 
        edge_x_boundary_mask:ti.template(), edge_y_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template()
        ):
    w_x.fill(0.0)
    w_y.fill(0.0)
    w_z.fill(0.0)
    p2g_weight_x.fill(0.0)
    p2g_weight_y.fill(0.0)
    p2g_weight_z.fill(0.0)

    for i in ti.ndrange(distribute_idx):
        if particles_active[i] == 1:
            weight = 0.
            T_grad_m = ti.Matrix.rows([T_x_grad_m[i], T_y_grad_m[i], T_z_grad_m[i]])
            F_grad_m = ti.Matrix.rows([F_x_grad_m[i], F_y_grad_m[i], F_z_grad_m[i]])
            init_C = ti.Matrix.rows([init_C_x[i], init_C_y[i], init_C_z[i]])
            F_init_C_T = F_grad_m @ init_C @ T_grad_m

            F_init = ti.Matrix.rows([F_x_init[i], F_y_init[i], F_z_init[i]])
            F = F_grad_m @ F_init

            particles_w = F @ particles_init_w[i]

            gradF_j0 = gradF_j0_particles[i]
            gradF_j1 = gradF_j1_particles[i]
            gradF_j2 = gradF_j2_particles[i]

            delta_1 = gradF_j0 * particles_mid_w[i][0] + gradF_j1 * particles_mid_w[i][1] + gradF_j2 * particles_mid_w[i][2]


            pos = particles_pos[i] / dx
            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 0))
            udim, vdim, wdim = w_x.shape
            for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
                face_id = base_face_id + offset
                if 0 <= face_id[0] <= udim - 1 and 0 < face_id[1] < vdim - 1 and 0 < face_id[2] < wdim- 1:
                    if edge_x_boundary_mask[face_id] <= 0:# and probe_wx[face_id] >= 1e-3:
                        weight = N_2(pos[0] - face_id[0] - 0.5) * N_2(pos[1] - face_id[1]) * N_2(pos[2] - face_id[2])
                        dpos = ti.Vector([face_id[0]  + 0.5 - pos[0], face_id[1] - pos[1], face_id[2] - pos[2]]) * dx
                        p2g_weight_x[face_id] += weight
                        delta = F_init_C_T[0, :].dot(dpos)
                        if use_hessian:
                            delta = (F_init_C_T + delta_1)[0, :].dot(dpos)
                        if use_APIC:
                            w_x[face_id] += (particles_w[0] + delta) * weight
                        else:
                            w_x[face_id] += (particles_w[0]) * weight

            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 1))
            udim, vdim, wdim = w_y.shape
            for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
                face_id = base_face_id + offset
                if 0 < face_id[0] < udim- 1 and 0 <= face_id[1] <= vdim- 1 and 0 < face_id[2] < wdim- 1:
                    if edge_y_boundary_mask[face_id] <= 0:# and probe_wy[face_id] >= 1e-3:
                        weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1] - 0.5) * N_2(pos[2] - face_id[2])
                        dpos = ti.Vector([face_id[0]- pos[0], face_id[1]  + 0.5 - pos[1], face_id[2] - pos[2]]) * dx
                        p2g_weight_y[face_id] += weight
                        delta = F_init_C_T[1, :].dot(dpos)
                        if use_hessian:
                            delta = (F_init_C_T + delta_1)[1, :].dot(dpos)
                        if use_APIC:
                            w_y[face_id] += (particles_w[1] + delta) * weight
                        else:
                            w_y[face_id] += (particles_w[1]) * weight

            base_face_id = int(pos - 0.5 * ti.Vector.unit(dim, 2))
            udim, vdim, wdim = w_z.shape
            for offset in ti.grouped(ti.ndrange(*((-1, 3),) * dim)):
                face_id = base_face_id + offset
                if 0 < face_id[0] < udim- 1 and 0 < face_id[1] < vdim- 1 and 0 <= face_id[2] <= wdim- 1:
                    if edge_z_boundary_mask[face_id] <= 0:# and probe_wz[face_id] >= 1e-3:
                        weight = N_2(pos[0] - face_id[0]) * N_2(pos[1] - face_id[1]) * N_2(pos[2] - face_id[2] - 0.5)
                        dpos = ti.Vector([face_id[0] - pos[0], face_id[1] - pos[1], face_id[2] + 0.5 - pos[2]]) * dx
                        p2g_weight_z[face_id] += weight
                        delta = F_init_C_T[2, :].dot(dpos)
                        if use_hessian:
                            delta = (F_init_C_T + delta_1)[2, :].dot(dpos)
                        if use_APIC:
                            w_z[face_id] += (particles_w[2] + delta) * weight
                        else:
                            w_z[face_id] += (particles_w[2]) * weight


    for I in ti.grouped(p2g_weight_x):
        if p2g_weight_x[I] > 1e-5:
            w_x[I] /= p2g_weight_x[I]

    for I in ti.grouped(p2g_weight_y):
        if p2g_weight_y[I] > 1e-5:
            w_y[I] /= p2g_weight_y[I]

    for I in ti.grouped(p2g_weight_z):
        if p2g_weight_z[I] > 1e-5:
            w_z[I] /= p2g_weight_z[I]


@ti.kernel
def mask_particles():
    particles_active.fill(1)
    for i in particles_active:
        if particles_pos[i][0] <= 0  or particles_pos[i][0] >=res_x * dx  or particles_pos[i][1] <= 0  or particles_pos[i][1] >=res_y * dx or particles_pos[i][2] <= 0  or particles_pos[i][2] >=res_z * dx:
            particles_active[i] = 0
            continue
        grid_idx = ti.floor((particles_pos[i]) / dx, int)
        if (grid_idx[0] >= 0 and grid_idx[0] < res_x - 0 and grid_idx[1] >= 0 and grid_idx[1] < res_y - 0 and grid_idx[2] >= 0 and grid_idx[2] < res_z - 0) and center_boundary_mask[grid_idx] >= 1:
            particles_active[i] = 0

def main(from_frame=0):
    mesh_prev_v, plesiosaur_faces = igl.read_triangle_mesh(f'./{motion_seq_name}/{mesh_name}{0}.obj')
    mesh_next_v, _ = igl.read_triangle_mesh(f'./{motion_seq_name}/{mesh_name}{1}.obj')
    vn, fn = mesh_prev_v.shape[0], plesiosaur_faces.shape[0]
    # init mesh
    total_length = np.max(np.max(mesh_prev_v, axis=0) - np.min(mesh_prev_v, axis=0))
    mesh_prev_v *= (0.9 / total_length)
    mesh_prev_v[:, 2] += 0.5
    mesh_prev_v[:, 1] += 0.29
    mesh_prev_v[:, 0] += 0.45
    total_length = np.max(np.max(mesh_next_v, axis=0) - np.min(mesh_next_v, axis=0))
    mesh_next_v *= (0.9 / total_length)
    mesh_next_v[:, 2] += 0.5
    mesh_next_v[:, 1] += 0.29
    mesh_next_v[:, 0] += 0.45

    prev_v = mesh_prev_v

    ti_faces_0.from_numpy(plesiosaur_faces)
    ti_vertices.from_numpy(mesh_prev_v)
    bvh = LBVH(ti_vertices, ti_faces_0, vn, fn)

    from_frame = max(0, from_frame)

    load_name = ""
    logsdir = os.path.join("logs", exp_name)
    logs_loaddir = os.path.join("logs", load_name)
    os.makedirs(logsdir, exist_ok=True)
    if from_frame <= 0:
        remove_everything_in(logsdir)

    vtkdir = "vtks"
    vtkdir = os.path.join(logsdir, vtkdir)
    os.makedirs(vtkdir, exist_ok=True)

    u_x.fill(0.)
    u_y.fill(0.)
    u_z.fill(0.)
    w_x.fill(0.0)
    w_y.fill(0.0)
    w_z.fill(0.0)
    sub_t = 0. # the time since last reinit
    frame_idx = from_frame
    last_output_substep = 0
    num_reinits = 0 # number of reinitializations already performed
    i = -1
    ik = 0

    frame_times = np.zeros(total_steps)
    # amgpcg_torch = PoissonSolver(tile_dim_vec, level_num=3, bottom_smoothing=40)
    amgpcg_torch = AMGPCGTorch(
        tile_dim, bottom_smoothing, verbose, False, False
    )
    total_time = 0.0
    while True:
        i += 1
        j = i % reinit_every
        i_next = i + 1
        j_next = i_next % reinit_every
        print("[Simulate] Running step: ", i, " / substep: ", j)

        calc_max_speed(u_x, u_y, u_z) # saved to max_speed[None]
        if i== 0:
            curr_dt = visualize_dt/15
        else:
            curr_dt = CFL * dx / max_speed[None]

        if save_frame_each_step:
            output_frame = True
            frame_idx += 1
        else:
            if sub_t+curr_dt >= visualize_dt: # if over
                curr_dt = visualize_dt-sub_t
                sub_t = 0. # empty sub_t
                v = mesh_next_v
                mesh_vel_np = (v-prev_v)/curr_dt
                prev_v = v
                if i <= total_steps - 1:
                    print(f'step execution time: {frame_times[i]:.6f} seconds')
                frame_idx += 1
                output_frame = True
                
                mesh_prev_v = mesh_next_v
                mesh_next_v, _ = igl.read_triangle_mesh(f'./{motion_seq_name}/{mesh_name}{frame_idx+1}.obj')
                total_length = np.max(np.max(mesh_next_v, axis=0) - np.min(mesh_next_v, axis=0))
                mesh_next_v *= (0.9 / total_length)
                mesh_next_v[:, 2] += 0.5
                mesh_next_v[:, 1] += 0.29
                mesh_next_v[:, 0] += 0.45

                print(f'Visualized frame {frame_idx}')
            else:
                sub_t += curr_dt
                
                v = (1.0 - sub_t/visualize_dt) * mesh_prev_v + (sub_t/visualize_dt) * mesh_next_v
                mesh_vel_np = (v-prev_v)/curr_dt
                prev_v = v
                print(f'Visualize time {sub_t}/{visualize_dt}')
                output_frame = False
        total_time += curr_dt
        print("[Simulate] Running step: ", i, " / substep: ", j)

        if j == 0:
            print("[Simulate] Reinitializing the flow map for the: ", num_reinits, " time!")
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, inv_dx)
            if reinit_particle_pos or i == 0:
                init_particles_pos_uniform(particles_pos, X, res_x, res_y, particles_per_cell, dx,
                                           particles_per_cell_axis, dist_between_neighbor)
                distribute_idx = particles_pos.shape[0] - num_extra_particles
                ik = i
                il = i
            init_particles_w(particles_init_w, particles_pos, w_x, w_y, w_z, init_C_x,
                               init_C_y, init_C_z, dx, distribute_idx)
            reset_T_to_identity(T_x_init, T_y_init, T_z_init)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m, T_z_grad_m)

            reset_T_to_identity(F_x_init, F_y_init, F_z_init)
            reset_T_to_identity(F_x_grad_m, F_y_grad_m, F_z_grad_m)
            mask_particles()


        k = (i - ik) % reinit_every_grad_m
        if k == 0:
            init_particles_w_grad_w(particles_mid_w, particles_pos, w_x, w_y, w_z,
                                      init_C_x, init_C_y, init_C_z, dx, distribute_idx)
            update_T(T_x_init, T_y_init, T_z_init, T_x_grad_m, T_y_grad_m, T_z_grad_m)
            reset_T_to_identity(T_x_grad_m, T_y_grad_m, T_z_grad_m)

            update_F(F_x_init, F_y_init, F_z_init, F_x_grad_m, F_y_grad_m, F_z_grad_m)
            reset_T_to_identity(F_x_grad_m, F_y_grad_m, F_z_grad_m)

            reset_to_zero_gradF(gradF_j0_particles, gradF_j1_particles, gradF_j2_particles)

        ti_new_vertices.from_numpy(v)
        bvh.update_bvh_tree(ti_new_vertices)
        tribox_ff_voxelize(bvh, surf_occupancy, ti_faces_0, ti_new_vertices, res_x, res_y, res_z, dx)
        initialize_occupancy(out_occupancy, surf_occupancy, res_x, res_y, res_z)
        changes[None] = 1
        while changes[None]:
            flood_fill(changes, out_occupancy, surf_occupancy, res_x, res_y, res_z)
        center_boundary_mask.fill(0)
        fill_internal_occupancy(out_occupancy, center_boundary_mask, surf_occupancy)
        fill_inner_boundary(center_boundary_mask, surf_occupancy, in_occupancy)
        
        mesh_vel.from_numpy(mesh_vel_np)
        boundary_vel.fill(0.0)
        boundary_vel_tan.fill(0.0)
        calculate_bv_splitnormal(bvh, boundary_vel, boundary_vel_tan, mesh_vel, ti_new_vertices, ti_faces_0, dx)
        bv_x.fill(0.0)
        bv_y.fill(0.0)
        bv_z.fill(0.0)

        enforce_surface_vel(center_boundary_mask, boundary_vel_tan, bv_x_tan, bv_y_tan, bv_z_tan, 0.0, 0.0, 0.0)
        enforce_surface_vel(center_boundary_mask, boundary_vel, bv_x, bv_y, bv_z, x_wall_v, y_wall_v, z_wall_v)
        edge_from_center_boundary(center_boundary_mask, edge_x_boundary_mask, edge_y_boundary_mask, edge_z_boundary_mask)
        face_from_center_boundary(center_boundary_mask, face_x_boundary_mask, face_y_boundary_mask, face_z_boundary_mask)
        extend_boundary_field(center_boundary_mask, center_boundary_mask_extend)
        extend_boundary_field(edge_x_boundary_mask, edge_x_boundary_mask_extend)
        extend_boundary_field(edge_y_boundary_mask, edge_y_boundary_mask_extend)
        extend_boundary_field(edge_z_boundary_mask, edge_z_boundary_mask_extend)


        surf_face_mask_x.fill(0)
        surf_face_mask_y.fill(0)
        surf_face_mask_z.fill(0)
        nx[None] = 0
        ny[None] = 0
        nz[None] = 0
        face_from_center_boundary0(surf_occupancy, surf_face_mask_x, surf_face_mask_y, surf_face_mask_z)
        assign_mapping_n2ijk(surf_face_mask_x, nx2ijk, nx)
        assign_mapping_n2ijk(surf_face_mask_y, ny2ijk, ny)
        assign_mapping_n2ijk(surf_face_mask_z, nz2ijk, nz)
        nx_value = nx[None]
        ny_value = ny[None]
        nz_value = nz[None]
        stride = 4
        fx_points = np.zeros((nx_value, stride * stride, 3), dtype=np.float64)
        fy_points = np.zeros((ny_value, stride * stride, 3), dtype=np.float64)
        fz_points = np.zeros((nz_value, stride * stride, 3), dtype=np.float64)
        sample_fx_points(nx2ijk, fx_points, nx, stride, dx)
        sample_fy_points(ny2ijk, fy_points, ny, stride, dx)
        sample_fz_points(nz2ijk, fz_points, nz, stride, dx)
        fx_points_flat = fx_points.reshape(-1, 3)
        fy_points_flat = fy_points.reshape(-1, 3)
        fz_points_flat = fz_points.reshape(-1, 3)

        start = time.time()
        winding_numbers_x = igl.fast_winding_number_for_meshes(v, plesiosaur_faces, fx_points_flat)
        winding_numbers_y = igl.fast_winding_number_for_meshes(v, plesiosaur_faces, fy_points_flat)
        winding_numbers_z = igl.fast_winding_number_for_meshes(v, plesiosaur_faces, fz_points_flat)
        end = time.time()

        fx_fluid = winding_numbers_x < 0.5
        fy_fluid = winding_numbers_y < 0.5
        fz_fluid = winding_numbers_z < 0.5

        fx_fluid = fx_fluid.reshape(nx_value, stride * stride)
        fy_fluid = fy_fluid.reshape(ny_value, stride * stride)
        fz_fluid = fz_fluid.reshape(nz_value, stride * stride)

        fx_fraction = np.mean(fx_fluid, axis=1)
        fy_fraction = np.mean(fy_fluid, axis=1)
        fz_fraction = np.mean(fz_fluid, axis=1)
        fx_fraction = np.where(fx_fraction < 0.1, 0, fx_fraction)
        fy_fraction = np.where(fy_fraction < 0.1, 0, fy_fraction)
        fz_fraction = np.where(fz_fraction < 0.1, 0, fz_fraction)


        surf_face_fraction_x.fill(0.0)
        surf_face_fraction_y.fill(0.0)
        surf_face_fraction_z.fill(0.0)
        assign_fractions(nx2ijk, fx_fraction, surf_face_fraction_x, nx)
        assign_fractions(ny2ijk, fy_fraction, surf_face_fraction_y, ny)
        assign_fractions(nz2ijk, fz_fraction, surf_face_fraction_z, nz)
        fill_fractions(surf_face_fraction_x, surf_face_fraction_y, surf_face_fraction_z, 
                    surf_occupancy, center_boundary_mask)

        extend_field(surf_face_fraction_x, surf_face_fraction_x_ext)
        extend_field(surf_face_fraction_y, surf_face_fraction_y_ext)
        extend_field(surf_face_fraction_z, surf_face_fraction_z_ext)

        test_inbound(in_occupancy, surf_face_fraction_x, surf_face_fraction_y, surf_face_fraction_z)
        extend_boundary_field(in_occupancy, in_occupancy_ext)

        if  use_midpoint_vel:
            # start midpoint
            reset_to_identity(psi_x, psi_y, psi_z, F_x_tmp, F_y_tmp, F_z_tmp)

            RK4_grid_graduT_psiF(psi_x, F_x_tmp, u_x, u_y, u_z, 0.5 * curr_dt)
            RK4_grid_graduT_psiF(psi_y, F_y_tmp, u_x, u_y, u_z, 0.5 * curr_dt)
            RK4_grid_graduT_psiF(psi_z, F_z_tmp, u_x, u_y, u_z, 0.5 * curr_dt)

            copy_to(w_x, penalty_w_x)
            copy_to(w_y, penalty_w_y)
            copy_to(w_z, penalty_w_z)
            advect_w_notrans(
                penalty_w_x, penalty_w_y, penalty_w_z,
                w_x, w_y, w_z,
                F_x_tmp, F_y_tmp, F_z_tmp,
                psi_x, psi_y, psi_z,
                dx,
            )
            rtf_solve(stream_x, stream_x_extend, w_x, edge_x_boundary_mask_noin, edge_x_boundary_mask_extend_noin, 0, amgpcg_torch)
            rtf_solve(stream_y, stream_y_extend, w_y, edge_y_boundary_mask_noin, edge_y_boundary_mask_extend_noin, 1, amgpcg_torch)
            rtf_solve(stream_z, stream_z_extend, w_z, edge_z_boundary_mask_noin, edge_z_boundary_mask_extend_noin, 2, amgpcg_torch)
            stream2velocity(u_x, u_y, u_z, stream_x, stream_y, stream_z, dx)
            rtf_solve_harmonic_cutcell(bv_x, bv_y, bv_z,
                   u_x, u_y, u_z, boundary_vel, p_ext,
                   x_wall_v, y_wall_v, z_wall_v, 
                   in_occupancy_ext,
                   surf_occupancy,
                   amgpcg_torch)


        stretch_T_F_and_advect_particles_hessian(particles_pos, T_x_grad_m, T_y_grad_m, T_z_grad_m, 
                                                    F_x_grad_m, F_y_grad_m, F_z_grad_m,
                                                    F_x_init, F_y_init, F_z_init,
                                                    u_x, u_y, u_z, curr_dt, distribute_idx)
        mask_particles()

        P2G_w(particles_init_w, particles_pos, w_x, w_y, w_z, p2g_weight_x,
            p2g_weight_y, p2g_weight_z, T_x_grad_m, T_y_grad_m, T_z_grad_m,
            F_x_grad_m, F_y_grad_m, F_z_grad_m, F_x_init, F_y_init, F_z_init, distribute_idx,
            edge_x_boundary_mask, edge_y_boundary_mask, edge_z_boundary_mask)
        

        rtf_solve(stream_x, stream_x_extend, w_x, edge_x_boundary_mask_noin, edge_x_boundary_mask_extend_noin, 0, amgpcg_torch)
        rtf_solve(stream_y, stream_y_extend, w_y, edge_y_boundary_mask_noin, edge_y_boundary_mask_extend_noin, 1, amgpcg_torch)
        rtf_solve(stream_z, stream_z_extend, w_z, edge_z_boundary_mask_noin, edge_z_boundary_mask_extend_noin, 2, amgpcg_torch)
        stream2velocity(u_x, u_y, u_z, stream_x, stream_y, stream_z, dx)
        rtf_solve_harmonic_cutcell(bv_x, bv_y, bv_z,
                   u_x, u_y, u_z, boundary_vel, p_ext,
                   x_wall_v, y_wall_v, z_wall_v, 
                   in_occupancy_ext,
                   surf_occupancy,
                   amgpcg_torch)
        apply_bc_w(u_x, u_y, u_z, w_x, w_y, w_z, 
                    stream_x, stream_y, stream_z, 
                    center_boundary_mask, boundary_vel, inv_dx)
        if use_noslip:
            copy_to(u_x, tmp_u_x)
            copy_to(u_y, tmp_u_y)
            copy_to(u_z, tmp_u_z)
            affected_by_solid_penalty_w_x.fill(0)
            affected_by_solid_penalty_w_y.fill(0)
            affected_by_solid_penalty_w_z.fill(0)
            movingsolid_noslip_ux_exam8adjecent(tmp_u_x, u_x,face_x_boundary_mask,center_boundary_mask, 
                                                             boundary_vel_tan, affected_by_solid_penalty_w_y, affected_by_solid_penalty_w_z)
            movingsolid_noslip_uy_exam8adjecent(tmp_u_y, u_y,face_y_boundary_mask,center_boundary_mask, 
                                                             boundary_vel_tan, affected_by_solid_penalty_w_x, affected_by_solid_penalty_w_z)
            movingsolid_noslip_uz_exam8adjecent(tmp_u_z, u_z,face_z_boundary_mask, center_boundary_mask,
                                                              boundary_vel_tan, affected_by_solid_penalty_w_x, affected_by_solid_penalty_w_y)

            add_fields(tmp_u_x, u_x, penalty_u_x, -1.0)
            add_fields(tmp_u_y, u_y, penalty_u_y, -1.0)
            add_fields(tmp_u_z, u_z, penalty_u_z, -1.0)

            penalty_w_x.fill(0)
            penalty_w_y.fill(0)
            penalty_w_z.fill(0)
            curl_f2e_x(penalty_u_z, penalty_u_y, penalty_w_x, inv_dx)
            curl_f2e_y(penalty_u_x, penalty_u_z, penalty_w_y, inv_dx)
            curl_f2e_z(penalty_u_y, penalty_u_x, penalty_w_z, inv_dx)
            apply_wb_0(penalty_w_x, penalty_w_y, penalty_w_z)

            mtply_fields(penalty_w_x, affected_by_solid_penalty_w_x, penalty_w_x, 1.0)
            mtply_fields(penalty_w_y, affected_by_solid_penalty_w_y, penalty_w_y, 1.0)
            mtply_fields(penalty_w_z, affected_by_solid_penalty_w_z, penalty_w_z, 1.0)

            interp_to_init_particles(particles_pos, particles_init_w, penalty_w_x, penalty_w_y, 
                                     penalty_w_z, T_x_grad_m, T_y_grad_m, T_z_grad_m, 
                                     T_x_init, T_y_init, T_z_init, alpha, dx)
            
            if interp_by_grid:
                interp_to_grid(w_x, w_y, w_z, penalty_w_x, penalty_w_y, penalty_w_z, alpha, dx)
        print("[Simulate] Done with step: ", i, " / substep: ", j, "\n", flush=True)

        if output_frame:
            get_central_vector(u_x, u_y, u_z, u)
            curl(u, w, inv_dx)
            w_numpy = w.to_numpy()
            w_norm = np.linalg.norm(w_numpy, axis=-1)
            write_vtk(w_norm, vtkdir, frame_idx, "vorticity")


            print(
                "[Simulate] Finished frame: ",
                frame_idx,
                " in ",
                i - last_output_substep,
                "substeps \n\n",
            )
            last_output_substep = i
            if frame_idx >= total_frames:
                break


if __name__ == "__main__":
    print("[Main] Begin")
    if len(sys.argv) <= 1:
        main(from_frame=from_frame)
    else:
        main(from_frame=from_frame, testing=True)
    print("[Main] Complete")
