from taichi_utils import *
import math
from hyperparameters import *
# 3D specific
# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: central position
# unit_x, unit_y: the direction
@ti.kernel
def compute_energy(smoke:ti.template(), u:ti.template(), energy:ti.template(), dx:float):
    energy[None] = 0.
    for i, j, k in smoke:
        energy[None] += 0.5 * (smoke[i, j, k].x + smoke[i, j, k].y) * (dx**3) * (u[i, j, k].x**2 + u[i, j, k].y**2 + u[i, j, k].z**2)

@ti.kernel
def add_vortex_ring(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
@ti.kernel
def no_bond(boundary_mask: ti.template(), boundary_vel: ti.template(), _t: float):
    pass
@ti.kernel
def add_vortex_ring_and_smoke_ink(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int, noise: ti.template(), positions: ti.template(), inv_dx:ti.template()):
    for l in range(num_samples):
        theta = l/num_samples * 2 * (math.pi)
        positions[l] = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
        noise_interped = interp_1(noise, positions[l], inv_dx)
        positions[l] += noise_interped

    #curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            p_sampled = positions[l]
            p_next = positions[(l+1)%num_samples]
            p_prev = positions[(l-1)%num_samples]
            curve_length = (p_next - p_sampled).norm()
            w_dir = (p_next - p_prev).normalized()
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * w_dir #(-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
    for i, j, k in vf:
        smoke_delta = 1 * delta
        for l in range(num_samples):
            p_sampled = positions[l]
            p_next = positions[(l+1)%num_samples]
            curve_length = (p_next - p_sampled).norm()
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            smokef[i,j,k][4] += curve_length * (ti.exp(-(r/smoke_delta) ** 3))
    
    for i, j, k in smokef:
        if smokef[i,j,k][4] > 0.002:
            if ti.random() < 0.1:
                smokef[i, j, k] = 10.0
            else:
                smokef[i,j,k][4] = 1.0
            smokef[i,j,k].xyz = color

# 3D specific
# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: central position
# unit_x, unit_y: the direction
@ti.func
def fractal_noise(x, y, z, octaves: ti.i32, lacunarity: float, gain: float):
    freq = 1.0
    amp = 1.0
    noise_sum = 0.0
    for _ in range(octaves):
        val = ti.sin(freq * x) * ti.sin(freq * y) * ti.cos(freq * z)
        noise_sum += amp * val
        freq *= lacunarity
        amp *= gain
    return noise_sum


@ti.kernel
def add_vortex_ring_and_smoke(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: int, num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k][3] > 0.002:
            if smokef[i,j,k][3] != 1.0:
                smokef[i, j, k][color] = 1.0
            smokef[i,j,k][3] = 1.0

def init_single_ring_ball(X, u, smoke):
    add_vortex_ring_and_smoke(w = 2.e-2, rad = 0.16, delta = 0.0168, c = ti.Vector([0.1,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)


def init_single_ring(X, u, smoke):
    add_vortex_ring_and_smoke(w = 2.e-2  * (127/159), rad = 0.21 * (127/159), delta = 0.0168  * (127/159), c = ti.Vector([0.3,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)

def init_single_ring_original(X, u, smoke):
    add_vortex_ring_and_smoke(w = 2.e-2, rad = 0.21, delta = 0.0168, c = ti.Vector([0.25,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)

def init_vorts_leapfrog(X, u):
    add_vortex_ring(w = 2.e-2, rad = 0.20, delta = 0.016, c = ti.Vector([0.23,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)

    add_vortex_ring(w = 2.e-2, rad = 0.20, delta = 0.016, c = ti.Vector([0.35,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)


def init_vorts_leapfrog_smoke(X, u, smoke):
    add_vortex_ring_and_smoke(w = 2.e-2, rad = 0.21, delta = 0.0168, c = ti.Vector([0.5,0.5,0.23]),
            unit_x = ti.Vector([1.,0.,0.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)

    add_vortex_ring_and_smoke(w = 2.e-2, rad = 0.21, delta = 0.0168, c = ti.Vector([0.5,0.5,0.36125]),
            unit_x = ti.Vector([1.,0.,0.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 1, num_samples = 500)

def init_vorts_headon(X, u):
    add_vortex_ring(w = 2.e-2, rad = 0.06, delta = 0.016, c = ti.Vector([0.4,1.0,1.0]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)

    add_vortex_ring(w = -2.e-2, rad = 0.06, delta = 0.016, c = ti.Vector([0.6,1.0,1.0]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, num_samples = 500)

def init_vorts_headon_smoke(X, u, smoke):
    add_vortex_ring_and_smoke(w = 2.e-2, rad = 0.06, delta = 0.016, c = ti.Vector([0.1,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)

    add_vortex_ring_and_smoke(w = -2.e-2, rad = 0.06, delta = 0.016, c = ti.Vector([0.4,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 1, num_samples = 500)

def init_vorts_headon_large_smoke(X, u, smoke):
    add_vortex_ring_and_smoke(w = 4.e-2, rad = 0.38, delta = 0.03, c = ti.Vector([0.45,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)

    add_vortex_ring_and_smoke(w = -4.e-2, rad = 0.38, delta = 0.03, c = ti.Vector([0.55,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke, color = 1, num_samples = 500)
    
def init_vorts_four(X, u, smoke):
    smoke.fill(0.)
    x_offset = 0.16
    y_offset = 0.16
    size = 0.15
    cos45 = ti.cos(math.pi/4)
    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5-y_offset, 0.5-x_offset, 1]),
        unit_x = ti.Vector([cos45, -cos45, 0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)

    add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5-y_offset, 0.5+x_offset, 1]),
        unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke, color = 1, num_samples = 500)
    
    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5+y_offset, 0.5-x_offset, 1]),
        unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke, color = 2, num_samples = 500)
    
    add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5+y_offset, 0.5+x_offset, 1]),
        unit_x = ti.Vector([cos45,-cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke, color = 3, num_samples = 500)

# def init_vorts_four(X, u, smoke):
#     smoke.fill(0.)
#     x_offset = 0.16
#     y_offset = 0.16
#     size = 0.15
#     cos45 = ti.cos(math.pi/4)
#     add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([1-x_offset,0.5-y_offset, 0.5]),
#         unit_x = ti.Vector([-cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
#         pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)
# 
#     add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([1+x_offset,0.5-y_offset, 0.5]),
#         unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
#         pf = X, vf = u, smokef = smoke, color = 1, num_samples = 500)
# 
#     add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([1-x_offset,0.5+y_offset, 0.5]),
#         unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
#         pf = X, vf = u, smokef = smoke, color = 2, num_samples = 500)
# 
# 
#     add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([1+x_offset,0.5+y_offset, 0.5]),
#         unit_x = ti.Vector([-cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
#         pf = X, vf = u, smokef = smoke, color = 3, num_samples = 500)
@ti.kernel
def add_vortex_ring_and_smoke_junwei(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point controls part of the curve
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k][3] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k][3] > 0.002:
            # if color[0] == 0. and color[1] == 0. and color[2] == 0.:
            #     smokef[i,j,k][3] = 1.0
            # else:
            #     smokef[i,j,k][3] = 0.0
            #     #smokef[i,j,k][3] = 4 * color[0] + 3 * color[1] + 2 * color[2]
            # smokef[i,j,k].xyz = color
            smokef[i, j, k][3] = 1.0
            smokef[i, j, k].xyz = color
        else:
            smokef[i,j,k] = ti.Vector([0.,0.,0.,0.])

def init_vorts_eight(X, u, smoke1, smoke2):
    smoke1.fill(0.)
    smoke2.fill(0.)
    size = 0.08
    cos45 = ti.cos(math.pi / 4)
    tan_theta = ti.sqrt(2) / 2
    cot_theta = ti.sqrt(2)
    x_offset = 0.16
    y_offset = 0.16
    z_offset = 0.16 / cos45 * tan_theta

    add_vortex_ring_and_smoke_junwei(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 - y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke1, color=ti.Vector([1, 0., 0.]), num_samples=500)

    add_vortex_ring_and_smoke_junwei(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 - y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0, 1, 0.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke_junwei(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 + y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0., 0., 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke_junwei(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 + y_offset, 0.5 - z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0., 0., 0.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke_junwei(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 - y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([0., 1, 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke_junwei(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 - y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([1, 0., 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke_junwei(w=2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 - x_offset, 0.5 + y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([1, 1, 0.]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke_junwei(w=-2.e-2, rad=size, delta=0.024, c=ti.Vector([0.5 + x_offset, 0.5 + y_offset, 0.5 + z_offset]),
                              unit_x=ti.Vector([-cos45, cos45, 0.]).normalized(), unit_y=ti.Vector([-cos45, -cos45, cot_theta]).normalized(),
                              pf=X, vf=u, smokef=smoke2, color=ti.Vector([1, 1, 1]), num_samples=500)

    add_fields(smoke1, smoke2, smoke1, 1.0)



def init_vorts_oblique(X, u):
    x_offset = 0.18
    z_offset = 0.18
    size = 0.15
    add_vortex_ring(w = -2.e-2, rad = size, delta = 0.018, c = ti.Vector([0.5-x_offset,0.5,0.5-z_offset]),
        unit_x = ti.Vector([-0.7,0.,0.7]).normalized(), unit_y = ti.Vector([0.,1., 0.]),
        pf = X, vf = u, num_samples = 500)
    add_vortex_ring(w = -2.e-2, rad = size, delta = 0.018, c = ti.Vector([0.5-x_offset,0.5,0.5+z_offset]),
        unit_x = ti.Vector([0.7,0.,0.7]).normalized(), unit_y = ti.Vector([0.,1., 0.]),
        pf = X, vf = u, num_samples = 500)

def init_vorts_oblique_smoke(X, u, smoke):
    x_offset = 0.18
    z_offset = 0.18
    size = 0.15
    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.018, c = ti.Vector([0.5-x_offset,0.5,0.5-z_offset]),
        unit_x = ti.Vector([-0.7,0.,0.7]).normalized(), unit_y = ti.Vector([0.,1., 0.]),
        pf = X, vf = u, smokef = smoke, color = 0, num_samples = 500)
    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.018, c = ti.Vector([0.5-x_offset,0.5,0.5+z_offset]),
        unit_x = ti.Vector([0.7,0.,0.7]).normalized(), unit_y = ti.Vector([0.,1., 0.]),
        pf = X, vf = u, smokef = smoke, color = 1, num_samples = 500)

# def init_vorts_oblique(X, u, smoke1, smoke2):
#     smoke1.fill(0.)
#     smoke2.fill(0.)
#     x_offset = 0.15
#     y_offset = 0.22
#     size = 0.13
#     add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.02, c = ti.Vector([0.5-x_offset,0.5-y_offset, 0.5]),
#         unit_x = ti.Vector([-0.7,0.7,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
#         pf = X, vf = u, smokef = smoke1, color = ti.Vector([1, 0.4, 0.4]), num_samples = 500)
# 
#     add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.02, c = ti.Vector([0.5+x_offset,0.5-y_offset, 0.5]),
#         unit_x = ti.Vector([0.7,0.7,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
#         pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 1, 0.6]), num_samples = 500)
#     
#     add_fields(smoke1, smoke2, smoke1, 1.0)

# some shapes (checkerboards...)

@ti.kernel
def stripe_func(qf: ti.template(), pf: ti.template(), \
                x_start: float, x_end: float,\
                y_start: float, y_end: float,
                z_start: float, z_end: float):
    for I in ti.grouped(qf):
        if x_start <= pf[I].x <= x_end and y_start <= pf[I].y <= y_end and z_start <= pf[I].z <= z_end:
            qf[I] = ti.Vector([1.0, 0.0, 0.0, 1.0])
        # else:
        #     qf[I] = ti.Vector([0.0, 0.0, 0.0, 0.0])

@ti.kernel
def double_stripe_func(qf: ti.template(), pf: ti.template(), 
                x_start1: float, x_end1: float,
                x_start2: float, x_end2: float):
    for I in ti.grouped(qf):
        if x_start1 <= pf[I].x <= x_end1 and 0.4 <= pf[I].y <= 0.6 and 0.4 <= pf[I].z <= 0.6:
            qf[I] = ti.Vector([1.0, 0., 0.])
        elif x_start2 <= pf[I].x <= x_end2 and 0.4 <= pf[I].y <= 0.6 and 0.4 <= pf[I].z <= 0.6:
            qf[I] = ti.Vector([0., 0., 1.0])
        else:
            qf[I] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def box_func(qf: ti.template(), pf: ti.template()):
    for I in ti.grouped(qf):
        if 0.9 <= pf[I].x <= 1.1 and 0.15 <= pf[I].y <= 0.85 and 0.15 <= pf[I].z <= 0.85:
            qf[I] = ti.Vector([1.0, 0.0, 0.0, 1])
        else:
            qf[I] = ti.Vector([0.0, 0.0, 0.0, 0.0])

@ti.kernel
def cylinder_func(qf: ti.template(), pf: ti.template(), x: float, z: float):
    radius = 0.02
    for I in ti.grouped(qf):
        if 0 < pf[I].y < 1: 
            dist = ti.sqrt((pf[I].x-x)**2 + (pf[I].z-z)**2)
            if dist <= radius:
                qf[I] = ti.Vector([1.0, 0.0, 0.0, 1])


@ti.kernel
def init_particles_pos_uniform(particles_pos: ti.template(), X: ti.template(),
                       res_x: int, res_y: int, particles_per_cell: int, dx: float, particles_per_cell_axis: int,
                       dist_between_neighbor: float):

    particles_x_num = particles_per_cell_axis * res_x
    particles_y_num = particles_per_cell_axis * res_y

    for i in particles_pos:
        # if particles_active[i] == 1:
            id_x = i % particles_x_num
            id_yz = i // particles_x_num
            id_y = id_yz % particles_y_num
            id_z = id_yz // particles_y_num
            particles_pos[i] = (ti.Vector([id_x, id_y, id_z]) + 0.5) * dist_between_neighbor

@ti.kernel
def distribute_particles_based_on_vorticity(particles_pos: ti.template(), 
                                            w: ti.template(),
                                            w_particles_per_cell: ti.template(),
                                            num_extra_particles: int, 
                                            dx: float) -> ti.i32:
    
    res_x, res_y, res_z = w.shape
    total_vorticity = 0.0
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        total_vorticity += w[i, j, k].norm()

    # w_particles_per_cell = ti.field(ti.i32, shape=(res_x, res_y, res_z))
    w_particles_per_cell.fill(0)
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        if w[i, j, k].norm() > 0:
            cell_vorticity = w[i, j, k].norm()
            w_particles_per_cell[i, j, k] = int(num_extra_particles * (cell_vorticity / total_vorticity))
    
    extra_particle_index = particles_pos.shape[0] - num_extra_particles

    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        if extra_particle_index <= particles_pos.shape[0] - 1:
            num_particles = w_particles_per_cell[i, j, k]
            particles_per_axis = int(num_particles ** (1/3)) + 1  # 确保每个轴上的粒子数
            if num_particles > 0:
                for pi, pj, pk in ti.ndrange(particles_per_axis, particles_per_axis, particles_per_axis):
                    if pi * particles_per_axis ** 2 + pj * particles_per_axis + pk < num_particles:
                        id_x = (i * particles_per_axis + pi) * dx / particles_per_axis
                        id_y = (j * particles_per_axis + pj) * dx / particles_per_axis
                        id_z = (k * particles_per_axis + pk) * dx / particles_per_axis
                        particles_pos[extra_particle_index] = ti.Vector([id_x, id_y, id_z])
                        extra_particle_index += 1
    
    return extra_particle_index


@ti.kernel
def init_particles_w(particles_init_w: ti.template(), particles_pos: ti.template(),
                       w_x: ti.template(), w_y: ti.template(), w_z:ti.template(), C_x: ti.template(),
                       C_y: ti.template(), C_z: ti.template(), dx: float, distribute_idx: int):
    for i in ti.ndrange(distribute_idx):
        particles_init_w[i], _, new_C_x, new_C_y, new_C_z = interp_u_MAC_grad_w(w_x, w_y, w_z, particles_pos[i], dx)
        # C_x[i] = new_C_x
        # C_y[i] = new_C_y
        # C_z[i] = new_C_z
        # particles_init_w[i] = particles_w[i]

@ti.kernel
def init_particles_w_grad_w(particles_w: ti.template(), particles_pos: ti.template(), w_x: ti.template(),
                              w_y: ti.template(), w_z: ti.template(), C_x: ti.template(), C_y: ti.template(),
                              C_z: ti.template(), dx: float, distribute_idx: int):
    for i in ti.ndrange(distribute_idx):
        particles_w[i], _, new_C_x, new_C_y, new_C_z = interp_u_MAC_grad_w(w_x, w_y, w_z, particles_pos[i], dx)
        C_x[i] = new_C_x
        C_y[i] = new_C_y
        C_z[i] = new_C_z

@ti.kernel
def init_particles_smoke(particles_smoke: ti.template(), particles_grad_smoke: ti.template(),
                         particles_pos: ti.template(), smoke: ti.template(), dx: float):
    for i in particles_smoke:
        particles_smoke[i], particles_grad_smoke[i] = interp_u_MAC_smoke(smoke, particles_pos[i], dx)