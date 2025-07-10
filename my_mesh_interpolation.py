import taichi as ti
import numpy as np
from my_lbvh import *
# ti.init(arch=ti.cuda)
padding = 1
@ti.func
def min3(a, b, c):
    return ti.min(a, ti.min(b, c))

@ti.func
def max3(a, b, c):
    return ti.max(a, ti.max(b, c))

@ti.func
def plane_box_overlap_3d(normal, vert, box_half_size):
    vmin = ti.Vector.zero(float, 3)
    vmax = ti.Vector.zero(float, 3)
    for q in range(3):
        v = vert[q]
        if normal[q] > 0.0:
            vmin[q] = -box_half_size[q] - v
            vmax[q] = box_half_size[q] - v
        else:
            vmin[q] = box_half_size[q] - v
            vmax[q] = -box_half_size[q] - v
    result = False
    if ti.math.dot(normal, vmin) > 0.0:
        result = False
    if ti.math.dot(normal, vmax) >= 0.0:
        result = True
    return result
            
@ti.func
def Axis_Test_X01(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = a * v0[1] - b * v0[2]
    p2 = a * v2[1] - b * v2[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p2): 
        min = p0
        max = p2
    else:
        min = p2
        max = p0
    rad = fa * box_half_size[1] + fb * box_half_size[2]
    if ((min > rad) or (max < -rad)):
        result = False
    return result

@ti.func
def Axis_Test_X2(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = a * v0[1] - b * v0[2]
    p1 = a * v1[1] - b * v1[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p1):
        min = p0
        max = p1
    else:
        min = p1
        max = p0
    rad = fa * box_half_size[1] + fb * box_half_size[2]
    if ((min > rad) or (max < -rad)):
        result = False
    return result
    
@ti.func
def Axis_Test_Y02(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = -a * v0[0] + b * v0[2]
    p2 = -a * v2[0] + b * v2[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p2):
        min = p0
        max = p2
    else:
        min = p2
        max = p0
    rad = fa * box_half_size[0] + fb * box_half_size[2]
    if ((min > rad) or (max < -rad)):
        result = False
    return result
    
@ti.func
def Axis_Test_Y1(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = -a * v0[0] + b * v0[2]
    p1 = -a * v1[0] + b * v1[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p1):
        min = p0
        max = p1
    else:
        min = p1
        max = p0
    rad = fa * box_half_size[0] + fb * box_half_size[2]
    if ((min > rad) or (max < -rad)):
        result = False
    return result

@ti.func
def Axis_Test_Z12(a, b, fa, fb, v0, v1, v2, box_half_size):
    p1 = a * v1[0] - b * v1[1]
    p2 = a * v2[0] - b * v2[1]
    min = 0.0
    max = 0.0
    result = True
    if (p2 < p1):
        min = p2
        max = p1
    else:
        min = p1
        max = p2
    
    rad = fa * box_half_size[0] + fb * box_half_size[1]
    if ((min > rad) or (max < -rad)):
        result = False
    return result
@ti.func
def Axis_Test_Z0(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = a * v0[0] - b * v0[1]
    p1 = a * v1[0] - b * v1[1]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p1):
        min = p0
        max = p1
    else:
        min = p1
        max = p0
    rad = fa * box_half_size[0] + fb * box_half_size[1]
    if ((min > rad) or (max < -rad)):
        result = False
    return result
    
@ti.func
def triangle_box_overlap_3d(box_center, box_half_size, x0, x1, x2):
    min = 0.0
    max = 0.0
    v0 = x0 - box_center
    v1 = x1 - box_center
    v2 = x2 - box_center
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    result = True
    fex = ti.abs(e0[0])
    fey = ti.abs(e0[1])
    fez = ti.abs(e0[2])
    if (not Axis_Test_X01(e0[2], e0[1], fez, fey, v0, v1, v2, box_half_size)):
         result = False
    if (not Axis_Test_Y02(e0[2], e0[0], fez, fex, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Z12(e0[1], e0[0], fey, fex, v0, v1, v2, box_half_size)):
        result = False
    
    
    fex = ti.abs(e1[0])
    fey = ti.abs(e1[1])
    fez = ti.abs(e1[2])
    if (not Axis_Test_X01(e1[2], e1[1], fez, fey, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Y02(e1[2], e1[0], fez, fex, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Z0(e1[1], e1[0], fey, fex, v0, v1, v2, box_half_size)):
        result = False
    
    fex = ti.abs(e2[0])
    fey = ti.abs(e2[1])
    fez = ti.abs(e2[2])
    if (not Axis_Test_X2(e2[2], e2[1], fez, fey, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Y1(e2[2], e2[0], fez, fex, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Z12(e2[1], e2[0], fey, fex, v0, v1, v2, box_half_size)):
        result = False

    min = min3(v0[0], v1[0], v2[0])
    max = max3(v0[0], v1[0], v2[0])
    if ((min > box_half_size[0]) or (max < -box_half_size[0])):
        result = False

    min = min3(v0[1], v1[1], v2[1])
    max = max3(v0[1], v1[1], v2[1])
    if ((min > box_half_size[1]) or (max < -box_half_size[1])):
        result = False

    min = min3(v0[2], v1[2], v2[2])
    max = max3(v0[2], v1[2], v2[2])
    if ((min > box_half_size[2]) or (max < -box_half_size[2])):
        result = False
    normal = ti.math.cross(e0, e1)
    if (not plane_box_overlap_3d(normal, v0, box_half_size)):
        result = False
    return result



@ti.func
def tri_area(v0, v1, v2):
    return 0.5 * ti.math.cross((v1 - v0), (v2 - v0)).norm()

@ti.func
def ray_box_intersect_3d(ray_origin, ray_direction, t0, t1, box_min, box_max):
    tmin = t0
    tmax = t1
    ret = True
    for i in ti.static(range(3)):
        invD = 1.0 / ray_direction[i]
        t0i = (box_min[i] - ray_origin[i]) * invD
        t1i = (box_max[i] - ray_origin[i]) * invD
        if invD < 0.0:
            t0i, t1i = t1i, t0i
        tmin = ti.max(tmin, t0i)
        tmax = ti.min(tmax, t1i)
        if tmax < tmin:
            ret = False
    return ret



@ti.func
def ray_tri_intersect_3d(ray_origin, ray_direction, p0, p1, p2):
    EPSILON = 1e-6
    intersects = False  # Initialize as False
    edge1 = p1 - p0
    edge2 = p2 - p0
    h = ray_direction.cross(edge2)
    a = edge1.dot(h)
    if ti.abs(a) >= EPSILON:
        f = 1.0 / a
        s = ray_origin - p0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(edge1)
            v = f * ray_direction.dot(q)
            if 0.0 <= v and u + v <= 1.0:
                t = f * edge2.dot(q)
                if t > EPSILON:
                    intersects = True  # Intersection occurs
    return intersects

@ti.func
def ray_tri_intersect_3d_new(ray_origin, ray_direction, p0, p1, p2):
    EPSILON = 1e-6
    intersects = False
    t = 0.0  # Initialize t
    edge1 = p1 - p0
    edge2 = p2 - p0
    h = ray_direction.cross(edge2)
    a = edge1.dot(h)
    if ti.abs(a) >= EPSILON:
        f = 1.0 / a
        s = ray_origin - p0
        u = f * s.dot(h)
        if 0.0 <= u <= 1.0:
            q = s.cross(edge1)
            v = f * ray_direction.dot(q)
            if 0.0 <= v and u + v <= 1.0:
                t_candidate = f * edge2.dot(q)
                if t_candidate > EPSILON:
                    intersects = True
                    t = t_candidate
    return intersects, t


@ti.kernel
def tribox_ff_voxelize(bvh: ti.template(), surf_occupancy: ti.template(), ti_faces: ti.template(), 
                       ti_vertices: ti.template(), res_x: int, res_y: int, res_z: int, dx: float):
    offset = 0.5 * ti.Vector([1.0, 1.0, 1.0])
    half_size = ti.Vector([dx * 0.5, dx * 0.5, dx * 0.5])
    for I in ti.grouped(surf_occupancy):
        # Test whether this voxel intersects with some triangle
        box_center = (I + offset) * dx
        query_min = box_center - half_size
        query_max = box_center + half_size
        num_candidates, candidate_list = bvh.get_candidate_triangles_box(query_min, query_max)
        # num_candidates = bvh.candidate_count[None]
        intersected = False
        for idx in range(num_candidates):
            tri_idx = candidate_list[idx]
            p0 = ti_vertices[ti_faces[tri_idx][0]]
            p1 = ti_vertices[ti_faces[tri_idx][1]]
            p2 = ti_vertices[ti_faces[tri_idx][2]]
            if triangle_box_overlap_3d(box_center, half_size, p0, p1, p2):
                intersected = True
                break  # Exit early if intersection is found
        surf_occupancy[I] = 1 if (intersected) else 0

@ti.kernel
def raytri_voxelize(bvh: ti.template(), surf_occupancy: ti.template(), ti_faces: ti.template(), 
                       ti_vertices: ti.template(), res_x: int, res_y: int, res_z: int, dx: float):
    offset = 0.5 * ti.Vector([1.0, 1.0, 1.0])
    half_size = ti.Vector([dx * 0.5, dx * 0.5, dx * 0.5])
    total_length = ti.sqrt(res_x**2 + res_y**2 + res_z**2) * dx
    for I in ti.grouped(surf_occupancy):
        # Test whether this voxel intersects with some triangle
        box_center = (I + offset) * dx
        ray_origin = ti.Vector([0., 0., 0.])
        ray_direction = box_center
        t0 = 0.0
        t1 = 1.0
        num_candidates, candidate_list = bvh.get_candidate_triangles_ray(ray_origin, ray_direction, t0, t1)
        # num_candidates = bvh.candidate_count[None]
        intersection_count = 0
        for idx in range(num_candidates):

            tri_idx = candidate_list[idx]
            p0 = ti_vertices[ti_faces[tri_idx][0]]
            p1 = ti_vertices[ti_faces[tri_idx][1]]
            p2 = ti_vertices[ti_faces[tri_idx][2]]
            intersects, t_intersect = ray_tri_intersect_3d_new(ray_origin, ray_direction, p0, p1, p2)
            if intersects and t_intersect < 1.0:
                intersection_count += 1
        # Determine if the voxel is inside the mesh based on the parity of intersection_count
        surf_occupancy[I] = 1 if (intersection_count % 2 == 1) else 0

@ti.kernel
def initialize_occupancy(occupancy:ti.template(), surf_occupancy:ti.template(),  res_x: int, res_y: int, res_z: int):
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        # Initialize occupancy to 0
        occupancy[i, j, k] = 0

        # Set occupancy to 1 on the boundary if not part of the surface
        if (i == 0 or i == res_x - 1 or
            j == 0 or j == res_y - 1 or
            k == 0 or k == res_z - 1):
            if surf_occupancy[i, j, k] == 0:
                occupancy[i, j, k] = 1

@ti.kernel
def flood_fill(changes:ti.template(), occupancy:ti.template(), surf_occupancy:ti.template(), res_x: int, res_y: int, res_z: int):
    changes[None] = 0  # Reset changes for this iteration
    for i, j, k in ti.ndrange(res_x, res_y, res_z):
        if occupancy[i, j, k] == 0 and surf_occupancy[i, j, k] == 0:
            # Check neighboring voxels
            filled = False
            if i > 0 and occupancy[i - 1, j, k] == 1:
                filled = True
            if i < res_x - 1 and occupancy[i + 1, j, k] == 1:
                filled = True
            if j > 0 and occupancy[i, j - 1, k] == 1:
                filled = True
            if j < res_y - 1 and occupancy[i, j + 1, k] == 1:
                filled = True
            if k > 0 and occupancy[i, j, k - 1] == 1:
                filled = True
            if k < res_z - 1 and occupancy[i, j, k + 1] == 1:
                filled = True
            if filled:
                occupancy[i, j, k] = 1
                changes[None] = 1  # Mark that a change has occurred

@ti.kernel
def fill_internal_occupancy(outside_occupancy:ti.template(), inner_occupancy:ti.template(), surf_occupancy:ti.template()):
    for i,j,k in inner_occupancy:
        if outside_occupancy[i,j,k] == 0 or surf_occupancy[i,j,k] == 1:
            inner_occupancy[i,j,k] = 1
        else:
            inner_occupancy[i,j,k] = 0

@ti.kernel
def fill_inner_boundary(all_occupancy:ti.template(), surf_occupancy:ti.template(), inner_occupancy:ti.template()):
    for i,j,k in inner_occupancy:
        if all_occupancy[i,j,k] == 1 and surf_occupancy[i,j,k] == 0:
            inner_occupancy[i,j,k] = 1
        else:
            inner_occupancy[i,j,k] = 0

@ti.kernel
def calculate_vel(v_old:ti.template(), v_new:ti.template(), vel:ti.template(), dt:float):
    for I in ti.grouped(v_old):
        vel_t = (v_new[I] - v_old[I]) / dt
        # when use bird
        if (vel[I] - vel_t).norm() / dt > 5:
            vel[I].fill(0.0)
        else:
            vel[I] = vel_t

@ti.func
def in_dual_cell(pos, center, dx):
    margin = padding * 0.505 * dx
    return (pos[0] > center[0] - margin) and (pos[0] < center[0] + margin) \
        and (pos[1] > center[1] - margin) and (pos[1] < center[1] + margin) \
        and (pos[2] > center[2] - margin) and (pos[2] < center[2] + margin)
        
@ti.kernel
def calculate_bv(bvh:ti.template(), bv:ti.template(), meshv:ti.template(), ti_v:ti.template(), ti_f:ti.template(), dx:ti.f32):
    offset = 0.5 * ti.Vector.one(float, 3)
    # occu_offset = ti.Vector.unit(3, dim)
    half_size = ti.Vector.one(float, 3) * dx * 0.5
    total_cell = 0
    for I in ti.grouped(bv):
        area_sum = 0.0
        box_center = (I + offset) * dx
        query_min = box_center - half_size
        query_max = box_center + half_size
        num_candidates, candidate_list = bvh.get_candidate_triangles_box(query_min, query_max)
        for idx in range(num_candidates):
            tri_idx = candidate_list[idx]
            p0 = ti_v[ti_f[tri_idx][0]]
            p1 = ti_v[ti_f[tri_idx][1]]
            p2 = ti_v[ti_f[tri_idx][2]]
            if triangle_box_overlap_3d(box_center, half_size, p0, p1, p2):
                counts = 0
                for ii in range(11):
                    for jj in range(11 - ii):
                        pt = ii / 10.0 * p0 + jj / 10.0 * p1 + (10 - ii - jj) / 10.0 * p2
                        if (in_dual_cell(pt, box_center, dx)):
                            counts += 1

                tarea = counts / 66.0 * tri_area(p0, p1, p2)
                area_sum += tarea
                bv[I] += 1.0 / 3 * tarea * meshv[ti_f[tri_idx][0]]
                bv[I] += 1.0 / 3 * tarea * meshv[ti_f[tri_idx][1]]
                bv[I] += 1.0 / 3 * tarea * meshv[ti_f[tri_idx][2]]
        if area_sum == 0.0:
            # field[I].fill(0.0)
            area_sum = 1e-6
        bv[I] /= area_sum

@ti.kernel
def calculate_bv_splitnormal(bvh: ti.template(),
                             bv: ti.template(),
                             bv_tangent: ti.template(),
                             meshv: ti.template(),
                             ti_v: ti.template(),
                             ti_f: ti.template(),
                             dx: ti.f32):
    """
    For each grid cell, decompose the mesh velocity into triangle-normal (stored in bv)
    and tangential (stored in bv_tangent). The velocity is area-weighted based on
    the portion of the triangle overlapping that cell.
    """
    offset = 0.5 * ti.Vector.one(float, 3)
    half_size = ti.Vector.one(float, 3) * dx * 0.5

    for I in ti.grouped(bv):
        area_sum = 0.0
        box_center = (I + offset) * dx
        query_min = box_center - half_size
        query_max = box_center + half_size

        # Zero out accumulators before summation
        bv[I] = ti.Vector.zero(float, 3)
        bv_tangent[I] = ti.Vector.zero(float, 3)

        num_candidates, candidate_list = bvh.get_candidate_triangles_box(query_min, query_max)
        for idx in range(num_candidates):
            tri_idx = candidate_list[idx]
            p0 = ti_v[ti_f[tri_idx][0]]
            p1 = ti_v[ti_f[tri_idx][1]]
            p2 = ti_v[ti_f[tri_idx][2]]

            if triangle_box_overlap_3d(box_center, half_size, p0, p1, p2):
                # We do a local sampling-based approach (counts) to estimate how much
                # of the triangle belongs to this cell
                counts = 0
                for ii in range(11):
                    for jj in range(11 - ii):
                        pt = (ii / 10.0) * p0 + (jj / 10.0) * p1 + ((10 - ii - jj) / 10.0) * p2
                        if in_dual_cell(pt, box_center, dx):
                            counts += 1

                # Weighted area contribution of this triangle portion to the cell
                tarea = (counts / 66.0) * tri_area(p0, p1, p2)
                area_sum += tarea

                # Compute the triangle normal (cross product) and normalize
                tri_normal = (p1 - p0).cross(p2 - p0)
                norm_len = tri_normal.norm()
                n_hat = ti.Vector([0.0, 0.0, 0.0])
                if norm_len > 1e-12:
                    n_hat = tri_normal / norm_len

                # For each vertex velocity, compute normal & tangent parts and accumulate
                for vidx in range(3):
                    v = meshv[ti_f[tri_idx][vidx]]
                    v_normal = (v.dot(n_hat)) * n_hat
                    v_tangent = v - v_normal
                    bv[I] += (1.0 / 3.0) * tarea * v_normal
                    bv_tangent[I] += (1.0 / 3.0) * tarea * v_tangent

        # Avoid division by zero
        if area_sum == 0.0:
            area_sum = 1e-6

        # Convert from 'summed velocity * area' to an average velocity contribution
        bv[I] /= area_sum
        bv_tangent[I] /= area_sum


@ti.kernel
def calculate_bv_face(bvh:ti.template(), bv_x:ti.template(), bv_y:ti.template(), 
                      bv_z:ti.template(), meshv:ti.template(), 
                      ti_v:ti.template(), ti_f:ti.template(), dx:ti.f32):
    offset = 0.5 * ti.Vector.one(float, 3)
    # occu_offset = ti.Vector.unit(3, dim)
    half_size = ti.Vector.one(float, 3) * dx * 0.5
    total_cell = 0
    for I in ti.grouped(bv_x):
        area_sum = 0.0
        box_center = (I + offset) * dx
        query_min = box_center - half_size
        query_max = box_center + half_size
        num_candidates, candidate_list = bvh.get_candidate_triangles_box(query_min, query_max)
        for idx in range(num_candidates):
            tri_idx = candidate_list[idx]
            p0 = ti_v[ti_f[tri_idx][0]]
            p1 = ti_v[ti_f[tri_idx][1]]
            p2 = ti_v[ti_f[tri_idx][2]]
            if triangle_box_overlap_3d(box_center, half_size, p0, p1, p2):
                counts = 0
                for ii in range(11):
                    for jj in range(11 - ii):
                        pt = ii / 10.0 * p0 + jj / 10.0 * p1 + (10 - ii - jj) / 10.0 * p2
                        if (in_dual_cell(pt, box_center, dx)):
                            counts += 1

                tarea = counts / 66.0 * tri_area(p0, p1, p2)
                area_sum += tarea
                bv[I] += 1.0 / 3 * tarea * meshv[ti_f[tri_idx][0]]
                bv[I] += 1.0 / 3 * tarea * meshv[ti_f[tri_idx][1]]
                bv[I] += 1.0 / 3 * tarea * meshv[ti_f[tri_idx][2]]
        if area_sum == 0.0:
            # field[I].fill(0.0)
            area_sum = 1e-6
        bv[I] /= area_sum


@ti.kernel
def gen_boundary_mask(field:ti.template(), occu:ti.template(), ti_v:ti.template(), ti_f:ti.template(), ti_vel:ti.template(), dx:float):
    offset = 0.5 * ti.Vector.one(float, 3)
    # occu_offset = ti.Vector.unit(3, dim)
    half_size = ti.Vector.one(float, 3) * dx * 0.5
    total_cell = 0
    for I in ti.grouped(field):
        inter = 0.0
        # field[I] = 
        face_center = (I + offset) * dx
        area_sum = 0.0
        
        for J in range(ti_f.shape[0]):
            p0 = ti_v[ti_f[J][0]]
            p1 = ti_v[ti_f[J][1]]
            p2 = ti_v[ti_f[J][2]]
            
            if not triangle_box_overlap_3d(face_center, half_size, p0, p1, p2):
                inter += 1
                continue
            # occu[I] = 1
            counts = 0
            for ii in range(11):
                for jj in range(11 - ii):
                    pt = ii / 10.0 * p0 + jj / 10.0 * p1 + (10 - ii - jj) / 10.0 * p2
                    if (in_dual_cell(pt, face_center, dx)):
                        counts += 1

            tarea = counts / 66.0 * tri_area(p0, p1, p2)
            area_sum += tarea
            field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][0], ti.int32)]
            field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][1], ti.int32)]
            field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][2], ti.int32)]
        if area_sum == 0.0:
            # field[I].fill(0.0)
            area_sum = 1e-6
        field[I] /= area_sum
            
#     for I in ti.grouped(field):
#         if occu[I] >= 1 and intersect[I] != 1:
#             nearest = 1e20
#             # loc = (I * dx)
#             for a in ti.ndrange(-5, 5):
#                 for b in ti.ndrange(-5, 5):
#                     for c in ti.ndrange(-5, 5):
#                         offset = ti.Vector([a, b, c])
#                         if intersect[I + ti.cast(offset, ti.int32)] >= 1 and offset.norm() * dx < nearest:
#                             nearest = offset.norm() * dx
#                             field[I] = intersect[I + ti.cast(offset, ti.int32)]
            
    

@ti.func
def point_segment_distance(x0 : ti.template(), x1 : ti.template(), x2 : ti.template()):
    dx = x2 - x1
    m2 = ti.math.dot(dx, dx)
    s12 = ti.math.dot(dx, x2 - x0) / m2
    if s12 < 0.0:
        s12 = 0.0
    elif s12 > 1.0:
        s12 = 1.0
    return (x0 - (s12 * x1 + (1 - s12) * x2)).norm() 

@ti.func
def point_triangle_distance(x0 : ti.template(), x1 : ti.template(), x2 : ti.template(), x3 : ti.template()):
    # first find barycentric coordinates of closest point on infinite plane
    result = 0.0
    x13 = x1 - x3
    x23 = x2 - x3
    x03 = x0 - x3
    m13 = ti.math.dot(x13, x13)
    m23 = ti.math.dot(x23, x23)
    d = ti.math.dot(x13, x23)
    invdet = 1.0 / ti.max(m13 * m23 - d * d, 1e-30)
    a = ti.math.dot(x13, x03)
    b = ti.math.dot(x23, x03)
    # the barycentric coordinates themselves
    w23 = invdet * (m23 * a - d * b)
    w31 = invdet * (m13 * b - d * a)
    w12 = 1 - w23 - w31
    if (w23 >= 0 and w31 >= 0 and w12 >= 0):
        # if we're inside the triangle
        result = (x0 - (w23 * x1 + w31 * x2 + w12 * x3)).norm()
    else:
        # we have to clamp to one of the edges
        if w23 > 0:
            #this rules out edge 2-3 for us
            result = ti.min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x1, x3))
        elif w31 > 0:
            # this rules out edge 1-3
            result = ti.min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x2, x3))
        else:
            # w12 must be >0, ruling out edge 1-2
            result = ti.min(point_segment_distance(x0, x1, x3), point_segment_distance(x0, x2, x3))
    return result

        
@ti.kernel
def update_boundary_mask(occu:ti.template(), ti_v:ti.template(), ti_f:ti.template(), dx:float):
    offset = 0.5 * ti.Vector.one(float, 3)
    # occu_offset = ti.Vector.unit(3, dim)
    half_size = ti.Vector.one(float, 3) * dx * 0.5
    for I in ti.grouped(occu):
        face_center = (I + offset) * dx
        area_sum = 0.0
        
        for J in range(ti_f.shape[0]):
            p0 = ti_v[ti_f[J][0]]
            p1 = ti_v[ti_f[J][1]]
            p2 = ti_v[ti_f[J][2]]
            
            if not triangle_box_overlap_3d(face_center, half_size, p0, p1, p2):
                continue
            occu[I] = ti.cast(1, ti.int32)
        
