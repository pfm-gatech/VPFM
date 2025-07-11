import taichi as ti
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
eps = 1.e-6
data_type = ti.f32


@ti.func
def interp_grad_2_smoke(vf, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))
    l = ti.max(1., ti.min(w, w_dim - 2 - eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = ti.Vector([0.0, 0.0, 0.0, 0.0])
    partial_y = ti.Vector([0.0, 0.0, 0.0, 0.0])
    partial_z = ti.Vector([0.0, 0.0, 0.0, 0.0])
    interped = ti.Vector([0.0, 0.0, 0.0, 0.0])

    new_C = ti.Matrix.zero(float, 3, 4)
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                # imp_value = sample(imp, iu + i, iv + j)
                dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)
                dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i)
                dw_z = 1. / dx * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i)
                partial_x += value * dw_x
                partial_y += value * dw_y
                partial_z += value * dw_z
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)
                new_C += ti.Vector([dw_x, dw_y, dw_z]).outer_product(value)
                # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, new_C


@ti.func
def interp_u_MAC_smoke(smoke, p, dx):
    interped_smoke, grad_smoke = interp_grad_2_smoke(smoke, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5)
    return interped_smoke, grad_smoke

@ti.kernel
def test_inbound(inbond:ti.template(), frx:ti.template(), fry:ti.template(), frz:ti.template()):
    for I in ti.grouped(inbond):
        if inbond[I] == 0 and (frx[I] <=0 and frx[I+ti.Vector.unit(3, 0)] <=0 and fry[I] <=0 and fry[I+ti.Vector.unit(3, 1)]<=0 and frz[I]<=0 and frz[I+ti.Vector.unit(3, 2)]<=0):
            inbond[I] = 1
@ti.kernel
def sample_fx_points(nx2ijk: ti.template(), points: ti.types.ndarray(), nx:ti.template(), stride:int, dx:float):
    for n in range(nx[None]):
        i, j, k = nx2ijk[n][0], nx2ijk[n][1], nx2ijk[n][2]
        for p in range(stride * stride):
            u = (p % stride + 0.5) / stride 
            v = (p // stride + 0.5) / stride
            x = i + 0.0
            y = j + u
            z = k + v
            points[n, p, 0] = x * dx
            points[n, p, 1] = y * dx
            points[n, p, 2] = z * dx

@ti.kernel
def sample_fy_points(ny2ijk: ti.template(), points: ti.types.ndarray(), ny:ti.template(), stride:int, dx:float):
    for n in range(ny[None]):
        i, j, k = ny2ijk[n][0], ny2ijk[n][1], ny2ijk[n][2]
        for p in range(stride * stride):
            u = (p % stride + 0.5) / stride 
            v = (p // stride + 0.5) / stride
            x = i + u   
            y = j + 0.0 
            z = k + v   
            points[n, p, 0] = x * dx
            points[n, p, 1] = y * dx
            points[n, p, 2] = z * dx

@ti.kernel
def sample_fz_points(nz2ijk: ti.template(), points: ti.types.ndarray(), nz:ti.template(), stride:int, dx:float):
    for n in range(nz[None]):
        i, j, k = nz2ijk[n][0], nz2ijk[n][1], nz2ijk[n][2]
        for p in range(stride * stride):
            u = (p % stride + 0.5) / stride 
            v = (p // stride + 0.5) / stride
            x = i + u  
            y = j + v  
            z = k + 0.0
            points[n, p, 0] = x * dx
            points[n, p, 1] = y * dx
            points[n, p, 2] = z * dx

@ti.kernel
def assign_fractions(nx2ijk: ti.template(), fractions: ti.types.ndarray(),
                    surf_face_fraction_x: ti.template(), na:ti.template()):
    for n in range(na[None]):
        i, j, k = nx2ijk[n][0], nx2ijk[n][1], nx2ijk[n][2]
        surf_face_fraction_x[i, j, k] = fractions[n]

@ti.kernel
def fill_fractions(fractions_x:ti.template(), fractions_y:ti.template(), fractions_z:ti.template(),
                     surf_occupancy:ti.template(), occupancy:ti.template()):
    for i,j,k in occupancy:
        if occupancy[i,j,k] <= 0:
            fractions_x[i,j,k]=1
            fractions_x[i+1,j,k]=1
            fractions_y[i,j,k]=1
            fractions_y[i,j+1,k]=1
            fractions_z[i,j,k]=1
            fractions_z[i,j,k+1]=1

    shape_x,shape_y,shape_z=fractions_x.shape
    for i,j,k in fractions_x:
        if(i==0 or i== shape_x-1):
            fractions_x[i,j,k]=0

    shape_x,shape_y,shape_z=fractions_y.shape
    for i,j,k in fractions_y:
        if(j==0 or j== shape_y-1):
            fractions_y[i,j,k]=0

    shape_x,shape_y,shape_z=fractions_z.shape
    for i,j,k in fractions_z:
        if(k==0 or k== shape_z-1):
            fractions_z[i,j,k]=0

@ti.kernel
def assign_mapping(mask:ti.template(), mapping_n2ijk:ti.template(), 
                   mapping_ijk2n:ti.template(), n:ti.template()):
    for i,j,k in mask:
        if mask[i,j,k] == 1:
            mapping_ijk2n[i,j,k] = ti.atomic_add(n[None], 1)
            mapping_n2ijk[mapping_ijk2n[i,j,k]] = ti.Vector([i,j,k])

@ti.kernel
def assign_mapping_n2ijk(mask:ti.template(), mapping_n2ijk:ti.template(), n:ti.template()):
    for i,j,k in mask:
        if mask[i,j,k] == 1:
            a = ti.atomic_add(n[None], 1)
            mapping_n2ijk[a] = ti.Vector([i,j,k])

@ti.kernel
def reset_to_zero_gradF(
    T_x: ti.template(),
    T_y: ti.template(),
    T_z: ti.template(),
):
    for I in ti.grouped(T_x):
        # T_x[I] = ti.Vector.unit(3, 0)
        T_x[I] = ti.Matrix.zero(float, 3, 3)
    for I in ti.grouped(T_y):
        # T_y[I] = ti.Vector.unit(3, 1)
        T_y[I] = ti.Matrix.zero(float, 3, 3)
    for I in ti.grouped(T_z):
        # T_z[I] = ti.Vector.unit(3, 2)
        T_z[I] = ti.Matrix.zero(float, 3, 3)

@ti.kernel
def subtract_grad_p(u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), p:ti.template(), bm:ti.template(),
                    surf_face_fraction_x: ti.template(), surf_face_fraction_y: ti.template(), \
                        surf_face_fraction_z: ti.template(),
                    wall_u:ti.f32, wall_v:ti.f32, wall_w:ti.f32):
    u_dim, v_dim, w_dim = p.shape
    for i, j, k in u_x:
        if i>=1 and i <=u_dim-2 and surf_face_fraction_x[i,j,k]>0:
            pr = sample(p, i, j, k)
            pl = sample(p, i-1, j, k)
            u_x[i,j,k] -= pr - pl
    for i, j, k in u_y:
        if j>=1 and j<=v_dim-2 and surf_face_fraction_y[i,j,k]>0:
            pt = sample(p, i, j, k)
            pb = sample(p, i, j-1, k)
            u_y[i,j,k] -= pt - pb
    for i, j, k in u_z:
        if k>=1 and k<= w_dim-2 and surf_face_fraction_z[i,j,k]>0:
            pc = sample(p, i, j, k)
            pa = sample(p, i, j, k-1)
            u_z[i,j,k] -= pc - pa
    
    udim, vdim, wdim = u_x.shape
    for i,j,k in u_x:
        if i==0 or i==udim-1:
            u_x[i,j,k] = wall_u
    
    udim, vdim, wdim = u_y.shape
    for i,j,k in u_y:
        if j==0 or j==vdim-1:
            u_y[i,j,k] = wall_v
    
    udim, vdim, wdim = u_z.shape
    for i,j,k in u_z:
        if k==0 or k==wdim-1:
            u_z[i,j,k] = wall_w


@ti.kernel
def compute_energy(u:ti.template(), energy:ti.template(), dx:float):
    energy[None] = 0.
    for i, j, k in u:
        energy[None] += (u[i, j, k].x**2 + u[i, j, k].y**2 + u[i, j, k].z**2)



@ti.func
def interp_grad_2_F_x(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = ti.Matrix.zero(float, 3, 3)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += inv_dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i))
    
    return partial_x

@ti.func
def interp_grad_2_F_y(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    partial_y = ti.Matrix.zero(float, 3, 3)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_y += inv_dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i))
    
    return partial_y

@ti.func
def interp_grad_2_F_z(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    partial_z = ti.Matrix.zero(float, 3, 3)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_z += inv_dx * (value * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i))
    
    return partial_z


@ti.kernel
def compute_gradF_interp(Fx_face:ti.template(), Fy_face:ti.template(), Fz_face:ti.template(), 
                  Fgradx_center:ti.template(), Fgrady_center:ti.template(), Fgradz_center:ti.template(), X:ti.template(), 
                  dx:ti.f32):
    for I in ti.grouped(Fgradx_center):
        Fgradx_center[I] = interp_grad_2_F_x(Fx_face, X[I], 1./dx, 0.0, 0.5, 0.5)
    
    for I in ti.grouped(Fgrady_center):
        Fgrady_center[I] = interp_grad_2_F_y(Fy_face, X[I], 1./dx, 0.5, 0.0, 0.5)

    for I in ti.grouped(Fgradz_center):
        Fgradz_center[I] = interp_grad_2_F_z(Fz_face, X[I], 1./dx, 0.5, 0.5, 0.0)

@ti.kernel
def edge_noin_mask(edge_x_boundary_mask:ti.template(),edge_y_boundary_mask:ti.template(),edge_z_boundary_mask:ti.template()):
    edge_x_boundary_mask.fill(0)
    edge_y_boundary_mask.fill(0)
    edge_z_boundary_mask.fill(0)
    shape_x,shape_y,shape_z=edge_x_boundary_mask.shape
    for i,j,k in edge_x_boundary_mask:
        if(k == 0 or j == 0 or k==shape_z-1 or j==shape_y-1):
            edge_x_boundary_mask[i,j,k]=1
    shape_x,shape_y,shape_z=edge_y_boundary_mask.shape
    for i,j,k in edge_y_boundary_mask:
        if(i == 0 or k == 0 or i==shape_x-1 or k==shape_z-1):
            edge_y_boundary_mask[i,j,k]=1
    shape_x,shape_y,shape_z=edge_z_boundary_mask.shape
    for i,j,k in edge_z_boundary_mask:
        if(i == 0 or j == 0 or i==shape_x-1 or j==shape_y-1):
            edge_z_boundary_mask[i,j,k]=1


@ti.func
def choose_ax(i, I, fx, fy, fz):
    ret = 0.0
    if i==0:
        ret = fx[I]
    elif i==1:
        ret = fy[I]
    elif i==2:
        ret = fz[I]
    # print(ret)
    # if ret!=0 and ret!=-0.1:
    #     print(ret)
    return ret

@ti.kernel
def back_extend_field(a:ti.template(),b:ti.template()):
    for i,j,k in a:
        a[i,j,k]=b[i,j,k]
        
@ti.kernel
def extend_field(a:ti.template(),b:ti.template()):
    b.fill(0.0)
    for i,j,k in a:
        b[i,j,k]=a[i,j,k]


@ti.kernel
def minus_field(a: ti.template(), b: ti.template(), c: ti.template()):
    for i, j, k in c:
        c[i, j, k] = a[i,j,k]-b[i,j,k]


@ti.func
def is_valid(i,j,k,w):
    u_dim, v_dim, w_dim = w.shape
    res=True
    if(i<0 or j<0 or k<0 or i>=u_dim or j>=v_dim or k>= w_dim):
        res=False
    return res

@ti.kernel
def compute_dwx(tmp_w_x: ti.template(), dwx_dx: ti.template(), dwx_dy: ti.template(), dwx_dz: ti.template()):
    u_dim, v_dim, w_dim = dwx_dx.shape
    for i, j, k in dwx_dx:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwx_dx[i, j, k] = tmp_w_x[i, j, k] - tmp_w_x[i - 1, j, k]
    
    u_dim, v_dim, w_dim = dwx_dy.shape
    for i, j, k in dwx_dy:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwx_dy[i, j, k] = tmp_w_x[i, j+1, k] - tmp_w_x[i, j, k]
    
    u_dim, v_dim, w_dim = dwx_dz.shape
    for i, j, k in dwx_dz:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwx_dz[i, j, k] = tmp_w_x[i, j, k+1] - tmp_w_x[i, j, k]

@ti.kernel
def compute_divwx(div_wx: ti.template(), dwx_dx: ti.template(), dwx_dy: ti.template(), dwx_dz: ti.template()):
    u_dim, v_dim, w_dim = div_wx.shape
    for i, j, k in div_wx:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            div_wx[i, j, k] = dwx_dx[i+1, j, k] - dwx_dx[i, j, k] +\
                            dwx_dy[i, j, k] - dwx_dy[i, j-1, k] +\
                            dwx_dz[i, j, k] - dwx_dz[i, j, k-1]
    

@ti.kernel
def add_visc_x(tmp_w_x: ti.template(), div_wx: ti.template(), w_x: ti.template(), smoke: ti.template(), coe: ti.template()):
    u_dim, v_dim, w_dim = w_x.shape
    for i, j, k in w_x:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            w_x[i, j, k] = tmp_w_x[i, j, k] + coe * div_wx[i, j, k]

@ti.kernel
def update_init_w_x(change_w: ti.template(), init_w_x: ti.template(), phi_x_e: ti.template(), dx: ti.template()):
    u_dim, v_dim, w_dim = init_w_x.shape
    for i, j, k in init_w_x:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:    
            sum = 0.
            sum += 0.5 * interp_1(change_w, phi_x_e[i, j, k], dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            # 4 neighbors
            pos = interp_1(phi_x_e, ti.Vector([i+0.5-0.25, j-0.25, k-0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            pos = interp_1(phi_x_e, ti.Vector([i+0.5-0.25, j+0.25, k-0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            pos = interp_1(phi_x_e, ti.Vector([i+0.5+0.25, j-0.25, k-0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            pos = interp_1(phi_x_e, ti.Vector([i+0.5+0.25, j+0.25, k-0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)

            pos = interp_1(phi_x_e, ti.Vector([i+0.5-0.25, j-0.25, k+0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            pos = interp_1(phi_x_e, ti.Vector([i+0.5-0.25, j+0.25, k+0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            pos = interp_1(phi_x_e, ti.Vector([i+0.5+0.25, j-0.25, k+0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            pos = interp_1(phi_x_e, ti.Vector([i+0.5+0.25, j+0.25, k+0.25]) * dx, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.5, BL_y = 0.0, BL_z = 0.0)

            init_w_x[i, j, k] += sum


@ti.kernel
def compute_dwy(tmp_w_y: ti.template(), dwy_dx: ti.template(), dwy_dy: ti.template(), dwy_dz: ti.template()):
    u_dim, v_dim, w_dim = dwy_dx.shape
    for i, j, k in dwy_dx:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwy_dx[i, j, k] = tmp_w_y[i+1, j, k] - tmp_w_y[i, j, k]
    
    u_dim, v_dim, w_dim = dwy_dy.shape
    for i, j, k in dwy_dy:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwy_dy[i, j, k] = tmp_w_y[i, j, k] - tmp_w_y[i, j-1, k]
    
    u_dim, v_dim, w_dim = dwy_dz.shape
    for i, j, k in dwy_dz:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwy_dz[i, j, k] = tmp_w_y[i, j, k+1] - tmp_w_y[i, j, k]

@ti.kernel
def compute_divwy(div_wy: ti.template(), dwy_dx: ti.template(), dwy_dy: ti.template(), dwy_dz: ti.template()):
    u_dim, v_dim, w_dim = div_wy.shape
    for i, j, k in div_wy:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            div_wy[i, j, k] = dwy_dx[i, j, k] - dwy_dx[i-1, j, k] +\
                            dwy_dy[i, j+1, k] - dwy_dy[i, j, k] +\
                            dwy_dz[i, j, k] - dwy_dz[i, j, k-1]

@ti.kernel
def add_visc_y(tmp_w_y: ti.template(), div_wy: ti.template(), w_y: ti.template(), smoke: ti.template(), coe: ti.template()):
    u_dim, v_dim, w_dim = w_y.shape
    for i, j, k in w_y:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            w_y[i, j, k] = tmp_w_y[i, j, k] + coe * div_wy[i, j, k]

@ti.kernel
def update_init_w_y(change_w: ti.template(), init_w_y: ti.template(), phi_y_e: ti.template(), dx: ti.template()):
    u_dim, v_dim, w_dim = init_w_y.shape
    for i, j, k in init_w_y:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:    
            sum = 0.
            sum += 0.5 * interp_1(change_w, phi_y_e[i, j, k], dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            # 4 neighbors
            pos = interp_1(phi_y_e, ti.Vector([i-0.25, j+0.5-0.25, k-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            pos = interp_1(phi_y_e, ti.Vector([i-0.25, j+0.5+0.25, k-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            pos = interp_1(phi_y_e, ti.Vector([i+0.25, j+0.5-0.25, k-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            pos = interp_1(phi_y_e, ti.Vector([i+0.25, j+0.5+0.25, k-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)

            pos = interp_1(phi_y_e, ti.Vector([i-0.25, j+0.5-0.25, k+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            pos = interp_1(phi_y_e, ti.Vector([i-0.25, j+0.5+0.25, k+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            pos = interp_1(phi_y_e, ti.Vector([i+0.25, j+0.5-0.25, k+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            pos = interp_1(phi_y_e, ti.Vector([i+0.25, j+0.5+0.25, k+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.5, BL_z = 0.0)

            init_w_y[i, j, k] += sum


@ti.kernel
def compute_dwz(tmp_w_z: ti.template(), dwz_dx: ti.template(), dwz_dy: ti.template(), dwz_dz: ti.template()):
    u_dim, v_dim, w_dim = dwz_dx.shape
    for i, j, k in dwz_dx:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwz_dx[i, j, k] = tmp_w_z[i+1, j, k] - tmp_w_z[i, j, k]
    
    u_dim, v_dim, w_dim = dwz_dy.shape
    for i, j, k in dwz_dy:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwz_dy[i, j, k] = tmp_w_z[i, j+1, k] - tmp_w_z[i, j, k]
    
    u_dim, v_dim, w_dim = dwz_dz.shape
    for i, j, k in dwz_dz:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            dwz_dz[i, j, k] = tmp_w_z[i, j, k] - tmp_w_z[i, j, k-1]

@ti.kernel
def compute_divwz(div_wz: ti.template(), dwz_dx: ti.template(), dwz_dy: ti.template(), dwz_dz: ti.template()):
    u_dim, v_dim, w_dim = div_wz.shape
    for i, j, k in div_wz:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            div_wz[i, j, k] = dwz_dx[i, j, k] - dwz_dx[i-1, j, k] +\
                            dwz_dy[i, j, k] - dwz_dy[i, j-1, k] +\
                            dwz_dz[i, j, k+1] - dwz_dz[i, j, k]

@ti.kernel
def add_visc_z(tmp_w_z: ti.template(), div_wz: ti.template(), w_z: ti.template(), smoke: ti.template(), coe: ti.template()):
    u_dim, v_dim, w_dim = w_z.shape
    for i, j, k in w_z:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:
            w_z[i, j, k] = tmp_w_z[i, j, k] + coe * div_wz[i, j, k]

@ti.kernel
def update_init_w_z(change_w: ti.template(), init_w_z: ti.template(), phi_z_e: ti.template(), dx: ti.template()):
    u_dim, v_dim, w_dim = init_w_z.shape
    for i, j, k in init_w_z:
        if i > 0 and i < u_dim - 1 and j > 0 and j < v_dim - 1 and k > 0 and k< w_dim - 1:    
            sum = 0.
            sum += 0.5 * interp_1(change_w, phi_z_e[i, j, k], dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            # 4 neighbors
            pos = interp_1(phi_z_e, ti.Vector([i-0.25, j-0.25, k+0.5-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            pos = interp_1(phi_z_e, ti.Vector([i-0.25, j+0.25, k+0.5-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            pos = interp_1(phi_z_e, ti.Vector([i+0.25, j-0.25, k+0.5-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            pos = interp_1(phi_z_e, ti.Vector([i+0.25, j+0.25, k+0.5-0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)

            pos = interp_1(phi_z_e, ti.Vector([i-0.25, j-0.25, k+0.5+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            pos = interp_1(phi_z_e, ti.Vector([i-0.25, j+0.25, k+0.5+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            pos = interp_1(phi_z_e, ti.Vector([i+0.25, j-0.25, k+0.5+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            pos = interp_1(phi_z_e, ti.Vector([i+0.25, j+0.25, k+0.5+0.25]) * dx, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)
            sum += 0.0625 * interp_1(change_w, pos, dx, BL_x = 0.0, BL_y = 0.0, BL_z = 0.5)

            init_w_z[i, j, k] += sum
@ti.kernel
def copy_to(source: ti.template(), dest: ti.template()):
    u_dim, v_dim, w_dim = dest.shape
    for i, j, k in source:
        if i<=u_dim-1 and j<=v_dim-1 and k<=w_dim-1:
            dest[i, j, k] = source[i, j, k]
@ti.kernel
def copy_to2(source: ti.template(), dest: ti.template()):
    u_dim, v_dim = dest.shape
    for i, j in source:
        if i<=u_dim-1 and j<=v_dim-1:
            dest[i, j] = source[i, j ]

@ti.kernel
def copy_to1(source: ti.template(), dest: ti.template()):
    u_dim = dest.shape
    for i in source:
        if i<=u_dim[0]-1:
            dest[i] = source[i]
@ti.kernel
def scale_field(a: ti.template(), alpha: float, result: ti.template()):
    for I in ti.grouped(result):
        result[I] = alpha * a[I]

@ti.kernel
def add_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] + multiplier * f2[I]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.kernel
def center_coords_func(pf: ti.template(), dx: float):
    for I in ti.grouped(pf):
        pf[I] = (I+0.5) * dx

@ti.kernel
def x_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j + 0.5, k + 0.5]) * dx

@ti.kernel
def y_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j, k + 0.5]) * dx

@ti.kernel
def z_coords_func(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j + 0.5, k]) * dx

@ti.kernel
def x_coords_func_edge(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i + 0.5, j, k]) * dx

@ti.kernel
def y_coords_func_edge(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j + 0.5, k]) * dx

@ti.kernel
def z_coords_func_edge(pf: ti.template(), dx: float):
    for i, j, k in pf:
        pf[i, j, k] = ti.Vector([i, j, k + 0.5]) * dx

@ti.func
def sample(qf: ti.template(), u: float, v: float, w: float):
    u_dim, v_dim, w_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    k = ti.max(0, ti.min(int(w), w_dim-1))
    return qf[i, j, k]
@ti.func
def sample_2D(qf: ti.template(), u: float, v: float):
    u_dim, v_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    return qf[i, j]

# limiters

@ti.kernel
def nodal_limiter(u: ti.template(), err: ti.template(), u_res: ti.template()):
    for i, j, k in u:
        u_l = sample(u, i - 1, j, k)
        u_r = sample(u, i + 1, j, k)
        u_b = sample(u, i, j - 1, k)
        u_t = sample(u, i, j + 1, k)
        u_a = sample(u, i, j, k - 1)
        u_c = sample(u, i, j, k + 1)
        u_center = u[i, j, k]

        l_diff = u_l - u_center
        r_diff = u_r - u_center
        b_diff = u_b - u_center
        t_diff = u_t - u_center
        a_diff = u_a - u_center
        c_diff = u_c - u_center

        diff_sum = l_diff + r_diff + b_diff + t_diff + a_diff + c_diff
        diff_abs_sum = abs(l_diff) + abs(r_diff) + abs(b_diff) + abs(t_diff) + abs(a_diff) + abs(c_diff)

        alpha = 1. - (abs(diff_sum) / (diff_abs_sum + 5.e-15)) ** 2
        u_res[i, j, k] = u_center - alpha * err[i, j, k]


@ti.func
def minmod(a, b):
    c = 0.
    if a * b > 0:
        c += min(abs(a), abs(b)) * (a/abs(a)) # ensures correct sign
    return c

@ti.kernel
def TVD_limiter(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_l = sample(u1, i - 1, j, k)
        u1_r = sample(u1, i + 1, j, k)
        u1_b = sample(u1, i, j - 1, k)
        u1_t = sample(u1, i, j + 1, k)
        u1_a = sample(u1, i, j, k - 1)
        u1_c = sample(u1, i, j, k + 1)
        
        # compute limited slopes in each direction
        dx = minmod(u1_r - u1[i, j, k], u1[i, j, k] - u1_l)
        dy = minmod(u1_t - u1[i, j, k], u1[i, j, k] - u1_b)
        dz = minmod(u1_c - u1[i, j, k], u1[i, j, k] - u1_a)
        
        # reconstruct u with respect to u1 and write to u2
        u2[i, j, k] = u[i, j, k] - dx - dy - dz


# limit (modify) u with u1, write to u2
@ti.kernel
def BFECC_limiter(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_l = sample(u1, i - 1, j, k)
        u1_r = sample(u1, i + 1, j, k)
        u1_b = sample(u1, i, j - 1, k)
        u1_t = sample(u1, i, j + 1, k)
        u1_a = sample(u1, i, j, k - 1)
        u1_c = sample(u1, i, j, k + 1)
        maxi = ti.math.max(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        mini = ti.math.min(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        u2[i, j, k] = ti.math.clamp(u[i, j, k], mini, maxi)

@ti.kernel
def BFECC_limiter_w_x(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_1 = sample(u1, i, j - 1, k - 1)
        u1_2 = sample(u1, i, j - 1, k)
        u1_3 = sample(u1, i, j - 1, k + 1)

        u1_4 = sample(u1, i, j, k - 1)
        u1_6 = sample(u1, i, j, k + 1)

        u1_7 = sample(u1, i, j + 1, k - 1)
        u1_8 = sample(u1, i, j + 1, k)
        u1_9 = sample(u1, i, j + 1, k + 1)

        maxi = ti.math.max(u1_1, u1_2, u1_3, u1_4, u1_6, u1_7, u1_8, u1_9)
        mini = ti.math.min(u1_1, u1_2, u1_3, u1_4, u1_6, u1_7, u1_8, u1_9)
        u2[i, j, k] = ti.math.clamp(u[i, j, k], mini, maxi)

@ti.kernel
def BFECC_limiter_w_y(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_1 = sample(u1, i - 1, j, k - 1)
        u1_2 = sample(u1, i - 1, j, k)
        u1_3 = sample(u1, i - 1, j, k + 1)

        u1_4 = sample(u1, i, j, k - 1)
        u1_6 = sample(u1, i, j, k + 1)

        u1_7 = sample(u1, i + 1, j, k - 1)
        u1_8 = sample(u1, i + 1, j, k)
        u1_9 = sample(u1, i + 1, j, k + 1)

        maxi = ti.math.max(u1_1, u1_2, u1_3, u1_4, u1_6, u1_7, u1_8, u1_9)
        mini = ti.math.min(u1_1, u1_2, u1_3, u1_4, u1_6, u1_7, u1_8, u1_9)
        u2[i, j, k] = ti.math.clamp(u[i, j, k], mini, maxi)

@ti.kernel
def BFECC_limiter_w_z(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i, j, k in u:
        u1_1 = sample(u1, i - 1, j - 1, k)
        u1_2 = sample(u1, i - 1, j, k)
        u1_3 = sample(u1, i - 1, j + 1, k)

        u1_4 = sample(u1, i, j - 1, k)
        u1_6 = sample(u1, i, j + 1, k)

        u1_7 = sample(u1, i + 1, j - 1, k)
        u1_8 = sample(u1, i + 1, j, k)
        u1_9 = sample(u1, i + 1, j + 1, k)

        maxi = ti.math.max(u1_1, u1_2, u1_3, u1_4, u1_6, u1_7, u1_8, u1_9)
        mini = ti.math.min(u1_1, u1_2, u1_3, u1_4, u1_6, u1_7, u1_8, u1_9)
        u2[i, j, k] = ti.math.clamp(u[i, j, k], mini, maxi)

@ti.kernel
def zero_bc_tangential_v(solid_u_x:ti.template(), solid_u_y:ti.template(), solid_u_z:ti.template(),
                         u_x:ti.template(), u_y:ti.template(), u_z:ti.template(),
                         affected_by_solid_penalty_w_x:ti.template(),
                         affected_by_solid_penalty_w_y:ti.template(),
                         affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = u_x.shape
    for i, j, k in u_x:
        if not (i == 0 or i == udim - 1) and (j == 0 or j == vdim - 1 or k == 0 or k == wdim - 1):
            solid_u_x[i, j, k] = 0.75 * u_x[i, j, k]
            affected_by_solid_penalty_w_z[i, j+1, k] = 1
            affected_by_solid_penalty_w_z[i, j, k] = 1
            affected_by_solid_penalty_w_y[i, j, k+1] = 1
            affected_by_solid_penalty_w_y[i, j, k] = 1

    udim, vdim, wdim = u_y.shape
    for i, j, k in u_y:
        if not (j == 0 or j == vdim - 1) and (i == 0 or i == udim - 1 or k == 0 or k == wdim - 1):
            solid_u_y[i, j, k] = 0.75 * u_y[i, j, k]
            affected_by_solid_penalty_w_z[i, j, k] = 1
            affected_by_solid_penalty_w_z[i+1, j, k] = 1
            affected_by_solid_penalty_w_x[i, j, k+1] = 1
            affected_by_solid_penalty_w_x[i, j, k] = 1


    udim, vdim, wdim = u_z.shape
    for i, j, k in u_z:
        if not (k == 0 or k == wdim - 1) and (j == 0 or j == vdim - 1 or i == 0 or i == udim - 1):
            solid_u_z[i, j, k] = 0.75 * u_z[i, j, k]
            affected_by_solid_penalty_w_y[i, j, k] = 1
            affected_by_solid_penalty_w_y[i+1, j, k] = 1
            affected_by_solid_penalty_w_x[i, j, k] = 1
            affected_by_solid_penalty_w_x[i, j+1, k] = 1
        
@ti.func
def on4bound(i: int, j: int, k:int,  u_dim: int, v_dim: int, w_dim: int):
    ret = False
    if i <= 0 or j <= 0 or k <= 0 or i >= u_dim - 1 or j >= v_dim - 1 or k >= w_dim - 1:
        ret = True
    return ret

@ti.func
def inSolid_x(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    ret = False
    if not on4bound(i, j, k, udim, vdim, wdim):
        if center_bond[i, j, k] >= 1 and center_bond[i, j-1, k] >= 1 and center_bond[i, j, k-1] >= 1 and center_bond[i, j-1, k-1] >= 1:
            ret = True
    return ret

@ti.func
def inSolid_y(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    ret = False
    if not on4bound(i, j, k, udim, vdim, wdim):
        if center_bond[i, j, k] >= 1 and center_bond[i-1, j, k] >= 1 and center_bond[i, j, k-1] >= 1 and center_bond[i-1, j, k-1] >= 1:
            ret = True
    return ret
@ti.func
def inSolid_z(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    ret = False
    if not on4bound(i, j, k, udim, vdim, wdim):
        if center_bond[i, j, k] >= 1 and center_bond[i, j-1, k] >= 1 and center_bond[i-1, j, k] >= 1 and center_bond[i-1, j-1, k] >= 1:
            ret = True
    return ret

@ti.func
def onSolidSurf_x(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    ret = False
    if not on4bound(i, j, k, udim, vdim, wdim):
        if not inSolid_x(i, j, k, point_bond, center_bond) and point_bond[i, j, k] >= 1:
            ret = True
    return ret

@ti.func
def onSolidSurf_y(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    ret = False
    if not on4bound(i, j, k, udim, vdim, wdim):
        if not inSolid_y(i, j, k, point_bond, center_bond) and point_bond[i, j, k] >= 1:
            ret = True
    return ret

@ti.func
def onSolidSurf_z(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    ret = False
    if not on4bound(i, j, k, udim, vdim, wdim):
        if not inSolid_z(i, j, k, point_bond, center_bond) and point_bond[i, j, k] >= 1:
            ret = True
    return ret

@ti.func
def onSolidSurf_x_countnum(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    num = 0
    if not on4bound(i, j, k, udim, vdim, wdim):
        if not inSolid_x(i, j, k, point_bond, center_bond) and point_bond[i, j, k] >= 1:
            num = judge_inside_w_boud_x(i, j, k, center_bond)
    return num

@ti.func
def onSolidSurf_y_countnum(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    num = 0
    if not on4bound(i, j, k, udim, vdim, wdim):
        if not inSolid_y(i, j, k, point_bond, center_bond) and point_bond[i, j, k] >= 1:
            num = judge_inside_w_boud_y(i, j, k, center_bond)
    return num

@ti.func
def onSolidSurf_z_countnum(i: int, j: int, k:int, point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = point_bond.shape
    num = 0
    if not on4bound(i, j, k, udim, vdim, wdim):
        if not inSolid_z(i, j, k, point_bond, center_bond) and point_bond[i, j, k] >= 1:
            num = judge_inside_w_boud_z(i, j, k, center_bond)
    return num


@ti.kernel
def construct_solid_surf_x(surf_bond:ti.template(), point_bond:ti.template(), center_bond:ti.template()):
    for i, j, k in surf_bond:
        if onSolidSurf_x(i, j, k, point_bond, center_bond):
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_y(surf_bond:ti.template(), point_bond:ti.template(), center_bond:ti.template()):
    for i, j, k in surf_bond:
        if onSolidSurf_y(i, j, k, point_bond, center_bond):
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_z(surf_bond:ti.template(), point_bond:ti.template(), center_bond:ti.template()):
    for i, j, k in surf_bond:
        if onSolidSurf_z(i, j, k, point_bond, center_bond):
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0


@ti.kernel
def construct_solid_surf_x_includebd(surf_bond:ti.template(), point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if onSolidSurf_x(i, j, k, point_bond, center_bond):
            surf_bond[i, j, k] = 1
        elif k==0 or j==0 or k==wdim-1 or j==vdim-1:
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_y_includebd(surf_bond:ti.template(), point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if onSolidSurf_y(i, j, k, point_bond, center_bond):
            surf_bond[i, j, k] = 1
        elif k==0 or i==0 or k==wdim-1 or i==udim-1:
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_z_includebd(surf_bond:ti.template(), point_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if onSolidSurf_z(i, j, k, point_bond, center_bond):
            surf_bond[i, j, k] = 1
        elif i==0 or j==0 or j==vdim-1 or i==udim-1:
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0


@ti.kernel
def construct_solid_surf_face_x(surf_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if not on4bound(i, j, k, udim, vdim, wdim) and (center_bond[i, j, k] + center_bond[i-1, j, k] == 1):
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_face_y(surf_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if not on4bound(i, j, k, udim, vdim, wdim) and (center_bond[i, j, k] + center_bond[i, j-1, k] == 1):
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_face_z(surf_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if not on4bound(i, j, k, udim, vdim, wdim) and (center_bond[i, j, k] + center_bond[i, j, k-1] == 1):
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_face_x_includebd(surf_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if not on4bound(i, j, k, udim, vdim, wdim) and (center_bond[i, j, k] + center_bond[i-1, j, k] == 1):
            surf_bond[i, j, k] = 1
        elif i==0 or i==udim-1:
            surf_bond[i,j,k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_face_y_includebd(surf_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if not on4bound(i, j, k, udim, vdim, wdim) and (center_bond[i, j, k] + center_bond[i, j-1, k] == 1):
            surf_bond[i, j, k] = 1
        elif j==0 or j==vdim-1:
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0

@ti.kernel
def construct_solid_surf_face_z_includebd(surf_bond:ti.template(), center_bond:ti.template()):
    udim, vdim, wdim = surf_bond.shape
    for i, j, k in surf_bond:
        if not on4bound(i, j, k, udim, vdim, wdim) and (center_bond[i, j, k] + center_bond[i, j, k-1] == 1):
            surf_bond[i, j, k] = 1
        elif k==0 or k==wdim-1:
            surf_bond[i, j, k] = 1
        else:
            surf_bond[i, j, k] = 0


@ti.kernel
def clear_insolid_omega(w_x:ti.template(), w_y:ti.template(), w_z:ti.template(), 
                        edge_x_boundary_mask:ti.template(), edge_y_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template()):
    udim, vdim, wdim = w_x.shape
    for i,j,k in w_x:
        if not on4bound(i, j, k, udim, vdim, wdim):
            if edge_x_boundary_mask[i, j, k] >= 1:
                w_x[i, j, k] = 0

    udim, vdim, wdim = w_y.shape
    for i,j,k in w_y:
        if not on4bound(i, j, k, udim, vdim, wdim):
            if edge_y_boundary_mask[i, j, k] >= 1:
                w_y[i, j, k] = 0

    udim, vdim, wdim = w_z.shape
    for i,j,k in w_z:
        if not on4bound(i, j, k, udim, vdim, wdim):
            if edge_z_boundary_mask[i, j, k] >= 1:
                w_z[i, j, k] = 0

@ti.kernel
def compute_influid_omega(u_x:ti.template(), u_y:ti.template(), u_z:ti.template(), w_x:ti.template(), w_y:ti.template(), w_z:ti.template(), \
                          point_bond_x:ti.template(), point_bond_y:ti.template(), point_bond_z:ti.template(), inv_dx: float):
    for i, j, k in w_x:
        if point_bond_x[i, j, k] <= 0:
            vt = sample(u_z, i, j, k)
            vb = sample(u_z, i, j - 1, k)
            vc = sample(u_y, i, j, k)
            va = sample(u_y, i, j, k - 1)

            w_x[i, j, k] = ((vt - vb) - (vc - va)) * inv_dx
    for i, j, k in w_y:
        if point_bond_y[i, j, k] <= 0:
            vc = sample(u_x, i, j, k)
            va = sample(u_x, i, j, k - 1)
            vr = sample(u_z, i, j, k)
            vl = sample(u_z, i - 1, j, k)

            w_y[i, j, k] = ((vc - va) - (vr - vl)) * inv_dx
    for i, j, k in w_z:
        if point_bond_z[i, j, k] <= 0:
            vr = sample(u_y, i, j, k)
            vl = sample(u_y, i - 1, j, k)
            vt = sample(u_x, i, j, k)
            vb = sample(u_x, i, j - 1, k)

            w_z[i, j, k] = ((vr - vl) - (vt - vb)) * inv_dx

@ti.kernel
def enlarge_center_bond(cb:ti.template(), cb_l:ti.template()):
    udim, vdim, wdim = cb_l.shape
    for i, j, k in cb_l:
        if not on4bound(i, j, k, udim, vdim, wdim):
            if cb[i-1, j, k] >= 1 or cb[i+1, j, k] >= 1 or cb[i, j-1, k] >= 1 or cb[i, j+1, k] >= 1 or cb[i, j, k-1] >= 1 or cb[i, j, k+1] >= 1 or cb[i, j, k] >= 1:
                cb_l[i, j, k] = 1
            else:
                cb_l[i, j, k] = 0
        else:
            cb_l[i, j, k] = 0

@ti.kernel
def enforce_surface_vel(center_boundary_mask:ti.template(), center_boundary_vel:ti.template(), 
                        bv_x:ti.template(), bv_y:ti.template(), bv_z:ti.template(),
                        wall_u:ti.f32, wall_v:ti.f32, wall_w:ti.f32):
    for i,j,k in center_boundary_mask:
        if center_boundary_mask[i,j,k] >= 1:
            bv_x[i,j,k] = center_boundary_vel[i,j,k][0]
            bv_x[i+1,j,k] = center_boundary_vel[i,j,k][0]
            bv_y[i,j,k] = center_boundary_vel[i,j,k][1]
            bv_y[i,j+1,k] = center_boundary_vel[i,j,k][1]
            bv_z[i,j,k] = center_boundary_vel[i,j,k][2]
            bv_z[i,j,k+1] = center_boundary_vel[i,j,k][2]
    
    udim, vdim, wdim = bv_x.shape
    for i,j,k in bv_x:
        if i==0 or i==udim-1:
            bv_x[i,j,k] = wall_u
    
    udim, vdim, wdim = bv_y.shape
    for i,j,k in bv_y:
        if j==0 or j==vdim-1:
            bv_y[i,j,k] = wall_v
    
    udim, vdim, wdim = bv_z.shape
    for i,j,k in bv_z:
        if k==0 or k==wdim-1:
            bv_z[i,j,k] = wall_w

@ti.kernel
def combine3components(bv3:ti.template(), bv_x:ti.template(), bv_y:ti.template(),
                  bv_z:ti.template()):
    udim_x, vdim_x, wdim_x = bv_x.shape
    udim_y, vdim_y, wdim_y = bv_y.shape
    udim_z, vdim_z, wdim_z = bv_z.shape
    for i,j,k,w in bv3:
        if w == 0:
            if 0<=i<=udim_x-1 and 0<=j<=vdim_x-1 and 0<=k<=wdim_x-1:
                bv3[i,j,k,w] = bv_x[i,j,k]
            else:
                bv3[i,j,k,w] = 0
        elif w == 1:
            if 0<=i<=udim_y-1 and 0<=j<=vdim_y-1 and 0<=k<=wdim_y-1:
                bv3[i,j,k,w] = bv_y[i,j,k]
            else:
                bv3[i,j,k,w] = 0
        elif w == 2:
            if 0<=i<=udim_z-1 and 0<=j<=vdim_z-1 and 0<=k<=wdim_z-1:
                bv3[i,j,k,w] = bv_z[i,j,k]
            else:
                bv3[i,j,k,w] = 0

@ti.kernel
def zero_noslip_ux_leavenothrough(solid_u_x:ti.template(), u_x:ti.template(), 
                                  edge_y_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template(), 
                                  face_boundary_mask_x:ti.template(), center_boundary_mask:ti.template(), 
                                  affected_by_solid_penalty_w_y:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_x.shape
    for i, j, k in solid_u_x:
        if face_boundary_mask_x[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            if onSolidSurf_z(i, j+1, k, edge_z_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_z(i, j, k, edge_z_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_y(i, j, k+1, edge_y_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_y(i, j, k, edge_y_boundary_mask, center_boundary_mask):
                num += 1
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j+1, k] = 1
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_y[i, j, k+1] = 1
                affected_by_solid_penalty_w_y[i, j, k] = 1
            solid_u_x[i, j, k] = u_x[i, j, k] - (num / 4.0) * u_x[i, j, k]
                
@ti.kernel
def zero_noslip_uy_leavenothrough(solid_u_y:ti.template(), u_y:ti.template(), 
                                  edge_x_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template(), 
                                  face_boundary_mask_y:ti.template(), center_boundary_mask:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_y.shape
    for i, j, k in solid_u_y:
        if face_boundary_mask_y[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            if onSolidSurf_z(i, j, k, edge_z_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_z(i+1, j, k, edge_z_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_x(i, j, k+1, edge_x_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_x(i, j, k, edge_x_boundary_mask, center_boundary_mask):
                num += 1
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_z[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k+1] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
            solid_u_y[i, j, k] = u_y[i, j, k] - (num / 4.0) * u_y[i, j, k]

@ti.kernel
def zero_noslip_uz_leavenothrough(solid_u_z:ti.template(), u_z:ti.template(), 
                                  edge_x_boundary_mask:ti.template(), edge_y_boundary_mask:ti.template(), 
                                  face_boundary_mask_z:ti.template(), center_boundary_mask:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_y:ti.template()):
    udim, vdim, wdim = solid_u_z.shape
    for i, j, k in solid_u_z:
        if face_boundary_mask_z[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            if onSolidSurf_y(i, j, k, edge_y_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_y(i+1, j, k, edge_y_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_x(i, j, k, edge_x_boundary_mask, center_boundary_mask):
                num += 1
            elif onSolidSurf_x(i, j+1, k, edge_x_boundary_mask, center_boundary_mask):
                num += 1
            if num>= 1:
                affected_by_solid_penalty_w_y[i, j, k] = 1
                affected_by_solid_penalty_w_y[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
                affected_by_solid_penalty_w_x[i, j+1, k] = 1
            solid_u_z[i, j, k] = u_z[i, j, k] - (num / 4.0) * u_z[i, j, k]

@ti.kernel
def zero_noslip_ux_leavenothrough_countcenter(solid_u_x:ti.template(), u_x:ti.template(), 
                                  edge_y_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template(), 
                                  face_boundary_mask_x:ti.template(), center_boundary_mask:ti.template(), 
                                  affected_by_solid_penalty_w_y:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_x.shape
    for i, j, k in solid_u_x:
        if face_boundary_mask_x[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            num += onSolidSurf_z_countnum(i, j+1, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_z_countnum(i, j, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_y_countnum(i, j, k+1, edge_y_boundary_mask, center_boundary_mask)
            num += onSolidSurf_y_countnum(i, j, k, edge_y_boundary_mask, center_boundary_mask)
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j+1, k] = 1
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_y[i, j, k+1] = 1
                affected_by_solid_penalty_w_y[i, j, k] = 1
            solid_u_x[i, j, k] = u_x[i, j, k] - (num / 8.0) * u_x[i, j, k]
                
@ti.kernel
def zero_noslip_uy_leavenothrough_countcenter(solid_u_y:ti.template(), u_y:ti.template(), 
                                  edge_x_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template(), 
                                  face_boundary_mask_y:ti.template(), center_boundary_mask:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_y.shape
    for i, j, k in solid_u_y:
        if face_boundary_mask_y[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            num += onSolidSurf_z_countnum(i, j, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_z_countnum(i+1, j, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j, k+1, edge_x_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j, k, edge_x_boundary_mask, center_boundary_mask)
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_z[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k+1] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
            solid_u_y[i, j, k] = u_y[i, j, k] - (num / 8.0) * u_y[i, j, k]

@ti.kernel
def zero_noslip_uz_leavenothrough_countcenter(solid_u_z:ti.template(), u_z:ti.template(), 
                                  edge_x_boundary_mask:ti.template(), edge_y_boundary_mask:ti.template(), 
                                  face_boundary_mask_z:ti.template(), center_boundary_mask:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_y:ti.template()):
    udim, vdim, wdim = solid_u_z.shape
    for i, j, k in solid_u_z:
        if face_boundary_mask_z[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            num += onSolidSurf_y_countnum(i, j, k, edge_y_boundary_mask, center_boundary_mask)
            num += onSolidSurf_y_countnum(i+1, j, k, edge_y_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j, k, edge_x_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j+1, k, edge_x_boundary_mask, center_boundary_mask)
            if num>= 1:
                affected_by_solid_penalty_w_y[i, j, k] = 1
                affected_by_solid_penalty_w_y[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
                affected_by_solid_penalty_w_x[i, j+1, k] = 1
            solid_u_z[i, j, k] = u_z[i, j, k] - (num / 8.0) * u_z[i, j, k]

@ti.kernel
def movingsolid_noslip_ux_leavenothrough_countcenter(solid_u_x:ti.template(), u_x:ti.template(), 
                                  edge_y_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template(), 
                                  face_boundary_mask_x:ti.template(), center_boundary_mask:ti.template(), bv_x:ti.template(),
                                  affected_by_solid_penalty_w_y:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_x.shape
    for i, j, k in solid_u_x:
        if face_boundary_mask_x[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            num += onSolidSurf_z_countnum(i, j+1, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_z_countnum(i, j, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_y_countnum(i, j, k+1, edge_y_boundary_mask, center_boundary_mask)
            num += onSolidSurf_y_countnum(i, j, k, edge_y_boundary_mask, center_boundary_mask)
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j+1, k] = 1
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_y[i, j, k+1] = 1
                affected_by_solid_penalty_w_y[i, j, k] = 1
            solid_u_x[i, j, k] = u_x[i, j, k] + (num / 8.0) * (bv_x[i,j,k] - u_x[i, j, k])
                
@ti.kernel
def movingsolid_noslip_uy_leavenothrough_countcenter(solid_u_y:ti.template(), u_y:ti.template(), 
                                  edge_x_boundary_mask:ti.template(), edge_z_boundary_mask:ti.template(), 
                                  face_boundary_mask_y:ti.template(), center_boundary_mask:ti.template(),bv_y:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_y.shape
    for i, j, k in solid_u_y:
        if face_boundary_mask_y[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            num += onSolidSurf_z_countnum(i, j, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_z_countnum(i+1, j, k, edge_z_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j, k+1, edge_x_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j, k, edge_x_boundary_mask, center_boundary_mask)
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_z[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k+1] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
            solid_u_y[i, j, k] = u_y[i, j, k] + (num / 8.0) * (bv_y[i,j,k] - u_y[i, j, k])

@ti.kernel
def movingsolid_noslip_uz_leavenothrough_countcenter(solid_u_z:ti.template(), u_z:ti.template(), 
                                  edge_x_boundary_mask:ti.template(), edge_y_boundary_mask:ti.template(), 
                                  face_boundary_mask_z:ti.template(), center_boundary_mask:ti.template(),bv_z:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_y:ti.template()):
    udim, vdim, wdim = solid_u_z.shape
    for i, j, k in solid_u_z:
        if face_boundary_mask_z[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            num += onSolidSurf_y_countnum(i, j, k, edge_y_boundary_mask, center_boundary_mask)
            num += onSolidSurf_y_countnum(i+1, j, k, edge_y_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j, k, edge_x_boundary_mask, center_boundary_mask)
            num += onSolidSurf_x_countnum(i, j+1, k, edge_x_boundary_mask, center_boundary_mask)
            if num>= 1:
                affected_by_solid_penalty_w_y[i, j, k] = 1
                affected_by_solid_penalty_w_y[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
                affected_by_solid_penalty_w_x[i, j+1, k] = 1
            solid_u_z[i, j, k] = u_z[i, j, k] + (num / 8.0) * (bv_z[i,j,k] - u_z[i, j, k])


@ti.kernel
def movingsolid_noslip_ux_exam8adjecent(solid_u_x:ti.template(), u_x:ti.template(), 
                                  face_boundary_mask_x:ti.template(),   center_boundary_mask:ti.template(), bv_tan:ti.template(),
                                  affected_by_solid_penalty_w_y:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_x.shape
    for i, j, k in solid_u_x:
        if face_boundary_mask_x[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            around_vel = 0.0
            this_vel = u_x[i,j,k]
            if center_boundary_mask[i-1, j-1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i-1,j-1,k][0]-this_vel)
            if center_boundary_mask[i-1, j+1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j+1, k][0]-this_vel)
            if center_boundary_mask[i, j-1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i, j-1, k][0]-this_vel)
            if center_boundary_mask[i, j+1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i, j+1, k][0]-this_vel)
            if center_boundary_mask[i-1, j, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j, k-1][0]-this_vel)
            if center_boundary_mask[i-1, j, k+1] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j, k+1][0]-this_vel)
            if center_boundary_mask[i, j, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j, k-1][0]-this_vel)
            if center_boundary_mask[i, j, k+1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j, k+1][0]-this_vel)
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j+1, k] = 1
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_y[i, j, k+1] = 1
                affected_by_solid_penalty_w_y[i, j, k] = 1
            solid_u_x[i, j, k] = this_vel + (num / 8.0) * (around_vel)


@ti.kernel
def movingsolid_noslip_uy_exam8adjecent(solid_u_y:ti.template(), u_y:ti.template(), 
                                  face_boundary_mask_y:ti.template(), center_boundary_mask:ti.template(),bv_tan:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_z:ti.template()):
    udim, vdim, wdim = solid_u_y.shape
    for i, j, k in solid_u_y:
        if face_boundary_mask_y[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            around_vel = 0.0
            this_vel = u_y[i,j,k]
            if center_boundary_mask[i-1, j-1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j-1, k][1] - this_vel)
            if center_boundary_mask[i+1, j-1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i+1, j-1, k][1] - this_vel)
            if center_boundary_mask[i-1, j, k] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j, k][1] - this_vel)
            if center_boundary_mask[i+1, j, k] >= 1:
                num += 1
                around_vel += (bv_tan[i+1, j, k][1] - this_vel)
            if center_boundary_mask[i, j-1, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j-1, k-1][1] - this_vel)
            if center_boundary_mask[i, j-1, k+1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j-1, k+1][1] - this_vel)
            if center_boundary_mask[i, j, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j, k-1][1] - this_vel)
            if center_boundary_mask[i, j, k+1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j, k+1][1] - this_vel)
            if num>= 1:
                affected_by_solid_penalty_w_z[i, j, k] = 1
                affected_by_solid_penalty_w_z[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k+1] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
            solid_u_y[i, j, k] = this_vel + (num / 8.0) * around_vel


@ti.kernel
def movingsolid_noslip_uz_exam8adjecent(solid_u_z:ti.template(), u_z:ti.template(), 
                                  face_boundary_mask_z:ti.template(), center_boundary_mask:ti.template(),bv_tan:ti.template(),
                                  affected_by_solid_penalty_w_x:ti.template(), affected_by_solid_penalty_w_y:ti.template()):
    udim, vdim, wdim = solid_u_z.shape
    for i, j, k in solid_u_z:
        if face_boundary_mask_z[i, j, k] <= 0 and not on4bound(i, j, k, udim, vdim, wdim):
            num = 0
            around_vel = 0.0
            this_vel = u_z[i,j,k]
            if center_boundary_mask[i-1, j, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j, k-1][2] - this_vel)
            if center_boundary_mask[i+1, j, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i+1, j, k-1][2] - this_vel)
            if center_boundary_mask[i, j-1, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j-1, k-1][2] - this_vel)
            if center_boundary_mask[i, j+1, k-1] >= 1:
                num += 1
                around_vel += (bv_tan[i, j+1, k-1][2] - this_vel)
            if center_boundary_mask[i-1, j, k] >= 1:
                num += 1
                around_vel += (bv_tan[i-1, j, k][2] - this_vel)
            if center_boundary_mask[i+1, j, k] >= 1:
                num += 1
                around_vel += (bv_tan[i+1, j, k][2] - this_vel)
            if center_boundary_mask[i, j-1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i, j-1, k][2] - this_vel)
            if center_boundary_mask[i, j+1, k] >= 1:
                num += 1
                around_vel += (bv_tan[i, j+1, k][2] - this_vel)
            if num>= 1:
                affected_by_solid_penalty_w_y[i, j, k] = 1
                affected_by_solid_penalty_w_y[i+1, j, k] = 1
                affected_by_solid_penalty_w_x[i, j, k] = 1
                affected_by_solid_penalty_w_x[i, j+1, k] = 1
            solid_u_z[i, j, k] = this_vel + (num / 8.0) * around_vel


@ti.kernel
def mtply_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] * f2[I] * multiplier
@ti.kernel
def mtply_reversemask(f1: ti.template(), mask: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] * (1 - mask[I]) * multiplier

@ti.kernel
def reverse_mask(mask:ti.template(), reversed_mask:ti.template()):
    for I in ti.grouped(mask):
        reversed_mask[I] = 1 - mask[I]
@ti.kernel
def curl(vf: ti.template(), cf: ti.template(), inv_dx: float):
    inv_dist = 0.5 * inv_dx
    for i, j, k in cf:
        vr = sample(vf, i+1, j, k)
        vl = sample(vf, i-1, j, k)
        vt = sample(vf, i, j+1, k)
        vb = sample(vf, i, j-1, k)
        vc = sample(vf, i, j, k+1)
        va = sample(vf, i, j, k-1)

        d_vx_dz = inv_dist * (vc.x - va.x)
        d_vx_dy = inv_dist * (vt.x - vb.x)
        
        d_vy_dx = inv_dist * (vr.y - vl.y)
        d_vy_dz = inv_dist * (vc.y - va.y)

        d_vz_dx = inv_dist * (vr.z - vl.z)
        d_vz_dy = inv_dist * (vt.z - vb.z)

        cf[i,j,k][0] = d_vz_dy - d_vy_dz
        cf[i,j,k][1] = d_vx_dz - d_vz_dx
        cf[i,j,k][2] = d_vy_dx - d_vx_dy

@ti.kernel
def curl_f2e_x(u_z: ti.template(), u_y: ti.template(), w_x: ti.template(), inv_dist: float):
    for i, j, k in w_x:
        vt = sample(u_z, i, j, k)
        vb = sample(u_z, i, j - 1, k)
        vc = sample(u_y, i, j, k)
        va = sample(u_y, i, j, k - 1)

        w_x[i, j, k] = ((vt - vb) - (vc - va)) * inv_dist

@ti.kernel
def curl_f2e_y(u_x: ti.template(), u_z: ti.template(), w_y: ti.template(), inv_dist: float):
    for i, j, k in w_y:
        vc = sample(u_x, i, j, k)
        va = sample(u_x, i, j, k - 1)
        vr = sample(u_z, i, j, k)
        vl = sample(u_z, i - 1, j, k)

        w_y[i, j, k] = ((vc - va) - (vr - vl)) * inv_dist

@ti.kernel
def curl_f2e_z(u_y: ti.template(), u_x: ti.template(), w_z: ti.template(), inv_dist: float):
    for i, j, k in w_z:
        vr = sample(u_y, i, j, k)
        vl = sample(u_y, i - 1, j, k)
        vt = sample(u_x, i, j, k)
        vb = sample(u_x, i, j - 1, k)

        w_z[i, j, k] = ((vr - vl) - (vt - vb)) * inv_dist
@ti.kernel
def stream2velocity(u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), stream_x: ti.template(), stream_y: ti.template(), stream_z: ti.template(), dx: float):
    u_dim, v_dim, w_dim = stream_x.shape #(255, 256, 256)
    for i, j, k in u_x: #(256, 255, 255)
        vt = sample(stream_z, i, j + 1, k)
        vb = sample(stream_z, i, j, k)
        vc = sample(stream_y, i, j, k + 1)
        va = sample(stream_y, i, j, k)
        u_x[i, j, k] = ((vt - vb) - (vc - va)) / dx

    u_dim, v_dim, w_dim = stream_y.shape
    for i, j, k in u_y:
        vc = sample(stream_x, i, j, k + 1)
        va = sample(stream_x, i, j, k)
        vr = sample(stream_z, i + 1, j, k)
        vl = sample(stream_z, i, j, k)
        u_y[i, j, k] = ((vc - va) - (vr - vl)) / dx

    u_dim, v_dim, w_dim = stream_z.shape #(256, 256, 255)
    for i, j, k in u_z: #(255, 255, 256), i range from 0 to 254
        vr = sample(stream_y, i + 1, j, k)
        vl = sample(stream_y, i, j, k)
        vt = sample(stream_x, i, j + 1, k)
        vb = sample(stream_x, i, j, k)
        u_z[i, j, k] = ((vr - vl) - (vt - vb)) / dx

@ti.kernel
def stream2velocity_movingsolid(center_boundary_mask:ti.template(), boundary_vel:ti.template(), u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), 
                                stream_x: ti.template(), stream_y: ti.template(), stream_z: ti.template(), dx: float):
    u_dim, v_dim, w_dim = stream_x.shape #(255, 256, 256)
    for i, j, k in u_x: #(256, 255, 255)
        vt = sample(stream_z, i, j + 1, k)
        vb = sample(stream_z, i, j, k)
        vc = sample(stream_y, i, j, k + 1)
        va = sample(stream_y, i, j, k)
        if on4bound(i, j, k, u_dim, v_dim, w_dim):
            u_x[i, j, k] = ((vt - vb) - (vc - va)) / dx
        elif center_boundary_mask[i,j,k] >= 1 and center_boundary_mask[i-1,j,k]>=1:
            u_x[i,j,k] = boundary_vel[i,j,k][0]
        else:
            u_x[i, j, k] = ((vt - vb) - (vc - va)) / dx

    u_dim, v_dim, w_dim = stream_y.shape
    for i, j, k in u_y:
        vc = sample(stream_x, i, j, k + 1)
        va = sample(stream_x, i, j, k)
        vr = sample(stream_z, i + 1, j, k)
        vl = sample(stream_z, i, j, k)
        if on4bound(i, j, k, u_dim, v_dim, w_dim):
            u_y[i, j, k] = ((vc - va) - (vr - vl)) / dx
        elif center_boundary_mask[i,j,k] >= 1 and center_boundary_mask[i,j-1,k]>=1:
            u_y[i,j,k] = boundary_vel[i,j,k][1]
        else:
            u_y[i, j, k] = ((vc - va) - (vr - vl)) / dx

    u_dim, v_dim, w_dim = stream_z.shape #(256, 256, 255)
    for i, j, k in u_z: #(255, 255, 256), i range from 0 to 254
        vr = sample(stream_y, i + 1, j, k)
        vl = sample(stream_y, i, j, k)
        vt = sample(stream_x, i, j + 1, k)
        vb = sample(stream_x, i, j, k)
        if on4bound(i, j, k, u_dim, v_dim, w_dim):
            u_z[i, j, k] = ((vr - vl) - (vt - vb)) / dx
        elif center_boundary_mask[i,j,k] >= 1 and center_boundary_mask[i,j,k-1]>=1:
            u_z[i, j, k] = boundary_vel[i,j,k][2]
        else:
            u_z[i, j, k] = ((vr - vl) - (vt - vb)) / dx

@ti.kernel
def stream2velocity_0(u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), stream_x: ti.template(), stream_y: ti.template(), stream_z: ti.template(), dx: float):
    u_dim, v_dim, w_dim = stream_x.shape #(255, 256, 256)
    for i, j, k in u_x: #(256, 255, 255)
        vt = sample(stream_z, i, j + 1, k)
        vb = sample(stream_z, i, j, k)
        vc = sample(stream_y, i, j, k + 1)
        va = sample(stream_y, i, j, k)

        if j <= 0:
            vb = 0
        if j+2>=v_dim:
            vt = 0
        if k<=0:
            va = 0
        if k+2>=w_dim:
            vc = 0

        u_x[i, j, k] = ((vt - vb) - (vc - va)) / dx

    u_dim, v_dim, w_dim = stream_y.shape
    for i, j, k in u_y:
        vc = sample(stream_x, i, j, k + 1)
        va = sample(stream_x, i, j, k)
        vr = sample(stream_z, i + 1, j, k)
        vl = sample(stream_z, i, j, k)

        if i <= 0:
            vl = 0
        if i+2>=u_dim:
            vr = 0
        if k<=0:
            va = 0
        if k+2>=w_dim:
            vc = 0

        u_y[i, j, k] = ((vc - va) - (vr - vl)) / dx

    u_dim, v_dim, w_dim = stream_z.shape #(256, 256, 255)
    for i, j, k in u_z: #(255, 255, 256), i range from 0 to 254
        vr = sample(stream_y, i + 1, j, k)
        vl = sample(stream_y, i, j, k)
        vt = sample(stream_x, i, j + 1, k)
        vb = sample(stream_x, i, j, k)

        if j <= 0:
            vb = 0
        if j+2>=v_dim:
            vt = 0
        if i <= 0:
            vl = 0
        if i+2>=u_dim:
            vr = 0

        u_z[i, j, k] = ((vr - vl) - (vt - vb)) / dx

@ti.kernel
def get_central_vector_from_edge_fields(wx: ti.template(), wy: ti.template(), wz: ti.template(), wc: ti.template()):
    for i, j, k in wc:
        wc_x = 0.25 * (
            wx[i, j, k] + wx[i, j+1, k] + wx[i, j, k+1] + wx[i, j+1, k+1]
        )
        wc_y = 0.25 * (
            wy[i, j, k] + wy[i+1, j, k] + wy[i, j, k+1] + wy[i+1, j, k+1]
        )
        wc_z = 0.25 * (
            wz[i, j, k] + wz[i+1, j, k] + wz[i, j+1, k] + wz[i+1, j+1, k]
        )
        wc[i, j, k] = ti.Vector([wc_x, wc_y, wc_z])

@ti.kernel
def split_central_vector_to_edge_fields(wc: ti.template(), wx: ti.template(), wy: ti.template(), wz: ti.template()):
    for i, j, k in wx:
        c1 = sample(wc, i, j - 1, k - 1)
        c2 = sample(wc, i, j, k - 1)
        c3 = sample(wc, i, j - 1, k)
        c4 = sample(wc, i, j, k)
        wx[i, j, k] = 0.25 * (c1.x + c2.x + c3.x + c4.x)
    for i, j, k in wy:
        c1 = sample(wc, i - 1, j, k - 1)
        c2 = sample(wc, i, j, k - 1)
        c3 = sample(wc, i - 1, j, k)
        c4 = sample(wc, i, j, k)
        wy[i, j, k] = 0.25 * (c1.y + c2.y + c3.y + c4.y)
    for i, j, k in wz:
        c1 = sample(wc, i - 1, j - 1, k)
        c2 = sample(wc, i, j - 1, k)
        c3 = sample(wc, i - 1, j, k)
        c4 = sample(wc, i, j, k)
        wz[i, j, k] = 0.25 * (c1.z + c2.z + c3.z + c4.z)


@ti.kernel
def get_central_vector(vx: ti.template(), vy: ti.template(), vz: ti.template(), vc: ti.template()):
    for i, j, k in vc:
        vc[i,j,k].x = 0.5 * (vx[i+1, j, k] + vx[i, j, k])
        vc[i,j,k].y = 0.5 * (vy[i, j+1, k] + vy[i, j, k])
        vc[i,j,k].z = 0.5 * (vz[i, j, k+1] + vz[i, j, k])

@ti.kernel
def split_central_vector(vc: ti.template(), vx: ti.template(), vy: ti.template(), vz: ti.template()):
    for i, j, k in vx:
        r = sample(vc, i, j, k)
        l = sample(vc, i-1, j, k)
        vx[i,j,k] = 0.5 * (r.x + l.x)
    for i, j, k in vy:
        t = sample(vc, i, j, k)
        b = sample(vc, i, j-1, k)
        vy[i,j,k] = 0.5 * (t.y + b.y)
    for i, j, k in vz:
        c = sample(vc, i, j, k)
        a = sample(vc, i, j, k-1)
        vz[i,j,k] = 0.5 * (c.z + a.z)

@ti.kernel
def sizing_function(u: ti.template(), sizing: ti.template(), dx: float):
    u_dim, v_dim, w_dim = u.shape
    for i, j, k in u:
        u_l = sample(u, i-1, j, k)
        u_r = sample(u, i+1, j, k)
        u_t = sample(u, i, j+1, k)
        u_b = sample(u, i, j-1, k)
        u_c = sample(u, i, j, k+1)
        u_a = sample(u, i, j, k-1)
        partial_x = 1./(2*dx) * (u_r - u_l)
        partial_y = 1./(2*dx) * (u_t - u_b)
        partial_z = 1./(2*dx) * (u_c - u_a)
        if i == 0:
            partial_x = 1./(2*dx) * (u_r - 0)
        elif i == u_dim - 1:
            partial_x = 1./(2*dx) * (0 - u_l)
        if j == 0:
            partial_y = 1./(2*dx) * (u_t - 0)
        elif j == v_dim - 1:
            partial_y = 1./(2*dx) * (0 - u_b)
        if k == 0:
            partial_z = 1./(2*dx) * (u_c - 0)
        elif k == w_dim - 1:
            partial_z = 1./(2*dx) * (0 - u_a)

        sizing[i, j, k] = ti.sqrt(partial_x.x ** 2 + partial_x.y ** 2 + partial_x.z ** 2\
                            + partial_y.x ** 2 + partial_y.y ** 2 + partial_y.z ** 2\
                            + partial_z.x ** 2 + partial_z.y ** 2 + partial_z.z ** 2)

@ti.kernel
def diffuse_grid(value: ti.template(), tmp: ti.template()):
    for I in ti.grouped(value):
        value[I] = ti.abs(value[I])
    for i, j, k in tmp:
        tmp[i,j,k] = 1./6 * (sample(value, i+1,j,k) + sample(value, i-1,j,k)\
                + sample(value, i,j+1,k) + sample(value, i,j-1,k)\
                + sample(value, i,j,k+1) + sample(value, i,j,k-1))
    for I in ti.grouped(tmp):
        value[I] = ti.max(value[I], tmp[I])

# temp field is used as the solution

def diffuse_field(field_temp, field, coe):
    copy_to(field, field_temp)
    for it in range(20):
        GS(field, field_temp, coe)
    copy_to(field_temp, field)

@ti.kernel
def GS(field:ti.template(), field_temp:ti.template(), coe:float):
    for i, j, k in field_temp:
        if (i + j + k)%2==0:
            field_temp[i, j, k] = (field[i, j, k] + coe * (
                                sample(field_temp, i - 1, j, k) +
                                sample(field_temp, i + 1, j, k) +
                                sample(field_temp, i, j - 1, k) +
                                sample(field_temp, i, j + 1, k) +
                                sample(field_temp, i, j, k - 1) +
                                sample(field_temp, i, j, k + 1)
                        )) / (1.0 + 6.0 * coe)
    for i, j, k in field_temp:
        if (i + j + k)%2==1:
            field_temp[i, j, k] = (field[i, j, k] + coe * (
                                sample(field_temp, i - 1, j, k) +
                                sample(field_temp, i + 1, j, k) +
                                sample(field_temp, i, j - 1, k) +
                                sample(field_temp, i, j + 1, k) +
                                sample(field_temp, i, j, k - 1) +
                                sample(field_temp, i, j, k + 1)
                        )) / (1.0 + 6.0 * coe)
# # # # interpolation kernels # # # #

@ti.kernel
def interp_f2e(
    ff_x: ti.template(),
    ff_y: ti.template(),
    ff_z: ti.template(),
    ef_x: ti.template(),
    ef_y: ti.template(),
    ef_z: ti.template(),
):
    for i, j, k in ef_x:
        ef_x[i, j, k] = (
            sample(ff_x, i, j, k)
            + sample(ff_x, i, j, k-1)
            + sample(ff_x, i, j-1, k)
            + sample(ff_x, i, j-1, k-1)
            + sample(ff_x, i+1, j, k)
            + sample(ff_x, i+1, j, k-1)
            + sample(ff_x, i+1, j-1, k)
            + sample(ff_x, i+1, j-1, k-1)
        ) * 0.125

    for i, j, k in ef_y:
        ef_y[i, j, k] = (
            sample(ff_y, i, j, k)
            + sample(ff_y, i-1, j, k)
            + sample(ff_y, i, j, k-1)
            + sample(ff_y, i-1, j, k-1)
            + sample(ff_y, i, j+1, k)
            + sample(ff_y, i-1, j+1, k)
            + sample(ff_y, i, j+1, k-1)
            + sample(ff_y, i-1, j+1, k-1)
        ) * 0.125

    for i, j, k in ef_z:
        ef_z[i, j, k] = (
            sample(ff_z, i, j, k)
            + sample(ff_z, i, j-1, k)
            + sample(ff_z, i-1, j, k)
            + sample(ff_z, i-1, j-1, k)
            + sample(ff_z, i, j, k+1)
            + sample(ff_z, i, j-1, k+1)
            + sample(ff_z, i-1, j, k+1)
            + sample(ff_z, i-1, j-1, k+1)
        ) * 0.125

@ti.kernel
def interp_e2f(
    ef_x: ti.template(),
    ef_y: ti.template(),
    ef_z: ti.template(),
    ff_x: ti.template(),
    ff_y: ti.template(),
    ff_z: ti.template(),
):
    for i, j, k in ff_x:
        ff_x[i, j, k] = (
            sample(ef_x, i, j, k)
            + sample(ef_x, i, j+1, k)
            + sample(ef_x, i, j, k+1)
            + sample(ef_x, i, j+1, k+1)
            + sample(ef_x, i-1, j, k)
            + sample(ef_x, i-1, j+1, k)
            + sample(ef_x, i-1, j, k+1)
            + sample(ef_x, i-1, j+1, k+1)
        ) * 0.125

    for i, j, k in ff_y:
        ff_y[i, j, k] = (
            sample(ef_y, i, j, k)
            + sample(ef_y, i+1, j, k)
            + sample(ef_y, i, j, k+1)
            + sample(ef_y, i+1, j, k+1)
            + sample(ef_y, i, j-1, k)
            + sample(ef_y, i+1, j-1, k)
            + sample(ef_y, i, j-1, k+1)
            + sample(ef_y, i+1, j-1, k+1)
        ) * 0.125

    for i, j, k in ff_z:
        ff_z[i, j, k] = (
            sample(ef_z, i, j, k)
            + sample(ef_z, i, j+1, k)
            + sample(ef_z, i+1, j, k)
            + sample(ef_z, i+1, j+1, k)
            + sample(ef_z, i, j, k-1)
            + sample(ef_z, i, j+1, k-1)
            + sample(ef_z, i+1, j, k-1)
            + sample(ef_z, i+1, j+1, k-1)
        ) * 0.125

# linear
@ti.func
def N_1(x):
    return 1.0-ti.abs(x)
    
@ti.func
def dN_1(x):
    result = -1.0
    if x < 0.:
        result = 1.0
    return result

@ti.func
def interp_grad_1(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += inv_dx * (value * dN_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i))
                partial_y += inv_dx * (value * N_1(x_p_x_i) * dN_1(y_p_y_i) * N_1(z_p_z_i))
                partial_z += inv_dx * (value * N_1(x_p_x_i) * N_1(y_p_y_i) * dN_1(z_p_z_i))
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])

@ti.func
def interp_1(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    interped = 0. * sample(vf, iu, iv, iw)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)  
    
    return interped

@ti.func
def sample_min_max_1(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-1-eps))
    t = ti.max(1., ti.min(v, v_dim-1-eps))
    l = ti.max(1., ti.min(w, w_dim-1-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    mini = sample(vf, iu, iv, iw)
    maxi = sample(vf, iu, iv, iw)

    # loop over indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                value = sample(vf, iu + i, iv + j, iw + k)
                mini = ti.min(mini, value)
                maxi = ti.max(maxi, value)

    return mini, maxi



# quadratic
@ti.func
def N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0/4.0 - abs_x ** 2
    elif abs_x < 1.5:
        result = 0.5 * (3.0/2.0-abs_x) ** 2
    return result
    
@ti.func
def dN_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    if x < 0.0: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result

@ti.func
def d2N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2.0
    elif abs_x < 1.5:
        result = 1.0
    else:
        result = 0.0
    return result

@ti.func
def N_3(x):
    abs_x = ti.abs(x)
    result = 0.0
    if abs_x < 1.0:
        result = (4.0/6.0) - x**2 + (abs_x**3)/2.0
    elif abs_x < 2.0:
        result = ((2.0 - abs_x)**3) / 6.0
    return result

# First derivative
@ti.func
def dN_3(x):
    abs_x = ti.abs(x)
    sgn = 1.0 if x >= 0 else -1.0
    result = 0.0
    if abs_x < 1.0:
        result = (-2.0 * x) + (1.5 * x * abs_x)
    elif abs_x < 2.0:
        result = -0.5 * (2.0 - abs_x)**2 * sgn
    return result

# Second derivative
@ti.func
def d2N_3(x):
    abs_x = ti.abs(x)
    result = 0.0
    if abs_x < 1.0:
        result = -2.0 + 3.0 * abs_x
    elif abs_x < 2.0:
        result = (2.0 - abs_x)
    return result


@ti.func
def interp_grad_2(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += inv_dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i))
                partial_y += inv_dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i))
                partial_z += inv_dx * (value * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i))
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])

@ti.func
def interp_grad_3(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    # loop over indices
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                partial_x += inv_dx * (value * dN_3(x_p_x_i) * N_3(y_p_y_i) * N_3(z_p_z_i))
                partial_y += inv_dx * (value * N_3(x_p_x_i) * dN_3(y_p_y_i) * N_3(z_p_z_i))
                partial_z += inv_dx * (value * N_3(x_p_x_i) * N_3(y_p_y_i) * dN_3(z_p_z_i))
                interped += value * N_3(x_p_x_i) * N_3(y_p_y_i) * N_3(z_p_z_i)  
    
    return interped, ti.Vector([partial_x, partial_y, partial_z])


@ti.func
def interp_2(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    interped = 0.

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped

@ti.func
def interp_2_mat(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    interped = ti.Matrix.zero(float, 3, 3)

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped

@ti.func
def interp_2_clamp(vf, p, inv_dx, max_min, I, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    interped = 0.

    max_v = -10000.0
    min_v = 10000.0

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                if value > max_v:
                    max_v = value
                if value < min_v:
                    min_v = value
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    max_min[I] = ti.Vector([min_v, max_v])
    return interped

@ti.func
def interp_2_v(vf, p, inv_dx, BL_x = 0.5, BL_y = 0.5, BL_z = 0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))
    l = ti.max(1., ti.min(w, w_dim-2-eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)
    interped = ti.Vector([0., 0., 0.])

    # loop over indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i) # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)  
    
    return interped

@ti.func
def interp_grad_2_w(vf, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))
    l = ti.max(1., ti.min(w, w_dim - 2 - eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0, 0.0])
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                # imp_value = sample(imp, iu + i, iv + j)
                dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)
                dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i)
                dw_z = 1. / dx * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i)
                partial_x += value * dw_x
                partial_y += value * dw_y
                partial_z += value * dw_z
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)
                new_C += ti.Vector([dw_x, dw_y, dw_z]) * value
                # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, ti.Vector([partial_x, partial_y, partial_z]), new_C


@ti.func
def interp_grad_1_w(vf, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim - 1 - eps))
    t = ti.max(1., ti.min(v, v_dim - 1 - eps))
    l = ti.max(1., ti.min(w, w_dim - 1 - eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0, 0.0])
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                # imp_value = sample(imp, iu + i, iv + j)
                dw_x = 1. / dx * dN_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)
                dw_y = 1. / dx * N_1(x_p_x_i) * dN_1(y_p_y_i) * N_1(z_p_z_i)
                dw_z = 1. / dx * N_1(x_p_x_i) * N_1(y_p_y_i) * dN_1(z_p_z_i)
                partial_x += value * dw_x
                partial_y += value * dw_y
                partial_z += value * dw_z
                interped += value * N_1(x_p_x_i) * N_1(y_p_y_i) * N_1(z_p_z_i)
                new_C += ti.Vector([dw_x, dw_y, dw_z]) * value
                # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, ti.Vector([partial_x, partial_y, partial_z]), new_C


@ti.func
def interp_grad_3_w(vf, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))
    l = ti.max(1., ti.min(w, w_dim - 2 - eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0, 0.0])
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                # imp_value = sample(imp, iu + i, iv + j)
                dw_x = 1. / dx * dN_3(x_p_x_i) * N_3(y_p_y_i) * N_3(z_p_z_i)
                dw_y = 1. / dx * N_3(x_p_x_i) * dN_3(y_p_y_i) * N_3(z_p_z_i)
                dw_z = 1. / dx * N_3(x_p_x_i) * N_3(y_p_y_i) * dN_3(z_p_z_i)
                partial_x += value * dw_x
                partial_y += value * dw_y
                partial_z += value * dw_z
                interped += value * N_3(x_p_x_i) * N_3(y_p_y_i) * N_3(z_p_z_i)
                new_C += ti.Vector([dw_x, dw_y, dw_z]) * value
                # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, ti.Vector([partial_x, partial_y, partial_z]), new_C



@ti.func
def interp_grad_grad_2(vf, p, inv_dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u -= BL_x
    v -= BL_y
    w -= BL_z
    s = ti.max(1.0, ti.min(u, u_dim - 2 - eps))
    t = ti.max(1.0, ti.min(v, v_dim - 2 - eps))
    l = ti.max(1.0, ti.min(w, w_dim - 2 - eps))

    # Floor indices
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    # Initialize accumulators
    interped = 0.0
    partial_x = 0.0
    partial_y = 0.0
    partial_z = 0.0

    partial_xx = 0.0
    partial_yy = 0.0
    partial_zz = 0.0
    partial_xy = 0.0
    partial_xz = 0.0
    partial_yz = 0.0

    # Loop over neighboring grid points
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i)
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)

                N_x = N_2(x_p_x_i)
                N_y = N_2(y_p_y_i)
                N_z = N_2(z_p_z_i)
                dN_x = dN_2(x_p_x_i)
                dN_y = dN_2(y_p_y_i)
                dN_z = dN_2(z_p_z_i)
                d2N_x = d2N_2(x_p_x_i)
                d2N_y = d2N_2(y_p_y_i)
                d2N_z = d2N_2(z_p_z_i)

                # Interpolated value
                interped += value * N_x * N_y * N_z

                # First derivatives
                partial_x += inv_dx * (value * dN_x * N_y * N_z)
                partial_y += inv_dx * (value * N_x * dN_y * N_z)
                partial_z += inv_dx * (value * N_x * N_y * dN_z)

                # Second derivatives
                partial_xx += inv_dx * inv_dx * (value * d2N_x * N_y * N_z)
                partial_yy += inv_dx * inv_dx * (value * N_x * d2N_y * N_z)
                partial_zz += inv_dx * inv_dx * (value * N_x * N_y * d2N_z)

                partial_xy += inv_dx * inv_dx * (value * dN_x * dN_y * N_z)
                partial_xz += inv_dx * inv_dx * (value * dN_x * N_y * dN_z)
                partial_yz += inv_dx * inv_dx * (value * N_x * dN_y * dN_z)

    grad = ti.Vector([partial_x, partial_y, partial_z])
    hessian = ti.Matrix([[partial_xx, partial_xy, partial_xz],
                         [partial_xy, partial_yy, partial_yz],
                         [partial_xz, partial_yz, partial_zz]])

    return interped, grad, hessian

@ti.kernel
def setup_rhs_harmonic(b:ti.types.ndarray(), bm:ti.template(),
                       penalty_u_x:ti.template(), penalty_u_y:ti.template(), penalty_u_z:ti.template(), dx:float):
    dims = ti.Vector(bm.shape)
    for I in ti.grouped(bm):
        ret = 0.0
        if bm[I] <= 0:
            for i in ti.static(range(3)):
                offset = ti.Vector.unit(3, i)
                if I[i] <= 0:
                    ret -= choose_ax(i, I, penalty_u_x, penalty_u_y, penalty_u_z)# / dx
                elif bm[I-offset] >= 1:
                    ret -= choose_ax(i, I, penalty_u_x, penalty_u_y, penalty_u_z)# / dx
                if I[i] >= dims[i] - 2:
                    ret += choose_ax(i, I+offset, penalty_u_x, penalty_u_y, penalty_u_z)# / dx
                elif bm[I+offset] >= 1:
                    ret += choose_ax(i, I+offset, penalty_u_x, penalty_u_y, penalty_u_z)# / dx
        b[I] = ret*dx

        
@ti.kernel
def extend_to_pretorch(a:ti.template(),b:ti.template()):
    shape_x,shape_y,shape_z=a.shape
    for i,j,k in b:
        if(i<shape_x and j<shape_y and k<shape_z):
            b[i,j,k]=a[i,j,k]
        else:
            b[i,j,k]=0


@ti.kernel
def copy_to_external(ti_field:ti.template(), external_array: ti.types.ndarray()):
    for i,j,k in external_array:
        external_array[i,j,k] = ti_field[i,j,k]
@ti.kernel
def copy_from_external(ti_field:ti.template(), external_array: ti.types.ndarray()):
    for i,j,k in ti_field:
        ti_field[i,j,k] = external_array[i,j,k]

@ti.kernel
def copy_to_external(ti_field:ti.template(), external_array: ti.types.ndarray()):
    for i,j,k in external_array:
        external_array[i,j,k] = ti_field[i,j,k]

@ti.kernel
def add_grad_p(u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), p:ti.template(), bm:ti.template()):
    u_dim, v_dim, w_dim = p.shape
    for i, j, k in u_x:
        if i>=1 and i <=u_dim-2 and (bm[i-1,j,k] + bm[i,j,k])<=1:
            pr = sample(p, i, j, k)
            pl = sample(p, i-1, j, k)
            u_x[i,j,k] += pr - pl
    for i, j, k in u_y:
        if j>=1 and j<=v_dim-2 and (bm[i,j-1,k] + bm[i,j,k])<=1:
            pt = sample(p, i, j, k)
            pb = sample(p, i, j-1, k)
            u_y[i,j,k] += pt - pb
    for i, j, k in u_z:
        if k>=1 and k<= w_dim-2 and (bm[i,j,k-1] + bm[i,j,k])<=1:
            pc = sample(p, i, j, k)
            pa = sample(p, i, j, k-1)
            u_z[i,j,k] += pc - pa
@ti.func
def interp_grad_grad_3(vf, p, inv_dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p * inv_dx
    u -= BL_x
    v -= BL_y
    w -= BL_z

    # Adjust indices for cubic B-spline support
    s = ti.max(2.0, ti.min(u, u_dim - 3 - eps))
    t = ti.max(2.0, ti.min(v, v_dim - 3 - eps))
    l = ti.max(2.0, ti.min(w, w_dim - 3 - eps))

    # Floor indices
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    # Initialize accumulators
    interped = 0.0
    partial_x = 0.0
    partial_y = 0.0
    partial_z = 0.0

    partial_xx = 0.0
    partial_yy = 0.0
    partial_zz = 0.0
    partial_xy = 0.0
    partial_xz = 0.0
    partial_yz = 0.0

    # Loop over neighboring grid points
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                x_p_x_i = s - (iu + i)
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)

                N_x = N_3(x_p_x_i)
                N_y = N_3(y_p_y_i)
                N_z = N_3(z_p_z_i)
                dN_x = dN_3(x_p_x_i)
                dN_y = dN_3(y_p_y_i)
                dN_z = dN_3(z_p_z_i)
                d2N_x = d2N_3(x_p_x_i)
                d2N_y = d2N_3(y_p_y_i)
                d2N_z = d2N_3(z_p_z_i)

                # Interpolated value
                interped += value * N_x * N_y * N_z

                # First derivatives
                partial_x += inv_dx * value * dN_x * N_y * N_z
                partial_y += inv_dx * value * N_x * dN_y * N_z
                partial_z += inv_dx * value * N_x * N_y * dN_z

                # Second derivatives
                partial_xx += inv_dx * inv_dx * value * d2N_x * N_y * N_z
                partial_yy += inv_dx * inv_dx * value * N_x * d2N_y * N_z
                partial_zz += inv_dx * inv_dx * value * N_x * N_y * d2N_z

                partial_xy += inv_dx * inv_dx * value * dN_x * dN_y * N_z
                partial_xz += inv_dx * inv_dx * value * dN_x * N_y * dN_z
                partial_yz += inv_dx * inv_dx * value * N_x * dN_y * dN_z

    grad = ti.Vector([partial_x, partial_y, partial_z])
    hessian = ti.Matrix([[partial_xx, partial_xy, partial_xz],
                         [partial_xy, partial_yy, partial_yz],
                         [partial_xz, partial_yz, partial_zz]])

    return interped, grad, hessian
@ti.func
def interp_grad_2_imp(vf, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.5):
    u_dim, v_dim, w_dim = vf.shape

    u, v, w = p / dx
    u = u - BL_x
    v = v - BL_y
    w = w - BL_z
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))
    l = ti.max(1., ti.min(w, w_dim - 2 - eps))

    # floor
    iu, iv, iw = ti.floor(s), ti.floor(t), ti.floor(l)

    partial_x = 0.
    partial_y = 0.
    partial_z = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0, 0.0])
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                x_p_x_i = s - (iu + i)  # x_p - x_i
                y_p_y_i = t - (iv + j)
                z_p_z_i = l - (iw + k)
                value = sample(vf, iu + i, iv + j, iw + k)
                # imp_value = sample(imp, iu + i, iv + j)
                dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)
                dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i) * N_2(z_p_z_i)
                dw_z = 1. / dx * N_2(x_p_x_i) * N_2(y_p_y_i) * dN_2(z_p_z_i)
                partial_x += value * dw_x
                partial_y += value * dw_y
                partial_z += value * dw_z
                interped += value * N_2(x_p_x_i) * N_2(y_p_y_i) * N_2(z_p_z_i)
                new_C += ti.Vector([dw_x, dw_y, dw_z]) * value
                # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, ti.Vector([partial_x, partial_y, partial_z]), new_C


@ti.func
def interp_u_MAC_grad_imp(u_x, u_y, u_z, p, dx):
    u_x_p, grad_u_x_p, C_x = interp_grad_2_imp(u_x, p, dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    u_y_p, grad_u_y_p, C_y = interp_grad_2_imp(u_y, p, dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    u_z_p, grad_u_z_p, C_z = interp_grad_2_imp(u_z, p, dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p, grad_u_z_p]), C_x, C_y, C_z

@ti.func
def interp_u_MAC_grad_w_1(w_x, w_y, w_z, p, dx):
    w_x_p, grad_w_x_p, C_x = interp_grad_1_w(w_x, p, dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
    w_y_p, grad_w_y_p, C_y = interp_grad_1_w(w_y, p, dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
    w_z_p, grad_w_z_p, C_z = interp_grad_1_w(w_z, p, dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
    return ti.Vector([w_x_p, w_y_p, w_z_p]), ti.Matrix.rows([grad_w_x_p, grad_w_y_p, grad_w_z_p]), C_x, C_y, C_z


@ti.func
def interp_u_MAC_grad_w(w_x, w_y, w_z, p, dx):
    w_x_p, grad_w_x_p, C_x = interp_grad_2_w(w_x, p, dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
    w_y_p, grad_w_y_p, C_y = interp_grad_2_w(w_y, p, dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
    w_z_p, grad_w_z_p, C_z = interp_grad_2_w(w_z, p, dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
    return ti.Vector([w_x_p, w_y_p, w_z_p]), ti.Matrix.rows([grad_w_x_p, grad_w_y_p, grad_w_z_p]), C_x, C_y, C_z

@ti.func
def interp_u_MAC_grad_w_3(w_x, w_y, w_z, p, dx):
    w_x_p, grad_w_x_p, C_x = interp_grad_3_w(w_x, p, dx, BL_x=0.5, BL_y=0.0, BL_z=0.0)
    w_y_p, grad_w_y_p, C_y = interp_grad_3_w(w_y, p, dx, BL_x=0.0, BL_y=0.5, BL_z=0.0)
    w_z_p, grad_w_z_p, C_z = interp_grad_3_w(w_z, p, dx, BL_x=0.0, BL_y=0.0, BL_z=0.5)
    return ti.Vector([w_x_p, w_y_p, w_z_p]), ti.Matrix.rows([grad_w_x_p, grad_w_y_p, grad_w_z_p]), C_x, C_y, C_z


@ti.func
def interp_u_MAC_grad_transpose(u_x, u_y, u_z, p, dx):
    u_x_p, grad_u_x_p = interp_grad_2(u_x, p, 1./dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    u_y_p, grad_u_y_p = interp_grad_2(u_y, p, 1./dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    u_z_p, grad_u_z_p = interp_grad_2(u_z, p, 1./dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p]), ti.Matrix.cols([grad_u_x_p, grad_u_y_p, grad_u_z_p])

@ti.func
def interp_u_MAC_grad_transpose_3(u_x, u_y, u_z, p, dx):
    u_x_p, grad_u_x_p = interp_grad_3(u_x, p, 1./dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    u_y_p, grad_u_y_p = interp_grad_3(u_y, p, 1./dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    u_z_p, grad_u_z_p = interp_grad_3(u_z, p, 1./dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)
    return ti.Vector([u_x_p, u_y_p, u_z_p]), ti.Matrix.cols([grad_u_x_p, grad_u_y_p, grad_u_z_p])


@ti.func
def interp_u_MAC_grad_grad_transpose(u_x, u_y, u_z, p, dx):
    inv_dx = 1.0 / dx
    # Interpolate u_x and its derivatives
    u_x_p, grad_u_x_p, hessian_u_x_p = interp_grad_grad_2(
        u_x, p, inv_dx, BL_x=0.0, BL_y=0.5, BL_z=0.5)
    # Interpolate u_y and its derivatives
    u_y_p, grad_u_y_p, hessian_u_y_p = interp_grad_grad_2(
        u_y, p, inv_dx, BL_x=0.5, BL_y=0.0, BL_z=0.5)
    # Interpolate u_z and its derivatives
    u_z_p, grad_u_z_p, hessian_u_z_p = interp_grad_grad_2(
        u_z, p, inv_dx, BL_x=0.5, BL_y=0.5, BL_z=0.0)

    # Assemble velocity vector
    u_p = ti.Vector([u_x_p, u_y_p, u_z_p])
    # Assemble gradient of velocity (Jacobian matrix)
    grad_u_p = ti.Matrix.cols([grad_u_x_p, grad_u_y_p, grad_u_z_p])

    return u_p, grad_u_p, hessian_u_x_p, hessian_u_y_p, hessian_u_z_p


@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

@ti.func
def judge_inside_w_boud_x(i, j, k, boundary_mask):
    ret = boundary_mask[i, j, k] + boundary_mask[i, j-1, k] + boundary_mask[i, j, k-1] + boundary_mask[i, j-1, k-1]
    return ret

@ti.func
def judge_inside_w_boud_y(i, j, k, boundary_mask):
    ret = boundary_mask[i, j, k] + boundary_mask[i-1, j, k] + boundary_mask[i-1, j, k-1] + boundary_mask[i, j, k-1]
    return ret

@ti.func
def judge_inside_w_boud_z(i, j, k, boundary_mask):
    ret = boundary_mask[i, j, k] + boundary_mask[i-1, j, k] + boundary_mask[i, j-1, k] + boundary_mask[i-1, j-1, k]
    return ret


@ti.kernel
def apply_wb_0(w_x: ti.template(), w_y: ti.template(), w_z: ti.template()):
    u_dim, v_dim, w_dim = w_x.shape
    for i, j, k in w_x:
        if j == 0 or j == v_dim - 1 or k == 0 or k == w_dim - 1:
            w_x[i, j, k] = 0.
            continue

    u_dim, v_dim, w_dim = w_y.shape
    for i, j, k in w_y:
        if i == 0 or i == u_dim - 1 or k == 0 or k == w_dim - 1:
            w_y[i, j, k] = 0.
            continue
    u_dim, v_dim, w_dim = w_z.shape
    for i, j, k in w_z:
        if j == 0 or j == v_dim - 1 or i == 0 or i == u_dim - 1:
            w_z[i, j, k] = 0.
            continue




@ti.kernel
def apply_onlyonwall_w_stream_noslip(
    u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
    w_x: ti.template(), w_y: ti.template(), w_z: ti.template(),
    stream_x: ti.template(), stream_y: ti.template(), stream_z: ti.template(),
    boundary_mask: ti.template(), boundary_vel: ti.template(), inv_dx: ti.f32):
    h2 = inv_dx * inv_dx
    u_dim, v_dim, w_dim = w_x.shape
    for i, j, k in w_x:
        psi_0 = stream_x[i, j, k]
        if j == 0:
            psi_1 = stream_x[i, j+1, k]
            psi_2 = stream_x[i, j+2, k]
            w_x[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

        if j == v_dim - 1:
            psi_1 = stream_x[i, j-1, k]
            psi_2 = stream_x[i, j-2, k]
            w_x[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

        if k == 0:
            psi_1 = stream_x[i, j, k+1]
            psi_2 = stream_x[i, j, k+2]
            w_x[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue
        
        if k == w_dim - 1:
            psi_1 = stream_x[i, j, k-1]
            psi_2 = stream_x[i, j, k-2]
            w_x[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

    u_dim, v_dim, w_dim = w_y.shape
    for i, j, k in w_y:
        psi_0 = stream_y[i, j, k]
        if i == 0:
            psi_1 = stream_y[i+1, j, k]
            psi_2 = stream_y[i+2, j, k]
            w_y[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

        if i == u_dim - 1:
            psi_1 = stream_y[i-1, j, k]
            psi_2 = stream_y[i-2, j, k]
            w_y[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

        if k == 0:
            psi_1 = stream_y[i, j, k+1]
            psi_2 = stream_y[i, j, k+2]
            w_y[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue
        
        if k == w_dim - 1:
            psi_1 = stream_y[i, j, k-1]
            psi_2 = stream_y[i, j, k-2]
            w_y[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

    u_dim, v_dim, w_dim = w_z.shape
    for i, j, k in w_z:
        psi_0 = stream_z[i, j, k]
        if i == 0:
            psi_1 = stream_z[i+1, j, k]
            psi_2 = stream_z[i+2, j, k]
            w_z[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

        if i == u_dim - 1:
            psi_1 = stream_z[i-1, j, k]
            psi_2 = stream_z[i-2, j, k]
            w_z[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

        if j == 0:
            psi_1 = stream_z[i, j+1, k]
            psi_2 = stream_z[i, j+2, k]
            w_z[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue
        
        if j == v_dim - 1:
            psi_1 = stream_z[i, j-1, k]
            psi_2 = stream_z[i, j-2, k]
            w_z[i, j, k] = -(4.0 * psi_1 - 0.5 * psi_2 - 3.5 * psi_0)*h2
            continue

@ti.kernel
def apply_onlyonwall_w_v_noslip(
    u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
    w_x: ti.template(), w_y: ti.template(), w_z: ti.template(),
    stream_x: ti.template(), stream_y: ti.template(), stream_z: ti.template(),
    boundary_mask: ti.template(), boundary_vel: ti.template(), inv_dx: ti.f32):
    h2 = inv_dx * inv_dx
    u_dim, v_dim, w_dim = w_x.shape
    for i, j, k in w_x:
        if j == 0:
            vt = u_z[i, j, k]
            w_x[i, j, k] = (vt) * inv_dx
            continue

        if j == v_dim - 1:
            vb = u_z[i, j-1, k]
            w_x[i, j, k] = - (vb) * inv_dx
            continue

        if k == 0:
            vc = u_y[i, j, k]
            w_x[i, j, k] = - (vc) * inv_dx
            continue
        
        if k == w_dim - 1:
            va = u_y[i, j, k-1]
            w_x[i, j, k] =  (va) * inv_dx
            continue


    u_dim, v_dim, w_dim = w_y.shape
    for i, j, k in w_y:
        if i == 0:
            vr = u_z[i, j, k]
            w_y[i, j, k] = - (vr) * inv_dx
            continue

        if i == u_dim - 1:
            vl = u_z[i - 1, j, k]
            w_y[i, j, k] = (vl) * inv_dx
            continue

        if k == 0:
            vc = u_x[i, j, k]
            w_y[i, j, k] = (vc) * inv_dx
            continue
        
        if k == w_dim - 1:
            va = u_x[i, j, k-1]
            w_y[i, j, k] =  - (va) * inv_dx
            continue

    u_dim, v_dim, w_dim = w_z.shape
    for i, j, k in w_z:
        if i == 0:
            vr = u_y[i, j, k]
            w_z[i, j, k] = (vr) * inv_dx
            continue

        if i == u_dim - 1:
            vl = u_y[i - 1, j, k]
            w_z[i, j, k] = -(vl) * inv_dx
            continue

        if j == 0:
            vt = u_x[i, j, k]
            w_z[i, j, k] = - (vt) * inv_dx
            continue
        
        if j == v_dim - 1:
            vb = u_x[i, j-1, k]
            w_z[i, j, k] =  (vb) * inv_dx
            continue


@ti.kernel
def apply_bc_w(
    u_x: ti.template(), u_y: ti.template(), u_z: ti.template(),
    w_x: ti.template(), w_y: ti.template(), w_z: ti.template(),
    stream_x: ti.template(), stream_y: ti.template(), stream_z: ti.template(),
    boundary_mask: ti.template(), boundary_vel: ti.template(), inv_dx: ti.f32):
    h2 = inv_dx * inv_dx
    u_dim, v_dim, w_dim = w_x.shape
    for i, j, k in w_x:
        vt = vb = vc = va = ti.cast(0., dtype = data_type)
        # 4 corners
        if (k == 0 and j == 0) or (k == 0 and j == v_dim - 1) or (k == w_dim - 1 and j == 0) or (k == w_dim - 1 and j == v_dim - 1):
            w_x[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        
        if j == 0:
            w_x[i,j,k] = 0
            continue

        if j == v_dim - 1:
            w_x[i,j,k] = 0
            continue

        if k == 0:
            w_x[i,j,k] = 0
            continue
    
        if k == w_dim - 1:
            w_x[i,j,k] = 0
            continue

        num_solid_neighbors = judge_inside_w_boud_x(i, j, k, boundary_mask)
        if num_solid_neighbors == 4:
            w_x[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        else:
            # back has wall
            if boundary_mask[i, j-1, k-1] > 0 and boundary_mask[i, j, k-1]>0:
                va = boundary_vel[i, j-1, k-1][1] + boundary_vel[i, j, k-1][1] - u_y[i, j, k]
            else:
                va = u_y[i, j, k-1]

            # frd has wall
            if boundary_mask[i, j-1, k] > 0 and boundary_mask[i, j, k]>0:
                vc = boundary_vel[i, j-1, k][1] + boundary_vel[i, j, k][1] - u_y[i, j, k-1]
            else:
                vc = u_y[i, j, k]

            # bottom has wall
            if boundary_mask[i, j-1, k] > 0 and boundary_mask[i, j-1, k-1]>0:
                vb = boundary_vel[i, j-1, k][2] + boundary_vel[i, j-1, k-1][2] - u_z[i, j, k]
            else:
                vb = u_z[i, j-1, k]

            # top has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i, j, k-1]>0:
                vt = boundary_vel[i, j, k][2] + boundary_vel[i, j, k-1][2] - u_z[i, j-1, k]
            else:
                vt = u_z[i, j, k]

            w_x[i, j, k] = ((vt - vb) - (vc - va)) * inv_dx


    u_dim, v_dim, w_dim = w_y.shape
    for i, j, k in w_y:
        vc = va = vr = vl = ti.cast(0., dtype = data_type)
        # 4 corners
        if (k == 0 and i == 0) or (k == 0 and i == u_dim - 1) or (k == w_dim - 1 and i == 0) or (k == w_dim - 1 and i == u_dim - 1):
            w_y[i, j, k] = ti.cast(0., dtype = data_type)
            continue
    
        if i == 0:
            w_y[i,j,k]=0
            continue

        if i == u_dim - 1:
            w_y[i,j,k]=0
            continue

            w_y[i,j,k]=0
            continue
    
        if k == w_dim - 1:
            w_y[i,j,k]=0
            continue

        num_solid_neighbors = judge_inside_w_boud_y(i, j, k, boundary_mask)
        if num_solid_neighbors == 4:
            w_y[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        else:
            # frd
            if boundary_mask[i, j, k] > 0 and boundary_mask[i-1, j, k]>0:
                vc = boundary_vel[i, j, k][0] + boundary_vel[i-1, j, k][0] - u_x[i, j, k-1]
            else:
                vc = u_x[i, j, k]

            # back has wall
            if boundary_mask[i, j, k-1] > 0 and boundary_mask[i-1, j, k-1]>0:
                va = boundary_vel[i, j, k-1][0] + boundary_vel[i-1, j, k-1][0] - u_x[i, j, k]
            else:
                va = u_x[i, j, k-1]

            # right has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i, j, k-1]>0:
                vr = boundary_vel[i, j, k][2] + boundary_vel[i, j, k-1][2] - u_z[i-1, j, k]
            else:
                vr = u_z[i, j, k]

            # left has wall
            if boundary_mask[i-1, j, k] > 0 and boundary_mask[i-1, j, k-1]>0:
                vl = boundary_vel[i-1, j, k][2] + boundary_vel[i-1, j, k-1][2] - u_z[i, j, k]
            else:
                vl = u_z[i - 1, j, k]

            w_y[i, j, k] = ((vc - va) - (vr - vl)) * inv_dx

    u_dim, v_dim, w_dim = w_z.shape
    for i, j, k in w_z:
        vr = vl = vt = vb = ti.cast(0., dtype = data_type)
        # 4 corners
        if (j == 0 and i == 0) or (j == 0 and i == u_dim - 1) or (j == v_dim - 1 and i == 0) or (j == v_dim - 1 and i == u_dim - 1):
            w_z[i, j, k] = ti.cast(0., dtype = data_type)
            continue

        if i == 0:
            w_z[i, j, k] = 0
            continue

        if i == u_dim - 1:
            w_z[i, j, k] = 0
            continue

        if j == 0:
            w_z[i, j, k] = 0
            continue
        
        if j == v_dim - 1:
            w_z[i, j, k] = 0
            continue
        
        num_solid_neighbors = judge_inside_w_boud_z(i, j, k, boundary_mask)
        if num_solid_neighbors == 4:
            w_z[i, j, k] = ti.cast(0., dtype = data_type)
            continue
        else:
            # right has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i, j-1, k]>0:
                vr = boundary_vel[i, j, k][1] + boundary_vel[i, j-1, k][1] - u_y[i-1, j, k]
            else:
                vr = u_y[i, j, k]

            # left has wall
            if boundary_mask[i-1, j, k] > 0 and boundary_mask[i-1, j-1, k]>0:
                vl = boundary_vel[i-1, j, k][1] + boundary_vel[i-1, j-1, k][1] - u_y[i, j, k]
            else:
                vl = u_y[i-1, j, k]

            # top has wall
            if boundary_mask[i, j, k] > 0 and boundary_mask[i-1, j, k]>0:
                vt = boundary_vel[i, j, k][0] + boundary_vel[i-1, j, k][0] - u_x[i, j - 1, k]
            else:
                vt = u_x[i, j, k]

            # bottom has wall
            if boundary_mask[i-1, j-1, k] > 0 and boundary_mask[i, j-1, k]>0:
                vb = boundary_vel[i-1, j-1, k][0] + boundary_vel[i, j-1, k][0] - u_x[i, j, k]
            else:
                vb = u_x[i, j - 1, k]

            w_z[i, j, k] = ((vr - vl) - (vt - vb)) * inv_dx

@ti.kernel
def copy_trefoil_buffer(smoke_buffer:ti.template(), smoke:ti.template()):
    for I in ti.grouped(smoke_buffer):
        smoke[I][0] = smoke_buffer[I][0]
        smoke[I][1] = smoke_buffer[I][1]
        smoke[I][2] = smoke_buffer[I][2]
        smoke[I][4] = smoke_buffer[I][3]


@ti.kernel
def comput_divw(w_x:ti.template(), w_y:ti.template(), w_z:ti.template(),div_w:ti.template(), inv_dx:float):
    for i, j, k in div_w:
        div_w[i, j, k] = (sample(w_x, i, j, k) - sample(w_x, i-1, j, k) + sample(w_y, i, j, k) - 
        sample(w_y, i, j-1, k) + sample(w_z, i, j, k) - sample(w_z, i, j, k-1)) * inv_dx


@ti.kernel
def edge_from_center_boundary(center_boundary_mask:ti.template(),edge_x_boundary_mask:ti.template(),edge_y_boundary_mask:ti.template(),edge_z_boundary_mask:ti.template()):
    edge_x_boundary_mask.fill(0)
    edge_y_boundary_mask.fill(0)
    edge_z_boundary_mask.fill(0)
    for i,j,k in center_boundary_mask:
        if(center_boundary_mask[i,j,k]>=1):
            edge_x_boundary_mask[i,j,k]=1
            edge_x_boundary_mask[i,j+1,k]=1
            edge_x_boundary_mask[i,j,k+1]=1
            edge_x_boundary_mask[i,j+1,k+1]=1

            edge_y_boundary_mask[i,j,k]=1
            edge_y_boundary_mask[i+1,j,k]=1
            edge_y_boundary_mask[i,j,k+1]=1
            edge_y_boundary_mask[i+1,j,k+1]=1

            edge_z_boundary_mask[i,j,k]=1
            edge_z_boundary_mask[i,j+1,k]=1
            edge_z_boundary_mask[i+1,j,k]=1
            edge_z_boundary_mask[i+1,j+1,k]=1

    shape_x,shape_y,shape_z=edge_x_boundary_mask.shape
    for i,j,k in edge_x_boundary_mask:
        if(k == 0 or j == 0 or k==shape_z-1 or j==shape_y-1):
            edge_x_boundary_mask[i,j,k]=1
    shape_x,shape_y,shape_z=edge_y_boundary_mask.shape
    for i,j,k in edge_y_boundary_mask:
        if(i == 0 or k == 0 or i==shape_x-1 or k==shape_z-1):
            edge_y_boundary_mask[i,j,k]=1
    shape_x,shape_y,shape_z=edge_z_boundary_mask.shape
    for i,j,k in edge_z_boundary_mask:
        if(i == 0 or j == 0 or i==shape_x-1 or j==shape_y-1):
            edge_z_boundary_mask[i,j,k]=1

@ti.kernel
def edge_from_center_boundary0(center_boundary_mask:ti.template(),edge_x_boundary_mask:ti.template(),edge_y_boundary_mask:ti.template(),edge_z_boundary_mask:ti.template()):
    edge_x_boundary_mask.fill(0)
    edge_y_boundary_mask.fill(0)
    edge_z_boundary_mask.fill(0)
    for i,j,k in center_boundary_mask:
        if(center_boundary_mask[i,j,k]>=1):
            edge_x_boundary_mask[i,j,k]=1
            edge_x_boundary_mask[i,j+1,k]=1
            edge_x_boundary_mask[i,j,k+1]=1
            edge_x_boundary_mask[i,j+1,k+1]=1

            edge_y_boundary_mask[i,j,k]=1
            edge_y_boundary_mask[i+1,j,k]=1
            edge_y_boundary_mask[i,j,k+1]=1
            edge_y_boundary_mask[i+1,j,k+1]=1

            edge_z_boundary_mask[i,j,k]=1
            edge_z_boundary_mask[i,j+1,k]=1
            edge_z_boundary_mask[i+1,j,k]=1
            edge_z_boundary_mask[i+1,j+1,k]=1

@ti.kernel
def face_from_center_boundary(center_boundary_mask:ti.template(),face_x_boundary_mask:ti.template(),face_y_boundary_mask:ti.template(), face_z_boundary_mask:ti.template()):
    face_x_boundary_mask.fill(0)
    face_y_boundary_mask.fill(0)
    face_z_boundary_mask.fill(0)
    for i,j,k in center_boundary_mask:
        if(center_boundary_mask[i,j,k]>=1):
            face_x_boundary_mask[i,j,k]=1
            face_x_boundary_mask[i+1,j,k]=1
            face_y_boundary_mask[i,j,k]=1
            face_y_boundary_mask[i,j+1,k]=1
            face_z_boundary_mask[i,j,k]=1
            face_z_boundary_mask[i,j,k+1]=1

    shape_x,shape_y,shape_z=face_x_boundary_mask.shape
    for i,j,k in face_x_boundary_mask:
        if(i==0 or i== shape_x-1):
            face_x_boundary_mask[i,j,k]=1

    shape_x,shape_y,shape_z=face_y_boundary_mask.shape
    for i,j,k in face_y_boundary_mask:
        if(j==0 or j== shape_y-1):
            face_y_boundary_mask[i,j,k]=1

    shape_x,shape_y,shape_z=face_z_boundary_mask.shape
    for i,j,k in face_z_boundary_mask:
        if(k==0 or k== shape_z-1):
            face_z_boundary_mask[i,j,k]=1

@ti.kernel
def face_from_center_boundary0(center_boundary_mask:ti.template(),face_x_boundary_mask:ti.template(),face_y_boundary_mask:ti.template(), face_z_boundary_mask:ti.template()):
    face_x_boundary_mask.fill(0)
    face_y_boundary_mask.fill(0)
    face_z_boundary_mask.fill(0)
    for i,j,k in center_boundary_mask:
        if(center_boundary_mask[i,j,k]>=1):
            face_x_boundary_mask[i,j,k]=1
            face_x_boundary_mask[i+1,j,k]=1
            face_y_boundary_mask[i,j,k]=1
            face_y_boundary_mask[i,j+1,k]=1
            face_z_boundary_mask[i,j,k]=1
            face_z_boundary_mask[i,j,k+1]=1

@ti.kernel
def extend_boundary_field(a:ti.template(),b:ti.template()):
    shape_x,shape_y,shape_z=a.shape
    for i,j,k in b:
        if(i<shape_x and j<shape_y and k<shape_z):
            b[i,j,k]=a[i,j,k]
        else:
            b[i,j,k]=1

@ti.kernel
def extend_surf_field(a:ti.template(),b:ti.template()):
    shape_x,shape_y,shape_z=a.shape
    for i,j,k in b:
        if(i<shape_x and j<shape_y and k<shape_z):
            b[i,j,k]=a[i,j,k]
        else:
            b[i,j,k]=0

@ti.kernel
def apply_bc_streamu(u_horizontal: ti.template(), u_vertical: ti.template(), u_frontback: ti.template(), boundary_types: ti.template(), center_boundary_mask: ti.template()):
    u_dim, v_dim, w_dim = u_horizontal.shape
    for i, j, k in u_horizontal:
        if i == 0 and boundary_types[0, 0] == 2:
            u_horizontal[i,j, k] = 0.0
        if i == u_dim - 1 and boundary_types[0, 1] == 2:
            u_horizontal[i,j, k] = 0.0

    u_dim, v_dim, w_dim = u_vertical.shape
    for i, j, k in u_vertical:
        if j == 0 and boundary_types[1,0] == 2:
            u_vertical[i,j, k] = 0.0
        if j == v_dim - 1 and boundary_types[1,1] == 2:
            u_vertical[i,j, k] = 0.0

    u_dim, v_dim, w_dim = u_frontback.shape
    for i, j, k in u_frontback:
        if k == 0 and boundary_types[2,0] == 2:
            u_frontback[i,j, k] = 0.0
        if k == w_dim - 1 and boundary_types[2,1] == 2:
            u_frontback[i,j, k] = 0.0

    for i, j, k in center_boundary_mask:
        if center_boundary_mask[i, j, k] > 0:
            u_horizontal[i, j, k] = 0.0
            u_horizontal[i + 1, j, k] = 0.0
            u_vertical[i, j, k] = 0.0
            u_vertical[i, j + 1, k] = 0.0
            u_frontback[i, j, k] = 0.0
            u_frontback[i, j, k+1] = 0.0

def fijkw_to4e(i, j, k, w):
    if w == 0:
        v1 = np.array([i, j+1, k, 2])
        v2 = np.array([i, j, k, 2])
        v3 = np.array([i, j, k+1, 1])
        v4 = np.array([i, j, k, 1])
    elif w == 1:
        v1 = np.array([i, j, k+1, 0])
        v2 = np.array([i, j, k, 0])
        v3 = np.array([i+1, j, k, 2])
        v4 = np.array([i, j, k, 2])
    elif w == 2:
        v1 = np.array([i+1, j, k, 1])
        v2 = np.array([i, j, k, 1])
        v3 = np.array([i, j+1, k, 0])
        v4 = np.array([i, j, k, 0])
    return v1, v2, v3, v4
# now suppose I have all the mappings
# first construct b, still need to handle the center to
def edgeijkw_ton(v,edge_ijkw_ton_x,edge_ijkw_ton_y,edge_ijkw_ton_z):
    if v[3] == 0:
        n = edge_ijkw_ton_x[v[0], v[1], v[2]]
    elif v[3] == 1:
        n = edge_ijkw_ton_y[v[0], v[1], v[2]]
    else:
        n = edge_ijkw_ton_z[v[0], v[1], v[2]]
    return n

def assign_mapping_np(surf_bonundary_mask, mapping, n_to_ijkw, n, axis):
    udim, vdim, wdim = surf_bonundary_mask.shape
    for i in range(udim):
        for j in range(vdim):
            for k in range(wdim):
                if surf_bonundary_mask[i, j, k] == 1:
                    mapping[i, j, k] = n
                    n_to_ijkw[n] = np.array([i, j, k, axis])
                    n += 1
    return mapping, n_to_ijkw, n


def construct_Ab(boundary_mask, boundary_vel, edge_ijkw_ton_x, 
                edge_ijkw_ton_y, edge_ijkw_ton_z, m2ijkw, m, n, dx):
    A_row_index= []
    A_col_index= []
    A_data_list= []
    b = np.zeros(m, dtype=float)
    for i in range(m):
        # if m2ijkw[i][0] != -1:
        index = m2ijkw[i]
        offset = np.eye(3)[index[3]]
        I = np.array([index[0],index[1],index[2]])
        I_offset = np.array(I - offset, dtype = np.int32)
        # print(I)
        if boundary_mask[I[0], I[1], I[2]] >= 1:
            b[i] = boundary_vel[I[0], I[1], I[2]][index[3]] * dx
        else:
            b[i] = boundary_vel[I_offset[0], I_offset[1], I_offset[2]][index[3]] * dx

        v1, v2, v3, v4 = fijkw_to4e(index[0], index[1], index[2], index[3])
        n1 = edgeijkw_ton(v1, edge_ijkw_ton_x, edge_ijkw_ton_y, edge_ijkw_ton_z)
        n2 = edgeijkw_ton(v2, edge_ijkw_ton_x, edge_ijkw_ton_y, edge_ijkw_ton_z)
        n3 = edgeijkw_ton(v3, edge_ijkw_ton_x, edge_ijkw_ton_y, edge_ijkw_ton_z)
        n4 = edgeijkw_ton(v4, edge_ijkw_ton_x, edge_ijkw_ton_y, edge_ijkw_ton_z)

        A_row_index.extend([i, i, i, i])
        A_col_index.extend([n1, n2, n3, n4])
        A_data_list.extend([1, -1, -1, 1])
    
    return A_row_index, A_col_index, A_data_list, b



def solve_stream_sparse(surf_edge_bonundary_mask_x_np, surf_edge_bonundary_mask_y_np, surf_edge_bonundary_mask_z_np,
                        surf_face_bonundary_mask_x_np, surf_face_bonundary_mask_y_np, surf_face_bonundary_mask_z_np,
                        edge_ijkw_ton_x_np, edge_ijkw_ton_y_np, edge_ijkw_ton_z_np, n2ijkw_np,
                        face_ijkw_ton_x_np, face_ijkw_ton_y_np, face_ijkw_ton_z_np, m2ijkw_np,
                        boundary_mask_np, boundary_vel_np, dx):
    import time
    t1 = time.time()
    n_np = 0
    m_np = 0
    edge_ijkw_ton_x_np, n2ijkw_np, n_np = assign_mapping_np(surf_edge_bonundary_mask_x_np, edge_ijkw_ton_x_np, n2ijkw_np, n_np, 0)
    edge_ijkw_ton_y_np, n2ijkw_np, n_np = assign_mapping_np(surf_edge_bonundary_mask_y_np, edge_ijkw_ton_y_np, n2ijkw_np, n_np, 1)
    edge_ijkw_ton_z_np, n2ijkw_np, n_np = assign_mapping_np(surf_edge_bonundary_mask_z_np, edge_ijkw_ton_z_np, n2ijkw_np, n_np, 2)

    face_ijkw_ton_x_np, m2ijkw_np, m_np = assign_mapping_np(surf_face_bonundary_mask_x_np, face_ijkw_ton_x_np, m2ijkw_np, m_np, 0)
    face_ijkw_ton_y_np, m2ijkw_np, m_np = assign_mapping_np(surf_face_bonundary_mask_y_np, face_ijkw_ton_y_np, m2ijkw_np, m_np, 1)
    face_ijkw_ton_z_np, m2ijkw_np, m_np = assign_mapping_np(surf_face_bonundary_mask_z_np, face_ijkw_ton_z_np, m2ijkw_np, m_np, 2)

    t2 = time.time()
    
    A_row_index, A_col_index, A_data_list, b = construct_Ab(boundary_mask_np, boundary_vel_np, edge_ijkw_ton_x_np, 
                    edge_ijkw_ton_y_np, edge_ijkw_ton_z_np, m2ijkw_np, m_np, n_np, dx)

    t3 = time.time()
    # print(b)
    A_data = np.array(A_data_list)
    A_row_indices = np.array(A_row_index)
    A_col_indices = np.array(A_col_index)
    A_shape = (m_np, n_np)

    A = csc_matrix((A_data, (A_row_indices, A_col_indices)), shape=A_shape)
    # print(A.transpose()[0])
    print(A.transpose()[10000])
    print(n2ijkw_np[10000], m2ijkw_np[1896], m2ijkw_np[1897])

    t4 = time.time()
    result = lsqr(A, b)

    t5 = time.time()
    # print("mapping time:", t2- t1)
    # print("construct ab time:", t3 - t2)
    # print("csc matrix construction time:", t4 - t3)
    # print("solving time:", t5-t4)
    # print(result)
    x = result[0]
    residual = A @ x - b
    # print("\nResidual (Ax - b):")
    # print(residual)
    # print("\nNorm of residual:")
    # print(np.linalg.norm(residual))
    x = np.array(x, dtype=np.float32)
    return x, n2ijkw_np, n_np

@ti.kernel
def x_back_to_stream(stream_x:ti.template(), stream_y:ti.template(), stream_z:ti.template(), x:ti.types.ndarray(dtype=ti.f32), n2ijkw:ti.types.ndarray(dtype=ti.i32), n:int):
    for i in range(n):
        index_0 = n2ijkw[i, 0]
        index_1 = n2ijkw[i, 1]
        index_2 = n2ijkw[i, 2]
        index_3 = n2ijkw[i, 3]
        if index_3 == 0:
            stream_x[index_0, index_1, index_2] = x[i]
        elif index_3 == 1:
            stream_y[index_0, index_1, index_2] = x[i]
        else:
            stream_z[index_0, index_1, index_2] = x[i]


