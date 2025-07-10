import open3d as o3d
import numpy as np
import vtk
from vtk.util import numpy_support
from scipy.ndimage import binary_fill_holes


def place_and_shift_in_larger_array(small_array, larger_shape, y_shift, x_shift):
    # Step 1: Create the larger array filled with zeros
    larger_array = np.zeros(larger_shape, dtype=small_array.dtype)

    # Step 2: Calculate the center of both arrays
    small_shape = np.array(small_array.shape)
    large_shape = np.array(larger_shape)
    
    # Calculate the centers
    small_center = small_shape // 2
    large_center = large_shape // 2
    
    # Step 3: Apply x_shift to the large center
    large_center[1] += y_shift  # Shift along the x direction
    large_center[0] += x_shift

    # Step 4: Calculate the starting and ending indices for placing the small array in the large array
    start_idx = large_center - small_center
    end_idx = start_idx + small_shape
    
    # Ensure that the indices are within bounds of the larger array
    start_idx = np.clip(start_idx, 0, large_shape)
    end_idx = np.clip(end_idx, 0, large_shape)

    large_slices = tuple(slice(start, end) for start, end in zip(start_idx, end_idx))
    small_slices = tuple(slice(max(0, -s), min(e - s, dim)) for s, e, dim in zip(start_idx, end_idx, small_shape))
    
    # Step 5: Place the small array in the larger array by slicing
    larger_array[large_slices] = small_array[small_slices]

    return larger_array

def voxelize(mesh, ref_grid_shape, ref_min_bound_y, larger_shape = (255, 127, 255), voxel_size=0.005, visualize = False):
    # mesh = mesh.translate((0, 0, 0), relative=False)
    total_length = np.max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.scale(1 / (total_length),
            center=np.array([0., 0., 0.]))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                                voxel_size=voxel_size)
    min_bound = voxel_grid.get_min_bound()
    max_bound = voxel_grid.get_max_bound()
    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    y_offset = (grid_shape[1] - ref_grid_shape[1])//2
    if mesh.get_min_bound()[1] <= ref_min_bound_y:
        y_offset *= -1
    voxel_array = np.zeros(grid_shape, dtype=np.uint8)
    for voxel in voxel_grid.get_voxels():
        voxel_index = voxel.grid_index
        if (0 <= voxel_index[0] < grid_shape[0] and
            0 <= voxel_index[1] < grid_shape[1] and
            0 <= voxel_index[2] < grid_shape[2]):
            voxel_array[voxel_index[0], voxel_index[1], voxel_index[2]] = 1

    voxel_array = binary_fill_holes(voxel_array).astype(np.uint8)

    larger_shape = (255, 127, 255)
    result_array = place_and_shift_in_larger_array(voxel_array, larger_shape, y_offset, 0)

    return result_array