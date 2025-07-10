import taichi as ti
import math
import numpy as np
from taichi_utils import *
# from mesh_interpolation import ray_box_intersect_3d
# Define necessary data structures outside the class
@ti.dataclass
class AABB:
    min: ti.types.vector(3, ti.f32)
    max: ti.types.vector(3, ti.f32)

@ti.dataclass
class BVHNode:
    aabb: AABB
    left: ti.i32  # Index of left child node
    right: ti.i32  # Index of right child node
    parent: ti.i32  # Index of parent node
    is_leaf: ti.i32  # 1 if leaf node, 0 otherwise
    triangle_index: ti.i32  # Index of the triangle (for leaf nodes)


@ti.data_oriented
class LBVH:
    def __init__(self, vertices, faces, vn, fn):
        
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=vn)
        self.new_vertices = ti.Vector.field(3, dtype=ti.f32, shape=vn)
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=fn)
        copy_to1(vertices, self.vertices)  # Taichi field of shape (num_vertices, 3)
        copy_to1(faces, self.faces)
        # self.new_vertices = vertices
        # self.faces = faces        # Taichi field of shape (num_faces, 3)
        self.n = self.faces.shape[0]
        
        self.max_stack_size = 1024
        self.max_candidate_num = 16
        # Data structures
        self.triangle_aabbs = AABB.field(shape=self.n)
        self.centroids = ti.Vector.field(3, dtype=ti.f32, shape=self.n)
        self.morton_codes = ti.field(dtype=ti.u32, shape=self.n)
        self.sorted_indices = ti.field(dtype=ti.i32, shape=self.n)
        (self.parent_indices, self.left_indices, self.right_indices, 
         self.level_indices, self.level_start_end, 
         self.total_nodes, self.num_levels, self.root_idx) = self.precompute_tree_indices(self.n)
        self.parent_indices_ti = ti.field(dtype=ti.i32,shape=self.total_nodes)
        self.left_indices_ti = ti.field(dtype=ti.i32,shape=self.total_nodes)
        self.right_indices_ti = ti.field(dtype=ti.i32,shape=self.total_nodes)
        self.level_indices_ti = ti.field(dtype=ti.i32,shape=self.total_nodes)
        self.level_start_end_ti = ti.field(dtype=ti.i32,shape=(self.num_levels+1, 2))
        
        # print(self.parent_indices)
        self.parent_indices_ti.from_numpy(self.parent_indices)
        self.left_indices_ti.from_numpy(self.left_indices)
        self.right_indices_ti.from_numpy(self.right_indices)
        self.level_indices_ti.from_numpy(self.level_indices)
        self.level_start_end_ti.from_numpy(self.level_start_end)

        self.bvh_nodes = BVHNode.field(shape=self.total_nodes)  # Will be initialized after computing total nodes

        self.changed_triangles = ti.field(dtype=ti.i32, shape=self.n)
        self.needs_update = ti.field(dtype=ti.i32, shape=self.total_nodes)
        self.triangle_to_leaf_idx = ti.field(dtype=ti.i32, shape=self.n)
        # self.candidate_list = ti.field(dtype=ti.i32, shape=(self.total_nodes,))
        # self.candidate_count = ti.field(dtype=ti.i32, shape=())

        # Build the BVH tree
        self.build_bvh_tree()

    def precompute_tree_indices(self, n):
        # Nodes will be stored in a list, with their indices assigned as their position in the list
        class Node:
            def __init__(self, left=-1, right=-1, level=0):
                self.parent = -1
                self.left = left
                self.right = right
                self.level = level

        all_nodes = []

        # Initialize leaf nodes
        for i in range(n):
            node = Node()
            all_nodes.append(node)

        current_level_indices = [i for i in range(n)]
        level = 0
        level_start_end = []
        while len(current_level_indices) > 1:
            # print(current_level_indices)
            level_start_end.append([current_level_indices[0], current_level_indices[-1]])
            for curr_level_idx in current_level_indices:
                all_nodes[curr_level_idx].level = level
            level += 1
            next_level_indices = []
            i = 0
            while i < len(current_level_indices):
                left_idx = current_level_indices[i]
                i += 1
                if i < len(current_level_indices):
                    right_idx = current_level_indices[i]
                    i += 1
                else:
                    right_idx = -1  # Handle the unpaired node by promoting it to the next level

                parent_node = Node(left=left_idx, right=right_idx)
                parent_idx = len(all_nodes)
                all_nodes.append(parent_node)

                all_nodes[left_idx].parent = parent_idx
                if right_idx != -1:
                    all_nodes[right_idx].parent = parent_idx

                next_level_indices.append(parent_idx)

            current_level_indices = next_level_indices

        level_start_end.append([current_level_indices[0], current_level_indices[-1]])
        # The last node is the root
        root_idx = current_level_indices[0]
        all_nodes[root_idx].parent = -1  # Root has no parent
        all_nodes[root_idx].level = level  # Root has no parent

        total_nodes = len(all_nodes)
        parent_indices = [-1] * total_nodes
        left_indices = [-1] * total_nodes
        right_indices = [-1] * total_nodes
        levels_indices = [-1] * total_nodes

        for idx, node in enumerate(all_nodes):
            parent_indices[idx] = node.parent
            left_indices[idx] = node.left
            right_indices[idx] = node.right
            levels_indices[idx] = node.level

        return np.array(parent_indices), np.array(left_indices), np.array(right_indices), np.array(levels_indices), np.array(level_start_end), total_nodes, level, root_idx
    
    def build_bvh_tree(self):
        # copy_to(vertices, self.vertices)
        self.compute_triangle_aabbs()
        self.compute_morton_codes()
        self.sort_triangles()
        self.construct_bvh()

    def update_bvh_tree(self, new_vertices):
        copy_to1(new_vertices, self.new_vertices)

        self.changed_triangles.fill(0)
        self.mark_changed_triangles()
        copy_to1(new_vertices, self.vertices)

        self.update_triangle_aabbs()

        self.needs_update.fill(0)
        self.mark_needs_update()

        self.update_leaf_nodes()
        for level in range(1, self.num_levels + 1):
            # level_nodes_start = self.level_start_end_ti[level, 0]
            # level_nodes_end = self.level_start_end_ti[level, 1]
            self.update_internal_nodes(self.level_start_end[level][0], self.level_start_end[level][1])


    @ti.kernel
    def compute_triangle_aabbs(self):
        for i in self.triangle_aabbs:
            idx0 = self.faces[i][0]
            idx1 = self.faces[i][1]
            idx2 = self.faces[i][2]
    
            v0 = self.vertices[idx0]
            v1 = self.vertices[idx1]
            v2 = self.vertices[idx2]
    
            min_corner = ti.min(ti.min(v0, v1), v2)
            max_corner = ti.max(ti.max(v0, v1), v2)
    
            self.triangle_aabbs[i].min = min_corner
            self.triangle_aabbs[i].max = max_corner
    
    @ti.func
    def expand_bits(self, v):
        v = ti.min(ti.max(v * 1024.0, 0.0), 1023.0)
        x = ti.cast(v, ti.u32)
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x << 8)) & 0x0300F00F
        x = (x | (x << 4)) & 0x030C30C3
        x = (x | (x << 2)) & 0x09249249
        return x
    
    @ti.kernel
    def compute_morton_codes(self):
        # Compute scene bounding box
        scene_min = ti.Vector([ti.f32(1e30), ti.f32(1e30), ti.f32(1e30)])
        scene_max = ti.Vector([ti.f32(-1e30), ti.f32(-1e30), ti.f32(-1e30)])
        for i in self.triangle_aabbs:
            ti.atomic_min(scene_min[0], self.triangle_aabbs[i].min[0])
            ti.atomic_min(scene_min[1], self.triangle_aabbs[i].min[1])
            ti.atomic_min(scene_min[2], self.triangle_aabbs[i].min[2])
            ti.atomic_max(scene_max[0], self.triangle_aabbs[i].max[0])
            ti.atomic_max(scene_max[1], self.triangle_aabbs[i].max[1])
            ti.atomic_max(scene_max[2], self.triangle_aabbs[i].max[2])
        extent = scene_max - scene_min + 1e-5  # Avoid division by zero
    
        # Compute centroids and Morton codes
        for i in self.centroids:
            centroid = (self.triangle_aabbs[i].min + self.triangle_aabbs[i].max) * 0.5
            self.centroids[i] = centroid
    
            # Normalize centroid to [0, 1]
            normalized = (centroid - scene_min) / extent
            normalized = ti.min(ti.max(normalized, 0.0), 1.0)
    
            # Compute Morton code
            x_bits = self.expand_bits(normalized.x)
            y_bits = self.expand_bits(normalized.y)
            z_bits = self.expand_bits(normalized.z)
            self.morton_codes[i] = (x_bits << 2) + (y_bits << 1) + z_bits
    
    def sort_triangles(self):
        # For simplicity, using Python's sorted function
        indices = [i for i in range(self.n)]
        codes = [self.morton_codes[i] for i in range(self.n)]
    
        sorted_pairs = sorted(zip(codes, indices))
        for i in range(self.n):
            self.sorted_indices[i] = sorted_pairs[i][1]
    
    def construct_bvh(self):
        self.build_leaf_nodes()
        for i in ti.static(range(1, self.num_levels+1)):
            self.build_internal_nodes(self.level_start_end[i][0], self.level_start_end[i][1])
    
    @ti.kernel
    def build_leaf_nodes(self):
        for i in range(self.n):
            idx = self.sorted_indices[i]
            leaf_idx = i
            self.bvh_nodes[leaf_idx].aabb = self.triangle_aabbs[idx]
            self.bvh_nodes[leaf_idx].is_leaf = 1
            self.bvh_nodes[leaf_idx].triangle_index = idx
            self.bvh_nodes[leaf_idx].left = -1
            self.bvh_nodes[leaf_idx].right = -1
            self.bvh_nodes[leaf_idx].parent = -1
            self.triangle_to_leaf_idx[idx] = leaf_idx
    
    @ti.kernel
    def build_internal_nodes(self, start_idx:int, end_idx:int):
        # Build internal nodes for this level
        # this level contains xx number of nodes
        # num_nodes = 2**(level - 1)
        # base_parent_idx = num_nodes - 1
        # base_children_idx = base_parent_idx * 2 + 1
        for i in range(start_idx, end_idx+1):
            node_idx = i
            left_idx = self.left_indices_ti[i]
            right_idx = self.right_indices_ti[i]
            # parent_idx = self.parent_indices[i]

            self.bvh_nodes[node_idx].left = left_idx
            self.bvh_nodes[node_idx].right = right_idx
            self.bvh_nodes[node_idx].is_leaf = 0
            self.bvh_nodes[node_idx].triangle_index = -1

            if left_idx != -1:
                self.bvh_nodes[left_idx].parent = node_idx
            if right_idx != -1:
                self.bvh_nodes[right_idx].parent = node_idx

            # Compute AABB for the internal node
            if left_idx != -1 and right_idx != -1:
                left_aabb = self.bvh_nodes[left_idx].aabb
                right_aabb = self.bvh_nodes[right_idx].aabb
                self.bvh_nodes[node_idx].aabb.min = ti.min(left_aabb.min, right_aabb.min)
                self.bvh_nodes[node_idx].aabb.max = ti.max(left_aabb.max, right_aabb.max)
            elif left_idx != -1:
                left_aabb = self.bvh_nodes[left_idx].aabb
                self.bvh_nodes[node_idx].aabb.min = left_aabb.min
                self.bvh_nodes[node_idx].aabb.max = left_aabb.max
            elif right_idx != -1:
                right_aabb = self.bvh_nodes[right_idx].aabb
                self.bvh_nodes[node_idx].aabb.min = right_aabb.min
                self.bvh_nodes[node_idx].aabb.max = right_aabb.max


    @ti.kernel
    def mark_changed_triangles(self):
        for i in range(self.n):
            idx0 = self.faces[i][0]
            idx1 = self.faces[i][1]
            idx2 = self.faces[i][2]
            if ((self.new_vertices[idx0] - self.vertices[idx0]).norm()>1e-14 
                or (self.new_vertices[idx1] - self.vertices[idx1]).norm()>1e-14
                or (self.new_vertices[idx2] - self.vertices[idx2]).norm()>1e-14):
                self.changed_triangles[i] = 1


    @ti.kernel
    def update_triangle_aabbs(self):
        for i in range(self.n):
            if self.changed_triangles[i]:
                idx0 = self.faces[i][0]
                idx1 = self.faces[i][1]
                idx2 = self.faces[i][2]
        
                v0 = self.vertices[idx0]
                v1 = self.vertices[idx1]
                v2 = self.vertices[idx2]
        
                min_corner = ti.min(ti.min(v0, v1), v2)
                max_corner = ti.max(ti.max(v0, v1), v2)
        
                self.triangle_aabbs[i].min = min_corner
                self.triangle_aabbs[i].max = max_corner

    @ti.kernel
    def mark_needs_update(self):
        for i in range(self.n):
            if self.changed_triangles[i]:
                leaf_idx = self.triangle_to_leaf_idx[i]
                self.needs_update[leaf_idx] = 1

    @ti.kernel
    def update_leaf_nodes(self):
        for i in range(self.n): # i is the node idx
            idx = self.sorted_indices[i]
            if self.needs_update[i]:
                self.bvh_nodes[i].aabb = self.triangle_aabbs[idx]
                parent_idx = self.bvh_nodes[i].parent
                if parent_idx != -1:
                    self.needs_update[parent_idx] = 1

    @ti.kernel
    def update_internal_nodes(self, start_idx: ti.i32, end_idx: ti.i32):
        for i in range(start_idx, end_idx + 1):
            if self.needs_update[i]:
                if not self.bvh_nodes[i].is_leaf:
                    left_idx = self.bvh_nodes[i].left
                    right_idx = self.bvh_nodes[i].right
                    if left_idx != -1 and right_idx != -1:
                        left_aabb = self.bvh_nodes[left_idx].aabb
                        right_aabb = self.bvh_nodes[right_idx].aabb
                        self.bvh_nodes[i].aabb.min = ti.min(left_aabb.min, right_aabb.min)
                        self.bvh_nodes[i].aabb.max = ti.max(left_aabb.max, right_aabb.max)
                    elif left_idx != -1:
                        left_aabb = self.bvh_nodes[left_idx].aabb
                        self.bvh_nodes[i].aabb.min = left_aabb.min
                        self.bvh_nodes[i].aabb.max = left_aabb.max
                    elif right_idx != -1:
                        right_aabb = self.bvh_nodes[right_idx].aabb
                        self.bvh_nodes[i].aabb.min = right_aabb.min
                        self.bvh_nodes[i].aabb.max = right_aabb.max
                parent_idx = self.bvh_nodes[i].parent
                if parent_idx != -1:
                    self.needs_update[parent_idx] = 1


    @ti.func
    def aabb_overlap(self, aabb1: AABB, aabb2_min: ti.types.vector(3, ti.f32), aabb2_max: ti.types.vector(3, ti.f32)) -> ti.i32:
        overlap = True
        for i in ti.static(range(3)):
            if aabb1.max[i] < aabb2_min[i] or aabb1.min[i] > aabb2_max[i]:
                overlap = False
        return overlap

    @ti.func
    def traverse_node(self, query_min: ti.types.vector(3, ti.f32), query_max: ti.types.vector(3, ti.f32)):
        candidate_count = 0
        # Initialize the stack
        # stack = ti.Array.field(dtype=ti.i32, shape=self.max_stack_size)

        stack = ti.Vector.zero(int, self.max_stack_size)
        candidate_list = ti.Vector.zero(int, self.max_candidate_num)
        # stack.fill(-1)
        # candidate_list.fill(-1)
        stack_ptr = 0

        # Push the root node index onto the stack
        stack[stack_ptr] = self.root_idx
        stack_ptr += 1

        while stack_ptr > 0:
            # print("hi")
            # Pop a node from the stack
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            node = self.bvh_nodes[node_idx]

            # Check overlap with the query AABB
            if self.aabb_overlap(node.aabb, query_min, query_max):
                if node.is_leaf == 1:
                    idx = candidate_count
                    candidate_count += 1
                    # if idx < self.max_candidates:
                    candidate_list[idx] = node.triangle_index
                else:
                    # Push child nodes onto the stack
                    if node.left != -1:
                        if stack_ptr < self.max_stack_size:
                            stack[stack_ptr] = node.left
                            stack_ptr += 1
                    if node.right != -1:
                        if stack_ptr < self.max_stack_size:
                            stack[stack_ptr] = node.right
                            stack_ptr += 1
        
        return candidate_count, candidate_list


    @ti.func
    def get_candidate_triangles_box(self, query_min: ti.types.vector(3, ti.f32), query_max: ti.types.vector(3, ti.f32)):
        # self.candidate_count[None] = 0
        candidate_count, candidate_list = self.traverse_node(query_min, query_max)
        return candidate_count, candidate_list

    @ti.func
    def ray_box_intersect_3d(self, ray_origin, ray_direction, t0, t1, box_min, box_max) -> int:
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
    def traverse_node_ray(self, ray_origin: ti.types.vector(3, ti.f32), ray_direction: ti.types.vector(3, ti.f32), t0:ti.f32, t1:ti.f32):
        candidate_count = 0
        # Initialize the stack
        # stack = ti.Array.field(dtype=ti.i32, shape=self.max_stack_size)

        stack = ti.Vector.zero(int, self.max_stack_size)
        candidate_list = ti.Vector.zero(int, self.max_candidate_num)
        # stack.fill(-1)
        # candidate_list.fill(-1)
        stack_ptr = 0

        # Push the root node index onto the stack
        stack[stack_ptr] = self.root_idx
        stack_ptr += 1

        while stack_ptr > 0:
            # print("hi")
            # Pop a node from the stack
            stack_ptr -= 1
            node_idx = stack[stack_ptr]
            node = self.bvh_nodes[node_idx]

            # Check overlap with the query AABB
            if self.ray_box_intersect_3d(ray_origin, ray_direction, t0, t1, node.aabb.min, node.aabb.max):
                if node.is_leaf == 1:
                    idx = candidate_count
                    candidate_count += 1
                    # if idx < self.max_candidates:
                    candidate_list[idx] = node.triangle_index
                else:
                    # Push child nodes onto the stack
                    if node.left != -1:
                        if stack_ptr < self.max_stack_size:
                            stack[stack_ptr] = node.left
                            stack_ptr += 1
                    if node.right != -1:
                        if stack_ptr < self.max_stack_size:
                            stack[stack_ptr] = node.right
                            stack_ptr += 1
        
        return candidate_count, candidate_list

    @ti.func
    def get_candidate_triangles_ray(self, ray_origin: ti.types.vector(3, ti.f32), ray_direction: ti.types.vector(3, ti.f32), t0:ti.f32, t1:ti.f32):
        # self.candidate_count[None] = 0
        candidate_count, candidate_list = self.traverse_node_ray(ray_origin, ray_direction, t0, t1)
        return candidate_count, candidate_list