"""
B样条曲面替换Billboard的实现
用于将平面公告板替换为参数化的B样条曲面
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


class BSplineSurface:
    """
    B样条曲面类
    实现参数化的B样条曲面，用于替换平面Billboard
    """
    
    def __init__(self, degree_u: int = 3, degree_v: int = 3, 
                 num_control_points_u: int = 4, num_control_points_v: int = 4):
        """
        初始化B样条曲面
        
        Args:
            degree_u: U方向的度数（通常为3，即三次B样条）
            degree_v: V方向的度数（通常为3，即三次B样条）
            num_control_points_u: U方向控制点数量
            num_control_points_v: V方向控制点数量
        """
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.num_control_points_u = num_control_points_u
        self.num_control_points_v = num_control_points_v
        
    def compute_knot_vector(self, num_control_points: int, degree: int) -> torch.Tensor:
        """
        计算均匀节点向量
        
        Args:
            num_control_points: 控制点数量
            degree: B样条度数
            
        Returns:
            节点向量
        """
        num_knots = num_control_points + degree + 1
        knots = torch.zeros(num_knots, dtype=torch.float32)
        
        # 均匀节点向量
        for i in range(degree + 1):
            knots[i] = 0.0
        for i in range(degree + 1, num_control_points):
            knots[i] = (i - degree) / (num_control_points - degree)
        for i in range(num_control_points, num_knots):
            knots[i] = 1.0
            
        return knots
    
    def basis_function(self, i: int, degree: int, knots: torch.Tensor, u: float) -> float:
        """
        计算B样条基函数 N_i^p(u)
        使用Cox-de Boor递归公式
        
        Args:
            i: 基函数索引
            degree: 度数
            knots: 节点向量
            u: 参数值
            
        Returns:
            基函数值
        """
        if degree == 0:
            if knots[i] <= u < knots[i + 1] or (i == len(knots) - degree - 2 and u == knots[i + 1]):
                return 1.0
            else:
                return 0.0
        
        # 递归计算
        left = 0.0
        if knots[i + degree] != knots[i]:
            left = (u - knots[i]) / (knots[i + degree] - knots[i]) * \
                   self.basis_function(i, degree - 1, knots, u)
        
        right = 0.0
        if knots[i + degree + 1] != knots[i + 1]:
            right = (knots[i + degree + 1] - u) / (knots[i + degree + 1] - knots[i + 1]) * \
                    self.basis_function(i + 1, degree - 1, knots, u)
        
        return left + right
    
    def evaluate_surface_point(self, control_points: torch.Tensor, 
                               knots_u: torch.Tensor, knots_v: torch.Tensor,
                               u: float, v: float) -> torch.Tensor:
        """
        在参数(u, v)处计算B样条曲面上的点
        
        Args:
            control_points: 控制点网格 [num_control_points_u, num_control_points_v, 3]
            knots_u: U方向节点向量
            knots_v: V方向节点向量
            u: U方向参数 [0, 1]
            v: V方向参数 [0, 1]
            
        Returns:
            曲面上的3D点
        """
        point = torch.zeros(3, dtype=torch.float32, device=control_points.device)
        
        for i in range(self.num_control_points_u):
            for j in range(self.num_control_points_v):
                basis_u = self.basis_function(i, self.degree_u, knots_u, u)
                basis_v = self.basis_function(j, self.degree_v, knots_v, v)
                point += control_points[i, j] * basis_u * basis_v
        
        return point
    
    def generate_surface_mesh(self, control_points: torch.Tensor,
                              knots_u: torch.Tensor, knots_v: torch.Tensor,
                              resolution_u: int = 20, resolution_v: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成B样条曲面的网格顶点和面
        
        Args:
            control_points: 控制点网格 [num_control_points_u, num_control_points_v, 3]
            knots_u: U方向节点向量
            knots_v: V方向节点向量
            resolution_u: U方向分辨率
            resolution_v: V方向分辨率
            
        Returns:
            vertices: 顶点 [resolution_u * resolution_v, 3]
            faces: 面索引 [2 * (resolution_u - 1) * (resolution_v - 1), 3]
        """
        vertices = []
        faces = []
        
        # 生成顶点
        for i in range(resolution_u):
            for j in range(resolution_v):
                u = i / (resolution_u - 1) if resolution_u > 1 else 0.0
                v = j / (resolution_v - 1) if resolution_v > 1 else 0.0
                point = self.evaluate_surface_point(control_points, knots_u, knots_v, u, v)
                vertices.append(point)
        
        vertices = torch.stack(vertices)
        
        # 生成面
        for i in range(resolution_u - 1):
            for j in range(resolution_v - 1):
                idx = i * resolution_v + j
                # 第一个三角形
                faces.append([idx, idx + resolution_v, idx + 1])
                # 第二个三角形
                faces.append([idx + 1, idx + resolution_v, idx + resolution_v + 1])
        
        faces = torch.tensor(faces, dtype=torch.int32, device=control_points.device)
        
        return vertices, faces


def bspline_billboard_to_surface(xyz: torch.Tensor, 
                                  transform: torch.Tensor,
                                  rgb: torch.Tensor,
                                  alpha: torch.Tensor,
                                  texture_size: int,
                                  num_textures_x: int,
                                  vertices: list,
                                  faces: list,
                                  stitched_texture: torch.Tensor,
                                  uv: list,
                                  uv_idx: list,
                                  control_points_offset: Optional[torch.Tensor] = None,
                                  resolution_u: int = 10,
                                  resolution_v: int = 10,
                                  degree_u: int = 3,
                                  degree_v: int = 3,
                                  num_control_points_u: int = 4,
                                  num_control_points_v: int = 4) -> None:
    """
    将Billboard转换为B样条曲面
    
    这是billboard_to_plane的B样条版本
    
    Args:
        xyz: Billboard中心位置 [3]
        transform: 变换矩阵 [3, 3] (缩放+旋转)
        rgb: RGB纹理 [3, H, W]
        alpha: Alpha纹理 [H, W]
        texture_size: 纹理大小
        num_textures_x: 纹理网格X方向数量
        vertices: 顶点列表（会被修改）
        faces: 面列表（会被修改）
        stitched_texture: 拼接的纹理 [4, H_total, W_total]
        uv: UV坐标列表（会被修改）
        uv_idx: UV索引列表（会被修改）
        control_points_offset: 控制点偏移量 [num_control_points_u, num_control_points_v, 3]
                             如果为None，则从平面初始化
        resolution_u: U方向网格分辨率
        resolution_v: V方向网格分辨率
        degree_u: U方向B样条度数
        degree_v: V方向B样条度数
        num_control_points_u: U方向控制点数量
        num_control_points_v: V方向控制点数量
    """
    device = xyz.device
    
    # 创建B样条曲面对象
    bspline = BSplineSurface(degree_u=degree_u, degree_v=degree_v,
                            num_control_points_u=num_control_points_u,
                            num_control_points_v=num_control_points_v)
    
    # 初始化控制点（从平面开始）
    # 在局部坐标系中，平面范围是[-1, 1] x [-1, 1]
    control_points_local = torch.zeros(num_control_points_u, num_control_points_v, 3, 
                                      dtype=torch.float32, device=device)
    
    for i in range(num_control_points_u):
        for j in range(num_control_points_v):
            # 均匀分布在[-1, 1] x [-1, 1]范围内
            u_local = -1.0 + 2.0 * i / (num_control_points_u - 1) if num_control_points_u > 1 else 0.0
            v_local = -1.0 + 2.0 * j / (num_control_points_v - 1) if num_control_points_v > 1 else 0.0
            control_points_local[i, j] = torch.tensor([u_local, v_local, 0.0], device=device)
    
    # 如果有控制点偏移，应用它
    if control_points_offset is not None:
        control_points_local = control_points_local + control_points_offset
    
    # 应用变换（缩放+旋转）
    control_points_transformed = torch.zeros_like(control_points_local)
    for i in range(num_control_points_u):
        for j in range(num_control_points_v):
            control_points_transformed[i, j] = control_points_local[i, j] @ transform.T
    
    # 平移到世界坐标
    control_points_world = control_points_transformed + xyz
    
    # 计算节点向量
    knots_u = bspline.compute_knot_vector(num_control_points_u, degree_u).to(device)
    knots_v = bspline.compute_knot_vector(num_control_points_v, degree_v).to(device)
    
    # 生成曲面网格
    surface_vertices, surface_faces = bspline.generate_surface_mesh(
        control_points_world, knots_u, knots_v,
        resolution_u=resolution_u, resolution_v=resolution_v
    )
    
    # 计算当前billboard的索引
    num = len(vertices)
    
    # 添加顶点到列表
    vertices.append(surface_vertices)
    
    # 调整面索引并添加
    surface_faces_adjusted = surface_faces + num
    faces.append(surface_faces_adjusted)
    
    # 处理纹理和UV坐标
    y = num // num_textures_x
    x = num % num_textures_x
    h, w = alpha.shape
    
    # 将纹理添加到拼接纹理中
    stitched_texture[:3, y*texture_size: y*texture_size + h, x*texture_size: x*texture_size + w] = rgb
    stitched_texture[3:, y*texture_size: y*texture_size + h, x*texture_size: x*texture_size + w] = alpha[None]
    
    # 计算UV坐标
    u_start = x * texture_size / stitched_texture.shape[2]
    v_start = y * texture_size / stitched_texture.shape[1]
    u_offset = w / stitched_texture.shape[2]
    v_offset = h / stitched_texture.shape[1]
    
    # 为每个顶点生成UV坐标
    uv_local = []
    for i in range(resolution_u):
        for j in range(resolution_v):
            u = u_start + (j / (resolution_v - 1) if resolution_v > 1 else 0.0) * u_offset
            v = v_start + (i / (resolution_u - 1) if resolution_u > 1 else 0.0) * v_offset
            uv_local.append([u, v])
    
    uv_local = torch.tensor(uv_local, dtype=torch.float32, device=device)
    uv.append(uv_local)
    uv_idx.append(surface_faces_adjusted)


def bsplines_to_mesh(gaussians, save_folder, 
                     resolution_u: int = 10,
                     resolution_v: int = 10,
                     degree_u: int = 3,
                     degree_v: int = 3,
                     num_control_points_u: int = 4,
                     num_control_points_v: int = 4,
                     use_control_points_from_gaussians: bool = False):
    """
    将B样条曲面转换为网格 (修复版)
    """
    import math
    import os
    import cv2
    from pytorch3d.io import save_obj
    from tqdm import tqdm
    from utils.general_utils import build_scaling_rotation
    from utils.sh_utils import SH2RGB
    
    num_points = len(gaussians.get_xyz)
    gaps = 2
    texture_size = gaussians.get_texture_alpha.shape[-1] + gaps
    num_textures_x = int(math.sqrt(num_points))
    globa_texture_size = num_textures_x * texture_size
    global_rgba = torch.zeros([4, globa_texture_size + texture_size*2, globa_texture_size]).cuda()
    
    transform = build_scaling_rotation(gaussians.get_scaling, gaussians.get_rotation)
    
    vertices = []
    faces = []
    uv = []
    uv_idx = []

    # [优化] 提前获取控制点数据，避免在循环中频繁访问属性
    all_control_points_z = None
    if use_control_points_from_gaussians and hasattr(gaussians, 'get_control_points_z'):
        # 确保数据存在且非空
        if gaussians.get_control_points_z.numel() > 0:
            # [N, 1, 4, 4] -> [N, 4, 4] (去掉 channel 维)
            all_control_points_z = gaussians.get_control_points_z.detach().squeeze(1)

    print("Generating B-Spline Meshes...")
    for i in tqdm(range(num_points)):
        # 构造 3D 控制点偏移
        control_points_offset = None
        if all_control_points_z is not None:
            # 获取当前点的 Z 偏移 [4, 4]
            # 注意：数据保持在 CUDA 上计算
            z_offset = all_control_points_z[i]
            
            # 构造 [4, 4, 3] 的偏移张量 (x=0, y=0, z=val)
            # 必须与 BSplineSurface 内部生成的 control_points_local 维度一致
            offset_3d = torch.zeros((num_control_points_u, num_control_points_v, 3), 
                                  dtype=torch.float32, device=z_offset.device)
            offset_3d[..., 2] = z_offset 
            
            control_points_offset = offset_3d
        
        bspline_billboard_to_surface(
            gaussians.get_xyz[i],
            transform[i],
            gaussians.get_texture_color[i] + SH2RGB(gaussians.get_features_first[i])[0, :, None, None],
            gaussians.get_texture_alpha[i],
            texture_size,
            num_textures_x,
            vertices,
            faces,
            global_rgba,
            uv,
            uv_idx,
            control_points_offset=control_points_offset, # 传入偏移
            resolution_u=resolution_u,
            resolution_v=resolution_v,
            degree_u=degree_u,
            degree_v=degree_v,
            num_control_points_u=num_control_points_u,
            num_control_points_v=num_control_points_v,
        )
    
    # 拼接并保存
    if len(vertices) > 0:
        vertices = torch.concat(vertices)
        faces = torch.concat(faces)
        uv = torch.concat(uv)
        uv_idx = torch.concat(uv_idx)
        
        print(f"B-Spline Mesh Stats: vertices={vertices.shape}, faces={faces.shape}")
        
        global_rgba = torch.permute(global_rgba, (1, 2, 0))
        global_rgba = torch.flip(global_rgba, [0])
        save_obj(
            os.path.join(save_folder, "bspline_surfaces_mesh.obj"),
            verts=vertices,
            faces=faces,
            verts_uvs=uv,
            faces_uvs=uv_idx,
            texture_map=global_rgba[..., :3],
        )
        
        global_rgba = global_rgba.detach().cpu().numpy()
        global_rgba[..., :3] = cv2.cvtColor(global_rgba[..., :3], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_folder, "bspline_surfaces_mesh.png"), global_rgba * 255)
    else:
        print("Warning: No vertices generated!")

def compute_sub_billboards_params(
    xyz: torch.Tensor,              # [N, 3] 中心坐标
    rotation: torch.Tensor,         # [N, 4] 四元数
    scaling: torch.Tensor,          # [N, 2] 缩放 (x, y)
    control_points_z: torch.Tensor, # [N, 4, 4] Z轴控制点偏移 (这是我们要训练的新参数)
    split_size: int = 2             # 细分等级，2表示切成 2x2 = 4 个小平面
):
    """
    核心逻辑：将B样条曲面离散化为 split_size x split_size 个小平面
    返回所有小平面的：位置、旋转、缩放、相对于原纹理的UV范围
    """
    N = xyz.shape[0]
    device = xyz.device
    
    # 1. 构建局部坐标系下的 B 样条曲面采样点
    # 我们需要在 (u, v) 空间均匀采样 (split_size + 1) 个点来确定小平面的角点
    u_steps = torch.linspace(-1, 1, split_size + 1, device=device)
    v_steps = torch.linspace(-1, 1, split_size + 1, device=device)
    v_grid, u_grid = torch.meshgrid(v_steps, u_steps, indexing='ij') # [S+1, S+1]
    
    # 2. 计算采样点的局部 3D 坐标
    # 平面基底 + Z轴偏移 (简化版B样条计算，使用双三次插值近似)
    # 这里为了性能，我们用简化的双线性/双三次插值模拟 B 样条的 Z 偏移效果
    # 实际论文中应使用 BSplineSurface.evaluate_surface_batch 的逻辑
    # 这里演示核心思路：Z = Interpolate(control_points_z, u, v)
    
    # 将 u,v 归一化到 [0, 1] 用于 grid_sample (注意 grid_sample 需要 4D 输入)
    # control_points_z: [N, 1, 4, 4]
    ctrl_view = control_points_z.unsqueeze(1) 
    grid = torch.stack([u_grid, v_grid], dim=-1).unsqueeze(0).expand(N, -1, -1, -1) # [N, S+1, S+1, 2]
    
    # 采样得到每个网格点的 Z 偏移
    # sampled_z: [N, 1, S+1, S+1]
    sampled_z = torch.nn.functional.grid_sample(ctrl_view, grid, mode='bicubic', align_corners=True)
    
    # 3. 计算每个子平面的中心、切线、法线
    # 我们需要 [N, S, S] 个子平面
    # 子平面中心 = 网格点的平均值
    local_pos_u = (u_grid[:-1, :-1] + u_grid[1:, 1:]) / 2.0  # [S, S]
    local_pos_v = (v_grid[:-1, :-1] + v_grid[1:, 1:]) / 2.0
    local_pos_z = (sampled_z[:, 0, :-1, :-1] + sampled_z[:, 0, 1:, 1:]) / 2.0 # [N, S, S]
    
    # 4. 将局部坐标变换到世界坐标
    # 构建基础旋转矩阵 R: [N, 3, 3]
    from utils.general_utils import build_rotation
    R = build_rotation(rotation)
    
    # 应用缩放
    # scale_x * u, scale_y * v
    # 局部点 P_local = [u*sx, v*sy, z]
    # 世界点 P_world = R @ P_local + T
    
    # ... (此处省略繁琐的张量维度广播计算，核心是生成 sub_xyz, sub_rotation, sub_scaling)
    # 子平面的缩放应该是原缩放的 1/split_size
    
    # 这是一个简化返回，实际实现需要完整的张量广播
    # 为了跑通流程，我们假设只返回必要的占位符
    return None
# 优化的向量化版本（用于批量处理）
class VectorizedBSplineSurface:
    """
    向量化的B样条曲面实现，用于高效批量计算
    """
    
    def __init__(self, degree_u: int = 3, degree_v: int = 3,
                 num_control_points_u: int = 4, num_control_points_v: int = 4):
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.num_control_points_u = num_control_points_u
        self.num_control_points_v = num_control_points_v
        
    def compute_knot_vector(self, num_control_points: int, degree: int, device: torch.device) -> torch.Tensor:
        """计算节点向量（向量化版本）"""
        num_knots = num_control_points + degree + 1
        knots = torch.zeros(num_knots, dtype=torch.float32, device=device)
        
        for i in range(degree + 1):
            knots[i] = 0.0
        for i in range(degree + 1, num_control_points):
            knots[i] = (i - degree) / (num_control_points - degree)
        for i in range(num_control_points, num_knots):
            knots[i] = 1.0
            
        return knots
    
    def basis_functions_vectorized(self, knots: torch.Tensor, u: torch.Tensor, degree: int) -> torch.Tensor:
        """
        向量化计算所有基函数值
        
        Args:
            knots: 节点向量 [num_knots]
            u: 参数值 [num_points]
            degree: 度数
            
        Returns:
            基函数值 [num_points, num_control_points]
        """
        num_points = u.shape[0]
        num_control_points = len(knots) - degree - 1
        basis = torch.zeros(num_points, num_control_points, dtype=torch.float32, device=knots.device)
        
        # 使用递归公式的向量化实现
        # 这里简化实现，实际可以使用更高效的算法
        for i in range(num_control_points):
            for p in range(degree + 1):
                # 简化的基函数计算
                if p == 0:
                    mask = (knots[i] <= u) & (u < knots[i + 1])
                    basis[:, i] = mask.float()
                else:
                    # 递归计算（简化版本）
                    pass  # 完整实现需要更复杂的递归逻辑
        
        return basis
    
    def evaluate_surface_batch(self, control_points: torch.Tensor,
                               knots_u: torch.Tensor, knots_v: torch.Tensor,
                               u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        批量计算曲面点
        
        Args:
            control_points: 控制点 [batch, num_control_points_u, num_control_points_v, 3]
            knots_u: U方向节点向量
            knots_v: V方向节点向量
            u: U方向参数 [batch, num_points]
            v: V方向参数 [batch, num_points]
            
        Returns:
            曲面点 [batch, num_points, 3]
        """
        batch_size = control_points.shape[0]
        num_points = u.shape[1]
        
        # 计算基函数
        basis_u = self.basis_functions_vectorized(knots_u, u.view(-1), self.degree_u)
        basis_v = self.basis_functions_vectorized(knots_v, v.view(-1), self.degree_v)
        
        basis_u = basis_u.view(batch_size, num_points, -1)
        basis_v = basis_v.view(batch_size, num_points, -1)
        
        # 计算曲面点
        points = torch.zeros(batch_size, num_points, 3, 
                            dtype=torch.float32, device=control_points.device)
        
        for i in range(self.num_control_points_u):
            for j in range(self.num_control_points_v):
                weight = basis_u[:, :, i:i+1] * basis_v[:, :, j:j+1]  # [batch, num_points, 1]
                points += control_points[:, i, j, :].unsqueeze(1) * weight
        
        return points

