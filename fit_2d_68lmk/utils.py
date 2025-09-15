import numpy as np
import torch
import cv2
from typing import Optional, Tuple


def save_obj(obj_path: str, vertices: np.ndarray, triangles: Optional[np.ndarray] = None) -> None:
    """
    保存3D模型为OBJ格式文件
    
    参数:
        obj_path: 输出OBJ文件路径
        vertices: 顶点坐标数组，形状为(n_vertices, 3)
        triangles: 三角形面片索引数组，形状为(n_triangles, 3)，可选
    """
    # 验证顶点数据格式
    if not isinstance(vertices, np.ndarray):
        raise TypeError("vertices必须是numpy数组")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices形状必须为(n_vertices, 3)，实际为{vertices.shape}")
    
    # 写入OBJ文件
    with open(obj_path, 'w', encoding='utf-8') as f:
        # 写入顶点数据
        f.write("# 顶点数据 (x, y, z)\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        # 写入面片数据（若提供）
        if triangles is not None:
            # 验证面片数据格式
            if not isinstance(triangles, np.ndarray):
                raise TypeError("triangles必须是numpy数组")
            if triangles.ndim != 2 or triangles.shape[1] != 3:
                raise ValueError(f"triangles形状必须为(n_triangles, 3)，实际为{triangles.shape}")
            
            # 转换为整数索引
            triangles = triangles.astype(int)
            
            # OBJ格式索引从1开始，自动适配0基索引输入
            if np.min(triangles) == 0:
                triangles += 1
                f.write("\n# 面片数据（已自动将0基索引转换为1基索引）\n")
            else:
                f.write("\n# 面片数据\n")
            
            for tri in triangles:
                f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    
    print(f"OBJ文件已保存至: {obj_path}")


def project_3d_to_2d(
    points_3d: torch.Tensor, 
    R: torch.Tensor, 
    t: torch.Tensor, 
    K: torch.Tensor
) -> torch.Tensor:
    """
    将3D点从世界坐标系投影到2D图像平面
    
    参数:
        points_3d: 3D点坐标，形状为(B, N, 3)，B为批次大小，N为点数量
        R: 旋转矩阵，形状为(B, 3, 3)
        t: 平移向量，形状为(B, 3)
        K: 相机内参矩阵，形状为(B, 3, 3)
    
    返回:
        points_2d: 投影后的2D坐标，形状为(B, N, 2)
    """
    # 确保所有张量在同一设备
    device = points_3d.device
    R = R.to(device)
    t = t.to(device)
    K = K.to(device)
    
    batch_size, num_points = points_3d.shape[0], points_3d.shape[1]
    
    # 1. 世界坐标系 -> 相机坐标系: P_cam = R @ P_world + t
    points_cam = torch.bmm(R, points_3d.permute(0, 2, 1))  # (B, 3, N)
    points_cam = points_cam.permute(0, 2, 1)  # (B, N, 3)
    points_cam += t.unsqueeze(1)  # (B, N, 3)
    
    # 2. 透视投影: (x, y, z) -> (u, v)
    x, y, z = points_cam[..., 0], points_cam[..., 1], points_cam[..., 2]
    
    # 防止除零错误（添加微小值）
    z = z + 1e-8
    
    # 提取相机内参
    fx = K[..., 0, 0].unsqueeze(1)  # (B, 1)
    fy = K[..., 1, 1].unsqueeze(1)  # (B, 1)
    cx = K[..., 0, 2].unsqueeze(1)  # (B, 1)
    cy = K[..., 1, 2].unsqueeze(1)  # (B, 1)
    
    # 计算2D坐标
    u = fx * x / z + cx  # (B, N)
    v = fy * y / z + cy  # (B, N)
    
    return torch.stack([u, v], dim=-1)  # (B, N, 2)


def solve_pnp_opencv(
    points_3d: np.ndarray, 
    points_2d: np.ndarray, 
    camera_matrix: np.ndarray, 
    dist_coeffs: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    使用OpenCV的solvePnP求解3D到2D的位姿估计
    
    参数:
        points_3d: 3D点坐标，形状为(n, 3)
        points_2d: 对应2D点坐标，形状为(n, 2)
        camera_matrix: 相机内参矩阵，形状为(3, 3)
        dist_coeffs: 畸变系数，形状为(5, 1)，默认为零畸变
    
    返回:
        rvec: 旋转向量，形状为(3, 1)
        tvec: 平移向量，形状为(3, 1)
        R: 旋转矩阵，形状为(3, 3)
        projected_points: 重投影的2D点，形状为(n, 2)
    """
    # 转换输入为numpy数组并验证形状
    points_3d = np.asarray(points_3d, dtype=np.float32)
    points_2d = np.asarray(points_2d, dtype=np.float32)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float32)
    
    if points_3d.shape[0] != points_2d.shape[0]:
        raise ValueError(f"3D点与2D点数量不匹配: {points_3d.shape[0]} vs {points_2d.shape[0]}")
    if points_3d.shape[1] != 3:
        raise ValueError(f"3D点形状应为(n, 3)，实际为{points_3d.shape}")
    if points_2d.shape[1] != 2:
        raise ValueError(f"2D点形状应为(n, 2)，实际为{points_2d.shape}")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"相机内参形状应为(3, 3)，实际为{camera_matrix.shape}")
    
    # 设置默认畸变系数
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)
    
    # 求解PnP
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=points_3d,
        imagePoints=points_2d,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE  # 迭代方法，对噪声更鲁棒
    )
    
    if not success:
        raise RuntimeError("PnP求解失败，请检查输入点和相机参数的有效性")
    
    # 旋转向量转旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    # 验证：计算重投影点
    projected_points, _ = cv2.projectPoints(
        objectPoints=points_3d,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    projected_points = projected_points.squeeze()  # 形状(n, 2)
    
    return rvec, tvec, R, projected_points


def draw_keypoints_on_image(
    image: np.ndarray, 
    keypoints: np.ndarray, 
    point_color: Tuple[int, int, int] = (255, 0, 0), 
    point_size: int = 3
) -> np.ndarray:
    """
    在图像上绘制关键点
    
    参数:
        image: 输入图像，形状为(h, w, 3)，BGR格式
        keypoints: 关键点坐标，形状为(n, 2)，格式为(x, y)
        point_color: 关键点颜色，BGR格式，默认为红色(255, 0, 0)
        point_size: 关键点大小，默认为3
    
    返回:
        draw_img: 绘制了关键点的图像
    """
    # 验证输入
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"图像形状必须为(h, w, 3)，实际为{image.shape}")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"关键点形状必须为(n, 2)，实际为{keypoints.shape}")
    
    # 复制图像避免修改原图
    draw_img = image.copy()
    
    # 绘制每个关键点
    for (x, y) in keypoints:
        # 转换为整数坐标
        x_int = int(round(x))
        y_int = int(round(y))
        # 绘制填充圆
        cv2.circle(
            img=draw_img,
            center=(x_int, y_int),
            radius=point_size,
            color=point_color,
            thickness=-1  # -1表示填充圆
        )
    
    return draw_img
    