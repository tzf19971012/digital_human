import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import face_alignment
from FLAME.flame import FLAME
from utils import draw_keypoints_on_image, solve_pnp_opencv, project_3d_to_2d
from pytorch3d.transforms import axis_angle_to_matrix

# ----------------------------
# 配置参数设置
# ----------------------------
class Config:
    def __init__(self):
        # 路径配置
        self.image_path = r'/mnt/d/Datasets/Brad/001_c04300ef.JPG'
        self.flame_model_path = r"/home/tianzefan/human/model/generic_model.pkl"
        self.landmark_embedding_path = r"/home/tianzefan/human/model/flame_static_embedding_68.pkl"
        self.log_dir = r"runs/exp1"
        self.output_dir = r'/home/tianzefan/digital__human/test_imgs'
        
        # 训练参数
        self.learning_rate = 0.001
        self.iterations = 3000
        self.log_interval = 100  # 日志记录间隔
        self.save_interval = 100  # 图像保存间隔
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# 初始化与准备工作
# ----------------------------
def initialize(config):
    """初始化所有必要组件：创建目录、加载模型、检测关键点"""
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)
    print(f"TensorBoard日志将保存至: {config.log_dir}")
    
    # 加载图像
    input_img = cv2.imread(config.image_path)
    if input_img is None:
        raise FileNotFoundError(f"无法加载图像: {config.image_path}")
    print(f"成功加载图像，尺寸: {input_img.shape[:2]}")
    
    # 检测人脸关键点
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False,
        device=str(config.device)
    )
    preds = fa.get_landmarks(input_img)
    if preds is None or len(preds) == 0:
        raise ValueError("未检测到人脸关键点")
    preds = preds[0].astype(np.float32)  # 取第一个检测到的人脸
    
    # 绘制原始关键点
    draw_img = draw_keypoints_on_image(input_img, preds, point_color=(0, 255, 0))
    
    # 相机内参
    h, w = input_img.shape[:2]
    camera_intrinsics = torch.tensor([
        [h, 0.0, w/2],
        [0, w, h/2],
        [0, 0, 1]
    ], device=config.device, dtype=torch.float32)
    
    # 初始化FLAME模型
    flamelayer = FLAME(
        config.flame_model_path, 
        config.landmark_embedding_path
    ).to(config.device)
    
    return {
        "input_img": input_img,
        "draw_img": draw_img,
        "preds": preds,
        "flamelayer": flamelayer,
        "camera_intrinsics": camera_intrinsics,
        "writer": writer
    }

# ----------------------------
# 模型参数优化
# ----------------------------
def optimize_parameters(config, init_data):
    """优化FLAME模型参数以拟合检测到的关键点"""
    # 解包初始化数据
    flamelayer = init_data["flamelayer"]
    preds = init_data["preds"]
    draw_img = init_data["draw_img"]
    camera_intrinsics = init_data["camera_intrinsics"]
    writer = init_data["writer"]
    
    # 初始化FLAME参数
    shape_params = torch.zeros(1, 100, device=config.device)
    pose_params = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        device=config.device
    )
    expression_params = torch.zeros(1, 50, dtype=torch.float32, device=config.device)
    
    # 初始PNP求解
    rvec, tvec, _, _ = solve_pnp_opencv(
        flamelayer(shape_params, expression_params, pose_params)[1][0].cpu(),
        preds,
        camera_intrinsics.cpu()
    )
    
    target_2dlmk = torch.tensor(preds, device=config.device, dtype=torch.float32)
    # 定义可学习参数
    pose_params_tensor = torch.nn.Parameter(pose_params)
    expression_params_tensor = torch.nn.Parameter(expression_params)
    shape_params_tensor = torch.nn.Parameter(shape_params)
    rot = torch.nn.Parameter(torch.tensor(rvec[:,0], device=config.device)) # 初始旋转
    trans = torch.nn.Parameter(torch.tensor(tvec[:,0], device=config.device)) # 初始平移
    # 优化器与学习率调度器
    optimizer = optim.Adam(
        [pose_params_tensor, expression_params_tensor, 
         shape_params_tensor, rot, trans],
        lr=config.learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, verbose=True
    )
    
    # 跟踪最佳损失
    best_loss = float('inf')
    best_params = None
    
    # 优化循环
    for i in range(config.iterations):
        optimizer.zero_grad()
        
        # 前向传播
        vertice, landmark = flamelayer(
            shape_params_tensor, 
            expression_params_tensor, 
            pose_params_tensor
        )
        
        # 投影3D关键点到2D图像平面
        projected_2d = project_3d_to_2d(
            landmark, 
            axis_angle_to_matrix(rot).unsqueeze(0).float(), 
            trans.unsqueeze(0).float(), 
            camera_intrinsics.unsqueeze(0)
        )
        
        # 计算损失
        loss = torch.mean(torch.norm(projected_2d[0] - target_2dlmk, dim=-1))
        
        # 反向传播与优化
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # 记录与可视化
        if (i + 1) % config.log_interval == 0:
            # 打印损失信息
            print(f"迭代 {i+1}/{config.iterations}, 损失: {loss.item():.6f}")
            
            # 记录到TensorBoard
            writer.add_scalar("Train/Loss", loss.item(), i)
            # 记录3D网格到TensorBoard
            writer.add_mesh(
                "FLAME_Mesh", 
                vertices=vertice.detach().cpu(), 
                faces=torch.tensor(flamelayer.faces).unsqueeze(0).repeat(vertice.shape[0],1,1),
                global_step=i
            )
        
        # 保存中间结果
        if (i + 1) % config.save_interval == 0:
            # 绘制并保存关键点对比图
            draw_img_tmp = draw_keypoints_on_image(
                draw_img.copy(), 
                projected_2d[0].detach().cpu().numpy(), 
                point_color=(255, 0, 0), 
                point_size=2
            )
            save_path = os.path.join(config.output_dir, f"iter_{i+1}.png")
            cv2.imwrite(save_path, draw_img_tmp)
            print(f"中间结果已保存至: {save_path}")
            
            
        
        # 更新最佳参数
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                'shape': shape_params_tensor.detach().cpu(),
                'expression': expression_params_tensor.detach().cpu(),
                'pose': pose_params_tensor.detach().cpu(),
                'rot': rot.detach().cpu(),
                'trans': trans.detach().cpu()
            }
    
    # 保存最佳参数
    torch.save(best_params, os.path.join(config.output_dir, "best_params.pth"))
    print(f"最佳参数已保存，最佳损失: {best_loss:.6f}")
    
    return best_params

# ----------------------------
# 主函数
# ----------------------------
def main():
    try:
        # 初始化配置
        config = Config()
        print(f"使用设备: {config.device}")
        
        # 准备工作
        init_data = initialize(config)
        
        # 执行优化
        best_params = optimize_parameters(config, init_data)
        
        print("优化完成!")
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        raise
    finally:
        # 确保资源正确释放
        if 'init_data' in locals() and 'writer' in init_data:
            init_data['writer'].close()
            print("TensorBoard writer已关闭")

if __name__ == "__main__":
    main()
