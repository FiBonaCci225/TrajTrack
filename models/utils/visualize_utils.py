import open3d as o3d
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, Box

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def create_thick_bbox(corners, color, line_radius):
    """
    从8个角点创建具有指定粗细的边界框

    参数:
    - corners: 8个角点的坐标，形状为(8, 3)
    - color: 线条颜色
    - line_radius: 线条半径（控制粗细）
    """
    cylinders = []
    standard_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 侧面连接
    ]
    for line in standard_lines:
        start = corners[line[0]]
        end = corners[line[1]]

        # 计算长度和方向
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length

        # 创建圆柱体表示线条
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=line_radius,
            height=length
        )

        # 计算旋转矩阵
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis):
            rotation_matrix = np.eye(3)
        elif np.allclose(direction, -z_axis):
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz([np.pi, 0, 0])
        else:
            rotation_axis = np.cross(z_axis, direction)
            rotation_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle
            )

        # 应用变换
        cylinder.rotate(rotation_matrix, center=(0, 0, 0))
        cylinder.translate((start + end) / 2)
        cylinder.paint_uniform_color(color)

        cylinders.append(cylinder)

    return cylinders


def open3d_show_frame(index, xyz, gt_box=None, refine_box=None, wo_refine_box=None, refine_traj=None, gt_traj=None,
                      window_name="Final View", zoom=1, point_size=1.0, line_radius=0.01, yaw=5, pitch=30):
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # 转换点云数据
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.cpu().numpy()
    points = xyz.reshape(-1, 3)

    # 创建黑色点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([0.52, 0.68, 0.92])  # RGB黑色
    pcd.paint_uniform_color([0, 0, 0])

    # 设置白色背景
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])  # RGB白色
    render_opt.point_size = point_size

    # 添加点云到场景
    vis.add_geometry(pcd)

    if refine_traj is not None:
        refine_traj = refine_traj.cpu().numpy()
        pcd_traj = o3d.geometry.PointCloud()
        pcd_traj.points = o3d.utility.Vector3dVector(refine_traj)
        # pcd.paint_uniform_color([0.52, 0.68, 0.92])  # RGB黑色
        pcd_traj.paint_uniform_color([1, 0.1, 0.1])
        vis.add_geometry(pcd_traj)

        gt_traj = gt_traj.cpu().numpy()
        pcd_traj = o3d.geometry.PointCloud()
        pcd_traj.points = o3d.utility.Vector3dVector(gt_traj)
        # pcd.paint_uniform_color([0.52, 0.68, 0.92])  # RGB黑色
        pcd_traj.paint_uniform_color([0.1, 0.8, 0.1])
        vis.add_geometry(pcd_traj)

    # 定义标准立方体边线连接关系（12条边）
    standard_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 侧面连接
    ]

    box_configs = [
        (gt_box, [0.1, 0.8, 0.1]),  # 绿色-GT框
        (refine_box, [1, 0.1, 0.1]),  # 红色-优化后框
        (wo_refine_box, [0.1, 0.1, 1])  # 蓝色-未优化框
    ]

    for box, color in box_configs:
        if box is not None:
            try:
                corners = box.corners().T.reshape(-1, 3)

                # 添加颜色方向微小偏移（解决重叠问题）
                offset = 0.001 * np.array(color)  # RGB方向各偏移1mm
                adjusted_corners = corners + offset

                # 使用TriangleMesh创建粗线条边界框
                thick_bbox_cylinders = create_thick_bbox(adjusted_corners, color, line_radius)

                # 将所有圆柱体添加到可视化器
                for cylinder in thick_bbox_cylinders:
                    vis.add_geometry(cylinder)

            except Exception as e:
                print(f"Error drawing {color} box: {str(e)}")

    # 自动适配视角
    if len(points) > 0:
        view_control = vis.get_view_control()
        if gt_box is not None:
            view_control.set_lookat(gt_box.center)
        view_control.set_front([0, 0, 1])
        view_control.set_up([0, 1, 0])
        # # 旋转15度（绕Z轴和X轴）
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
        new_front = [np.sin(yaw),  # X分量
                     -np.sin(pitch) * np.cos(yaw),  # Y分量
                     np.cos(pitch) * np.cos(pitch)]  # Z分量（向下为负）
        new_front = new_front / np.linalg.norm(new_front)
        view_control.set_front(new_front)
        view_control.set_zoom(zoom)

    # 运行可视化
    vis.update_renderer()
    vis.run()
    output_dir = "/home/fbc/Code/P2P_traj/paper/correct_traj"
    frame_path = os.path.join(output_dir, f"frame_{index:04d}.png")
    vis.capture_screen_image(frame_path)
    vis.destroy_window()


def open3d_show_frame_scene(relate_dist, index, xyz, gt_box=None, refine_box=None, wo_refine_box=None,
                      refine_iou=0, wo_refine_iou=0, window_name="Final View",
                      zoom=1, point_size=1.0, line_radius=0.3, num_lines=15, yaw=45, pitch=60):

    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=3840, height=2160)

    # 转换点云数据
    if not isinstance(xyz, np.ndarray):
        xyz = xyz.cpu().numpy()
    points = xyz.reshape(-1, 3)

    # 创建黑色点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.paint_uniform_color([0.52, 0.68, 0.92])  # RGB黑色
    pcd.paint_uniform_color([0, 0, 0])

    # 设置白色背景
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])  # RGB白色
    render_opt.point_size = point_size

    # 添加点云到场景
    vis.add_geometry(pcd)

    # 定义标准立方体边线连接关系（12条边）
    standard_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7],  # 侧面连接
    ]

    # 定义边界框配置
    box_configs = [
        ("gt", gt_box, [0.1, 0.8, 0.1]),  # 绿色-GT框
        ("refine", refine_box, [1, 0.1, 0.1]),  # 红色-优化后框
        ("wo_refine", wo_refine_box, [0.1, 0.1, 1])  # 蓝色-未优化框
    ]

    # # 根据IOU差异决定显示哪些框
    # if relate_dist < 1e-2:
    #     box_configs = [b for b in box_configs if b[0] != "refine"]

    # 创建厚线边界框
    for name, box, color in box_configs:
        if box is None:
            continue

        try:
            corners = box.corners().T.reshape(-1, 3)
            if name == 'wo_refine':
                # 添加颜色方向微小偏移（解决重叠问题）
                offset = 0.01 * np.array([1,1,1])  # RGB方向各偏移1mm
                adjusted_corners = corners + offset
            else:
                adjusted_corners = corners

            # 使用TriangleMesh创建粗线条边界框
            thick_bbox_cylinders = create_thick_bbox(adjusted_corners, color, line_radius)

            # 将所有圆柱体添加到可视化器
            for cylinder in thick_bbox_cylinders:
                vis.add_geometry(cylinder)

        except Exception as e:
            print(f"Error drawing {color} box: {str(e)}")

    # 设置可视化参数

    lookat_point = points.mean(axis=0) if len(points) > 0 else np.array([0, 0, 0])

    # 设置视图控制
    view_control = vis.get_view_control()
    view_control.set_lookat(lookat_point)
    view_control.set_front([0, 0, 1])  # Z轴向前
    view_control.set_up([0, 1, 0])  # Y轴向上
    # # 旋转15度（绕Z轴和X轴）
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    new_front = [np.sin(yaw),  # X分量
                 -np.sin(pitch) * np.cos(yaw),  # Y分量
                 np.cos(pitch) * np.cos(pitch)]  # Z分量（向下为负）
    new_front = new_front / np.linalg.norm(new_front)
    view_control.set_front(new_front)
    view_control.set_zoom(zoom)

    # 设置渲染选项
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])  # 白色背景
    render_opt.point_size = point_size

    # 更新渲染器
    vis.update_renderer()
    vis.run()
    output_dir = "/home/fbc/Code/P2P_traj/video"
    frame_path = os.path.join(output_dir, f"frame_{index:04d}.png")
    vis.capture_screen_image(frame_path)
    vis.destroy_window()

    return vis


def open3d_show_tracklet(xyz, gt_boxs=None, refine_boxs=None, wo_refine_boxs=None, refine_iou=0, wo_refine_iou=0, window_name="Final View", zoom=0.7, point_size=1.0):
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)

    # 转换点云数据
    # if not isinstance(xyz, np.ndarray):
    #     xyz = xyz.cpu().numpy()
    #     points = xyz.reshape(-1, 3)

    # 定义标准立方体边线连接关系（12条边）
    standard_lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面连接
    ]
    if refine_boxs is not None and gt_boxs is not None and wo_refine_boxs is None:
        color = [1, 0, 0]
        for i, (pc, box,refine_box) in enumerate(zip(xyz, gt_boxs,refine_boxs)):
            if box is not None:
                # 创建黑色点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)

                pcd.paint_uniform_color([1 - (0.48 * (i / len(xyz))), 1 -  (0.32 * (i / len(xyz))), 1 - (0.08 * (i / len(xyz)))])  # RGB黑色
                # 0.52 + (0.48 * (i / len(xyz))), 0.68 + (0.32 * (i / len(xyz))), 0.92 + (0.08 * (i / len(xyz)))
                # 添加点云到场景
                vis.add_geometry(pcd)
                try:
                    gt_color = [0, 1, 0]
                    corners = box.corners().T.reshape(-1, 3)

                    # 添加颜色方向微小偏移（解决重叠问题）
                    offset = 0.001 * np.array(gt_color)  # RGB方向各偏移1mm
                    adjusted_corners = corners + offset

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(adjusted_corners)
                    line_set.lines = o3d.utility.Vector2iVector(standard_lines)
                    line_set.colors = o3d.utility.Vector3dVector([gt_color] * len(standard_lines))
                    vis.add_geometry(line_set)

                    corners = refine_box.corners().T.reshape(-1, 3)

                    # 添加颜色方向微小偏移（解决重叠问题）
                    offset = 0.001 * np.array(color)  # RGB方向各偏移1mm
                    adjusted_corners = corners + offset

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(adjusted_corners)
                    line_set.lines = o3d.utility.Vector2iVector(standard_lines)
                    line_set.colors = o3d.utility.Vector3dVector([color] * len(standard_lines))
                    vis.add_geometry(line_set)
                except Exception as e:
                    print(f"Error drawing {color} box: {str(e)}")
    elif refine_boxs is None and gt_boxs is not None and wo_refine_boxs is None:
        color = [1, 0, 0]
        for i, (pc, box) in enumerate(zip(xyz, gt_boxs)):
            if box is not None:
                # 创建黑色点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                #
                pcd.paint_uniform_color(
                    [1 - (0.48 * (i / len(xyz))), 1 - (0.32 * (i / len(xyz))), 1 - (0.08 * (i / len(xyz)))])  # RGB黑色
                # 0.52 + (0.48 * (i / len(xyz))), 0.68 + (0.32 * (i / len(xyz))), 0.92 + (0.08 * (i / len(xyz)))
                # 添加点云到场景
                vis.add_geometry(pcd)
                try:
                    gt_color = [0, 1, 0]
                    corners = box.corners().T.reshape(-1, 3)

                    # 添加颜色方向微小偏移（解决重叠问题）
                    offset = 0.001 * np.array(gt_color)  # RGB方向各偏移1mm
                    adjusted_corners = corners + offset

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(adjusted_corners)
                    line_set.lines = o3d.utility.Vector2iVector(standard_lines)
                    line_set.colors = o3d.utility.Vector3dVector([gt_color] * len(standard_lines))
                    vis.add_geometry(line_set)
                except Exception as e:
                    print(f"Error drawing {color} box: {str(e)}")
    elif refine_boxs is not None and gt_boxs is not None and wo_refine_boxs is not None:
        color = [0, 0, 1]
        ref_color = [1, 0, 0]
        for i, (pc, box, refine_box, wo_refine_box) in enumerate(zip(xyz, gt_boxs, refine_boxs, wo_refine_boxs)):
            if box is not None:
                # 创建黑色点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)

                pcd.paint_uniform_color(
                    [1 - (0.48 * (i / len(xyz))), 1 - (0.32 * (i / len(xyz))), 1 - (0.08 * (i / len(xyz)))])  # RGB黑色
                # 0.52 + (0.48 * (i / len(xyz))), 0.68 + (0.32 * (i / len(xyz))), 0.92 + (0.08 * (i / len(xyz)))
                # 添加点云到场景
                vis.add_geometry(pcd)
                try:
                    gt_color = [0, 1, 0]
                    corners = box.corners().T.reshape(-1, 3)

                    # 添加颜色方向微小偏移（解决重叠问题）
                    offset = 0.001 * np.array(gt_color)  # RGB方向各偏移1mm
                    adjusted_corners = corners + offset

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(adjusted_corners)
                    line_set.lines = o3d.utility.Vector2iVector(standard_lines)
                    line_set.colors = o3d.utility.Vector3dVector([gt_color] * len(standard_lines))
                    vis.add_geometry(line_set)

                    corners = refine_box.corners().T.reshape(-1, 3)

                    # 添加颜色方向微小偏移（解决重叠问题）
                    offset = 0.001 * np.array(ref_color)  # RGB方向各偏移1mm
                    adjusted_corners = corners + offset

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(adjusted_corners)
                    line_set.lines = o3d.utility.Vector2iVector(standard_lines)
                    line_set.colors = o3d.utility.Vector3dVector([ref_color] * len(standard_lines))
                    vis.add_geometry(line_set)

                    corners = wo_refine_box.corners().T.reshape(-1, 3)

                    # 添加颜色方向微小偏移（解决重叠问题）
                    offset = 0.001 * np.array(color)  # RGB方向各偏移1mm
                    adjusted_corners = corners + offset

                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(adjusted_corners)
                    line_set.lines = o3d.utility.Vector2iVector(standard_lines)
                    line_set.colors = o3d.utility.Vector3dVector([color] * len(standard_lines))
                    vis.add_geometry(line_set)
                except Exception as e:
                    print(f"Error drawing {color} box: {str(e)}")
    elif refine_boxs is None and gt_boxs is None and wo_refine_boxs is None:
        color = [1, 0, 0]
        for i, pc in enumerate(xyz):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.paint_uniform_color([1 - (i / len(xyz)),1 - (i / len(xyz)), 1 - (i / len(xyz))])
            vis.add_geometry(pcd)

    # 设置白色背景
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])  # RGB白色
    render_opt.point_size = point_size


    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(zoom)

    # 运行可视化
    vis.run()
    vis.destroy_window()


def open3d_show_traj(xyz, coords=None, window_name="Final View", zoom=0.7, point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    color = [1, 0, 0]
    for i, pc in enumerate(xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.paint_uniform_color([1 - (i / len(xyz)), 1 - (i / len(xyz)), 1 - (i / len(xyz))])
        vis.add_geometry(pcd)
    # coords = np.array(coords)
    # pcd_past = o3d.geometry.PointCloud()
    # pcd_past.points = o3d.utility.Vector3dVector(coords[:4])
    # pcd_past.paint_uniform_color([1, 0, 0])
    # vis.add_geometry(pcd_past)
    # pcd_fut = o3d.geometry.PointCloud()
    # pcd_fut.points = o3d.utility.Vector3dVector(coords[4:])
    # pcd_fut.paint_uniform_color([0, 1, 0])
    # vis.add_geometry(pcd_fut)

    # 设置白色背景
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([1, 1, 1])  # RGB白色
    render_opt.point_size = point_size

    view_control = vis.get_view_control()
    view_control.set_front([0, 0, 1])
    view_control.set_up([0, 1, 0])
    view_control.set_zoom(zoom)

    # 运行可视化
    vis.run()
    vis.destroy_window()


def open3d_show_video(len, xyz, gt_box=None, refine_box=None, wo_refine_box=None,
                      refine_iou=0, wo_refine_iou=0, zoom=1, point_size=1.0, line_thickness=0.2, num_lines=10, yaw=45, pitch=60):
    out_path = "track.mp4"
    output_dir = "/home/fbc/Code/P2P_traj/video"
    frames = []
    for i in range(len):
        vis = open3d_show_frame(xyz[i], gt_box=gt_box[i], refine_box=refine_box[i], wo_refine_box=wo_refine_box[i],
                                zoom=zoom, point_size=point_size,
                                line_thickness=line_thickness, num_lines=num_lines, yaw=yaw, pitch=pitch)
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path)  # 直接保存为PNG
        frames.append(cv2.imread(frame_path))  # 读取到内存
        vis.destroy_window()

    # 使用OpenCV合成视频（修复时间轴问题）
    video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264编码
    video = cv2.VideoWriter(video_path, fourcc, 30, (1280, 720))
    for frame in frames:
        video.write(frame)
    video.release()
    # 清理临时文件
    for filename in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, filename))
    os.rmdir(output_dir)
    vis.destroy_window()
    print(f"视频已保存至: {out_path}")


def get_scene_pc(self, meta):
    sample = self.nusc.get('sample', meta['box_anno']['sample_token'])
    lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    pc_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
    pointcloud = LidarPointCloud.from_file(pc_path)
    cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pointcloud.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pointcloud.translate(np.array(cs_record['translation']))
    poserecord = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pointcloud.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pointcloud.translate(np.array(poserecord['translation']))
    return pointcloud