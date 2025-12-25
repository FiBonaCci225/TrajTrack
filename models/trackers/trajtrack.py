from mmengine.model import BaseModel
# from open3d.examples.visualization.remove_geometry import start_time

from datasets.metrics import estimateOverlap, estimateAccuracy
from datasets import points_utils
from nuscenes.utils import geometry_utils
from mmengine.registry import MODELS
from models.traj_utils.config import Config
from ..utils.visualize_utils import *
import copy
from .imm import IMM
from nuscenes.nuscenes import NuScenes

@MODELS.register_module()
class TrajTrack(BaseModel):

    def __init__(self,
                 backbone=None,
                 fuser=None,
                 head=None,
                 cfg=None):
        super().__init__()
        self.config = cfg
        self.backbone = MODELS.build(backbone)
        self.fuse = MODELS.build(fuser)
        self.head = MODELS.build(head)
        self.imm = IMM(cfg=Config('Implicit_Trajectory_Prediction'))
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='/home/fbc/data/nuscenes/v1.0-trainval', verbose=False)

    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'loss',
                **kwargs):
        # self.set_data(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def get_feats(self, inputs, mode='loss'):
        prev_points = inputs['prev_points']
        this_points = inputs['this_points']
        stack_points = prev_points + this_points

        # Explicit Motion Proposal
        # ================================================================================
        stack_feats = self.backbone(stack_points)
        cat_feats = self.fuse(stack_feats)
        if self.config.box_aware:
            wlh = torch.stack(inputs['wlh']) if isinstance(inputs['wlh'], list) \
                else inputs['wlh'].unsqueeze(0)
            results = self.head(cat_feats, wlh)
        else:
            results = self.head(cat_feats)
        # ================================================================================

        # Implicit Trajectory Prediction
        # ================================================================================
        if mode == 'loss':
            data = {i: torch.stack(inputs['traj'][i]) for i in inputs['traj']}
            self.imm.set_data(data)
            self.imm.data['motion'] = cat_feats
            self.imm.get_feats()
        # ================================================================================

        return results

    def inference(self, inputs):
        results = self.get_feats(inputs, mode='infer')
        coors = results['coors'][0]
        if self.config.use_rot:
            rot = results['rotation'][0]
            return coors, rot
        return coors

    def loss(self, inputs, data_samples):
        results = self.get_feats(inputs)
        losses = dict()
        losses.update(self.head.loss(results, data_samples))
        _, loss_dict, _ = self.imm.compute_loss()
        losses.update(loss_dict)

        return losses

    def predict(self, inputs):
        iou_ref, dist_ref, result_bbs_ref = self.pre_w_refine(inputs, 0.2)
        # index = 0
        # vel = np.array([inputs[i]['3d_bbox'].center for i in range(len(inputs))])[:, :2]
        # vel = vel[1:] - vel[:-1]
        # speed = np.linalg.norm(vel, axis=1)
        # # speed = np.abs(np.diff(vel)) / vel[:-1]
        # if abs(np.array([inputs[i]['3d_bbox'].orientation.yaw_pitch_roll[0] for i in range(len(inputs))]).max() - np.array([inputs[i]['3d_bbox'].orientation.yaw_pitch_roll[0] for i in range(len(inputs))]).min()) > 0.5:
        #     iou_wo_ref, dist_wo_ref, result_bbs_wo_ref = self.pre_wo_refine(inputs)
        #     open3d_show_frame(inputs[index]["pc"].points.transpose(1, 0)[:, :3], gt_box=inputs[index]["3d_bbox"],
        #                       refine_box=result_bbs_ref[index], wo_refine_box=result_bbs_wo_ref[index],
        #                       zoom=1, point_size=5.0, line_radius=0.05, yaw=5,
        #                       pitch=30)
        # if len(speed)!=0 and max(speed)>=1.5:
        #     iou_wo_ref, dist_wo_ref, result_bbs_wo_ref = self.pre_wo_refine(inputs)
        #     open3d_show_frame(inputs[index]["pc"].points.transpose(1, 0)[:, :3], gt_box=inputs[index]["3d_bbox"],
        #                       refine_box=result_bbs_ref[index], wo_refine_box=result_bbs_wo_ref[index],
        #                       zoom=1, point_size=5.0, line_radius=0.05, yaw=5,
        #                       pitch=30)


        # open3d_show_frame_scene(index, get_scene_pc(self, inputs[index]['meta']).points.transpose(1, 0)[:, :3], gt_box=inputs[index]["3d_bbox"], refine_box=result_bbs_ref[index],zoom=0.15, point_size=3, line_radius=0.05)
        return iou_ref, dist_ref

    def pre_wo_refine(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]

            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                last_coors = np.array([0., 0.])
            else:
                data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
                if flag:
                    if self.config.use_rot:
                        coors, rot = self.inference(data_dict)
                        rot = float(rot)
                    else:
                        coors = self.inference(data_dict)
                        rot = 0.
                    coors_x = float(coors[0])
                    coors_y = float(coors[1])
                    coors_z = float(coors[2])
                    last_coors = np.array([coors_x, coors_y])
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [coors_x, coors_y, coors_z, rot],
                        degrees=True, use_z=True, limit_box=False)
                else:
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [last_coors[0], last_coors[1], 0, 0],
                        degrees=True, use_z=True, limit_box=False)
                results_bbs.append(candidate_box)
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            ious.append(this_overlap)
            distances.append(this_accuracy)
        return ious, distances, results_bbs

    # def pre_w_refine(self, inputs, th=0.2):
    #     ious = []
    #     distances = []
    #     results_bbs = []
    #     point_in_box_num = sum(geometry_utils.points_in_box(inputs[0]['3d_bbox'], inputs[0]['pc'].points[:3], 1.25))
    #
    #     for frame_id in range(len(inputs)):  # tracklet
    #         this_bb = inputs[frame_id]["3d_bbox"]
    #         if frame_id == 0:
    #             # the first frame
    #             results_bbs.append(this_bb)
    #             last_coors = np.array([0., 0.])
    #         else:
    #             data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
    #             if flag:
    #                 if self.config.use_rot:
    #                     coors, rot = self.inference(data_dict)
    #                     rot = float(rot)
    #                 else:
    #                     coors = self.inference(data_dict)
    #                     rot = 0.
    #                 coors_x = float(coors[0])
    #                 coors_y = float(coors[1])
    #                 coors_z = float(coors[2])
    #                 last_coors = np.array([coors_x, coors_y])
    #                 candidate_box = points_utils.getOffsetBB(
    #                     ref_bb, [coors_x, coors_y, coors_z, rot],
    #                     degrees=True, use_z=True, limit_box=False)
    #             else:
    #                 candidate_box = points_utils.getOffsetBB(
    #                     ref_bb, [last_coors[0], last_coors[1], 0, 0],
    #                     degrees=True, use_z=True, limit_box=False)
    #             results_bbs.append(candidate_box)
    #
    #         this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
    #         this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
    #         # Trajectory-guide Proposal Refinement
    #         # ================================================================================
    #         if this_overlap <= th:
    #             offset_def = 3
    #             past_nums = 3
    #             if frame_id > offset_def + past_nums:
    #                 offset = offset_def
    #                 cur_id = frame_id - offset
    #             elif frame_id > past_nums:
    #                 offset = frame_id - past_nums - 1
    #                 cur_id = past_nums + 1
    #             elif frame_id >= 0:
    #                 offset = 0
    #                 cur_id = frame_id
    #             results_bb = results_bbs[:cur_id]
    #             res = self.traj_refine(results_bb, past_nums)
    #             refine_box = copy.deepcopy(results_bbs[-2])
    #             refine_overlap = []
    #             refine_overlap.append(this_overlap)
    #
    #             for sample_idx in range(res.shape[0]):
    #                 refine_box.center[:2] = np.array(res[sample_idx, offset].data.cpu())
    #                 refine_overlap.append(estimateOverlap(this_bb, refine_box, dim=3, up_axis=[0, 0, 1]))
    #             this_overlap = max(refine_overlap)
    #             idx = refine_overlap.index(max(refine_overlap))
    #             if idx != 0:
    #                 refine_box.center[:2] = np.array(res[idx - 1, offset].data.cpu())
    #                 results_bbs[-1] = refine_box
    #                 this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
    #         # ================================================================================
    #         ious.append(this_overlap)
    #         distances.append(this_accuracy)
    #     return ious, distances, results_bbs
    def pre_w_refine(self, inputs, th=0.2):
        ious = []
        distances = []
        results_bbs = []

        point_in_box_num = sum(
            geometry_utils.points_in_box(
                inputs[0]['3d_bbox'],
                inputs[0]['pc'].points[:3],
                1.25
            )
        )

        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]

            if frame_id == 0:
                results_bbs.append(this_bb)
                last_coors = np.array([0., 0.])
            else:
                data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
                if flag:
                    if self.config.use_rot:
                        coors, rot = self.inference(data_dict)
                        rot = float(rot)
                    else:
                        coors = self.inference(data_dict)
                        rot = 0.
                    coors_x = float(coors[0])
                    coors_y = float(coors[1])
                    coors_z = float(coors[2])
                    last_coors = np.array([coors_x, coors_y])
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [coors_x, coors_y, coors_z, rot],
                        degrees=True, use_z=True, limit_box=False
                    )
                else:
                    candidate_box = points_utils.getOffsetBB(
                        ref_bb, [last_coors[0], last_coors[1], 0, 0],
                        degrees=True, use_z=True, limit_box=False
                    )
                results_bbs.append(candidate_box)

            # 当前预测与 GT 的 IoU / 误差
            this_overlap = estimateOverlap(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
            this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])

            # ================================================================
            #  Trajectory-Guided IMM Refinement (每帧都进入 IMM)
            # ================================================================
            if frame_id != 0:
                offset_def = 3
                past_nums = 3

                if frame_id > offset_def + past_nums:
                    offset = offset_def
                    cur_id = frame_id - offset
                elif frame_id > past_nums:
                    offset = frame_id - past_nums - 1
                    cur_id = past_nums + 1
                else:  # frame_id >= 0
                    offset = 0
                    cur_id = frame_id

                results_bb_hist = results_bbs[:cur_id]
                res = self.traj_refine(results_bb_hist, past_nums)  # [5, seq_len, 2]

                # 基于上一帧框构建 5 个候选框
                if len(results_bbs) >= 2:
                    base_box = copy.deepcopy(results_bbs[-2])
                else:
                    base_box = copy.deepcopy(results_bbs[-1])

                candidate_boxes = []
                candidate_overlaps = []

                for sample_idx in range(res.shape[0]):
                    new_box = copy.deepcopy(base_box)
                    new_box.center[:2] = np.array(res[sample_idx, offset].data.cpu())
                    ov = estimateOverlap(this_bb, new_box, dim=3, up_axis=[0, 0, 1])
                    candidate_boxes.append(new_box)
                    candidate_overlaps.append(ov)

                # 找所有 IoU >= 阈值 的 candidate
                valid_ids = [i for i, ov in enumerate(candidate_overlaps) if ov >= th]

                if len(valid_ids) > 0:
                    # 选 IoU 最大的那个
                    best_id = max(valid_ids, key=lambda idx: candidate_overlaps[idx])
                    results_bbs[-1] = candidate_boxes[best_id]
                    this_overlap = candidate_overlaps[best_id]
                    this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])
                # 否则保持原始 candidate_box，不变

            # ================================================================

            ious.append(this_overlap)
            distances.append(this_accuracy)

        return ious, distances, results_bbs

    def traj_refine(self, results_bbs, past_nums):
        # if len(results_bbs) <= 0:
        pre_motion_3D = ([results_bbs[0].center[:2] / 10 for _ in range((past_nums+1) - len(results_bbs))] +
                         [bb.center[:2] / 10 for bb in results_bbs[-(past_nums+1):]])
        pre_motion_3D = torch.stack([torch.tensor(arr) for arr in pre_motion_3D]).unsqueeze(0).to(torch.float32)
        pre_motion_mask = torch.ones((past_nums+1)).to(torch.float32).unsqueeze(0)

        # map_file = f'{self.cfg_traj.data_root_nuscenes_pred}/map_{self.cfg_traj.map_version}/{seq_name}.png'
        # map_meta_file = f'{self.cfg_traj.data_root_nuscenes_pred}/map_{self.cfg_traj.map_version}/meta_{seq_name}.txt'
        # scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        # meta = np.loadtxt(map_meta_file)
        # map_origin = meta[:2]
        # scale = meta[2]
        # homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        # geom_scene_map = GeometricMap(scene_map, homography, map_origin)

        data = {
            'pre_motion_3D': pre_motion_3D.to('cuda'),
            'pre_motion_mask': pre_motion_mask.to('cuda'),
            'heading': torch.tensor([results_bbs[-1].orientation.yaw_pitch_roll[0]]).to('cuda'),
            'valid_id': torch.tensor([0.0]).to('cuda'),
            # 'traj_scale': torch.tensor([self.cfg_traj.traj_scale]).to(results_bbs.device),
            'pred_mask': torch.tensor([[1]]).to('cuda'),
        }
        # motion = self.imm.data['motion']
        self.imm.set_data(data)
        # self.imm.data['motion'] = motion
        self.imm.inference(sample_num=5)
        res = self.imm.data[f'infer_dec_motion'].squeeze(0) * 10
        return res

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"
        offset_def = 3
        past_nums = 3
        if frame_id >= offset_def + past_nums:
            offset = offset_def
            cur_id = frame_id - offset
        elif frame_id >= past_nums:
            offset = frame_id - past_nums
            cur_id = past_nums
        elif frame_id >= 0:
            offset = 0
            cur_id = frame_id
        traj_id = list(range(cur_id - 3, cur_id + 1))
        coor = []
        mask = []
        heading = sequence[frame_id]['3d_bbox'].orientation.yaw_pitch_roll[0]
        for idx in traj_id:
            if idx < 0:
                coor.append(sequence[0]['3d_bbox'].center[:2])
                mask.append(0)
            elif idx >= len(sequence):
                coor.append(sequence[-1]['3d_bbox'].center[:2])
                mask.append(0)
            else:
                xy = sequence[idx]['3d_bbox'].center[:2]
                coor.append(xy)
                mask.append(1)
        coor = np.array(coor) / 10
        mask = np.array(mask)
        rotation = np.array(heading)
        offset = np.array([offset])
        traj = {'pre_motion_3D': torch.as_tensor(coor, dtype=torch.float32).cuda(),
                'pre_motion_mask': torch.as_tensor(mask, dtype=torch.bool).cuda(),
                'heading': torch.as_tensor(rotation, dtype=torch.float32).cuda(),
                'offset': torch.as_tensor(offset, dtype=torch.int).cuda()}
        prev_frame = sequence[frame_id - 1]
        this_frame = sequence[frame_id]

        prev_pc = prev_frame['pc']
        this_pc = this_frame['pc']
        ref_box = results_bbs[-1]

        prev_frame_pc = points_utils.crop_pc_in_range(prev_pc, ref_box, self.config.point_cloud_range)
        this_frame_pc = points_utils.crop_pc_in_range(this_pc, ref_box, self.config.point_cloud_range)

        prev_points = prev_frame_pc.points.T
        this_points = this_frame_pc.points.T

        if self.config.post_processing is True:
            ref_bb = points_utils.transform_box(ref_box, ref_box)
            prev_idx = geometry_utils.points_in_box(ref_bb, prev_points[:,:3].T, 1.25)
            if sum(prev_idx) < 3 and this_points.shape[0] < 25 and frame_id < 15:
                # not enough points for tracking
                flag = False
            else:
                flag = True
        else:
            flag = True

        if prev_points.shape[0] < 1:
            prev_points = np.zeros((1, 4), dtype='float32')
        if this_points.shape[0] < 1:
            this_points = np.zeros((1, 4), dtype='float32')

        data_dict = {'prev_points': [torch.as_tensor(prev_points, dtype=torch.float32).cuda()],
                     'this_points': [torch.as_tensor(this_points, dtype=torch.float32).cuda()],
                     'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32).cuda(),
                     'traj': traj
                     }

        return data_dict, results_bbs[-1], flag


