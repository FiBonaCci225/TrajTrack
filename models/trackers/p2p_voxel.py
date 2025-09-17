import torch
from mmengine.model import BaseModel
from datasets.metrics import estimateOverlap, estimateAccuracy
import numpy as np
from datasets import points_utils
from nuscenes.utils import geometry_utils
from mmengine.registry import MODELS
# from ..traj_pred.model.model_lib import model_dict
# from ..traj_pred.utils.config import Config
# from ..traj_pred.data.map import GeometricMap
import cv2

@MODELS.register_module()
class P2PVoxel(BaseModel):

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

    def forward(self,
                inputs,
                data_samples=None,
                mode: str = 'loss',
                **kwargs):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def get_feats(self, inputs):
        prev_points = inputs['prev_points']
        this_points = inputs['this_points']
        stack_points = prev_points + this_points

        stack_feats = self.backbone(stack_points)
        cat_feats = self.fuse(stack_feats)
        if self.config.box_aware:
            wlh = torch.stack(inputs['wlh']) if isinstance(inputs['wlh'], list) \
                else inputs['wlh'].unsqueeze(0)
            results = self.head(cat_feats, wlh)
        else:
            results = self.head(cat_feats)

        return results

    def inference(self, inputs):
        results = self.get_feats(inputs)
        coors = results['coors'][0]
        if self.config.use_rot:
            rot = results['rotation'][0]
            return coors, rot
        return coors

    def loss(self, inputs, data_samples):
        results = self.get_feats(inputs)
        losses = dict()
        losses.update(self.head.loss(results, data_samples))

        return losses

    def predict(self, inputs):
        ious = []
        distances = []
        results_bbs = []
        traj_preds = []
        for frame_id in range(len(inputs)):  # tracklet
            this_bb = inputs[frame_id]["3d_bbox"]

            if frame_id == 0:
                # the first frame
                results_bbs.append(this_bb)
                last_coors = np.array([0., 0.])
            else:
                data_dict, ref_bb, flag = self.build_input_dict(inputs, frame_id, results_bbs)
                # if torch.sum(data_dict['prev_points'][0][:,:3]) ==0 and torch.sum(data_dict['this_points'][0][:,:3]) == 0:
                #     # results_bbs.append(ref_bb)
                #     print("Empty pointcloud!")
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
            # if this_overlap <= 0.2 and inputs[0]['meta']['sample_data_lidar']['scene_name'] != 'scene-0771' and inputs[0]['meta']['sample_data_lidar']['scene_name'] != 'scene-0778' \
            #         and inputs[0]['meta']['sample_data_lidar']['scene_name'] != 'scene-0917' and inputs[0]['meta']['sample_data_lidar']['scene_name'] != 'scene-0930' \
            #         and inputs[0]['meta']['sample_data_lidar']['scene_name'] != 'scene-0931' and inputs[0]['meta']['sample_data_lidar']['scene_name'] != 'scene-0924':
            #     if frame_id >= 8:
            #         results_bb = results_bbs[:-5]
            #     elif frame_id >=4 & frame_id <8:
            #         results_bb = results_bbs[:4]
            #     res = self.traj_refine(results_bb, inputs[0])
            #     refine_box = results_bbs[-2].copy()
            #     refine_overlap = []
            #     refine_overlap.append(this_overlap)
            #     # z = torch.tensor([results_bbs[-5:][i].center for i in range(5)])[:, 2]
            #     # xyz = np.array(torch.cat((res[:, :5], z.reshape(5, 1).repeat(5, 1, 1).cuda()), dim=-1).data.cpu())
            #     # diff = xyz - np.expand_dims(
            #     #     np.array([inputs[frame_id - 4:frame_id + 1][i]['3d_bbox'].center for i in range(5)]), axis=0)
            #     # diff_p2p = np.array([results_bbs[-5:][i].center for i in range(5)]) - np.expand_dims(
            #     #     np.array([inputs[frame_id - 4:frame_id + 1][i]['3d_bbox'].center for i in range(5)]), axis=0)
            #     # dis = np.linalg.norm(diff, axis=-1)
            #     # dis_p2p = np.linalg.norm(diff_p2p, axis=-1)
            #     for sample_idx in range(res.shape[0]):
            #         refine_box.center[:2] = np.array(res[sample_idx, frame_id - len(results_bb)].data.cpu())
            #         refine_overlap.append(estimateOverlap(this_bb, refine_box, dim=3, up_axis=[0, 0, 1]))
            #     this_overlap = max(refine_overlap)
            #     idx = refine_overlap.index(max(refine_overlap))
            #     if idx != 0:
            #         refine_box.center[:2] = np.array(res[idx - 1, frame_id - len(results_bb)].data.cpu())
            #         results_bbs[-1] = refine_box
            #         this_accuracy = estimateAccuracy(this_bb, results_bbs[-1], dim=3, up_axis=[0, 0, 1])

        # success_5flame = all(iou > 0.5 for iou in ious[:5])
        # success_4flame = all(iou > 0.5 for iou in ious[:4])
        # success_3flame = all(iou > 0.5 for iou in ious[:3])
        return ious, distances

    def traj_refine(self, results_bbs, inputs):
        # if len(results_bbs) <= 0:
        pre_motion_3D = [torch.stack(
            [torch.tensor(results_bbs[i].center[:2] / self.cfg_traj.traj_scale).to(torch.float32) for i in
             range(len(results_bbs))][-4:])]
        pre_motion_mask = [torch.ones(4).to(torch.float32)]

        seq_name = inputs['meta']['sample_data_lidar']['scene_name']
        map_file = f'{self.cfg_traj.data_root_nuscenes_pred}/map_{self.cfg_traj.map_version}/{seq_name}.png'
        map_meta_file = f'{self.cfg_traj.data_root_nuscenes_pred}/map_{self.cfg_traj.map_version}/meta_{seq_name}.txt'
        # if cv2.imread(map_file) is not None:
        #     self.traj_pref.pred_model[0].use_map = True
        #     scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
        #     meta = np.loadtxt(map_meta_file)
        #     map_origin = meta[:2]
        #     scale = meta[2]
        #     homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]])
        #     geom_scene_map = GeometricMap(scene_map, homography, map_origin)
        # else:
        #     self.traj_pref.pred_model[0].use_map = False
        geom_scene_map = None

        data = {
            'pre_motion_3D': pre_motion_3D,
            'pre_motion_mask': pre_motion_mask,
            'heading': np.array([results_bbs[-1].orientation.yaw_pitch_roll[0]]),
            'valid_id': [0.0],
            'traj_scale': self.cfg_traj.traj_scale,
            'pred_mask': np.array([1]),
            'scene_map': geom_scene_map,
            'seq': seq_name,
        }
        self.traj_pref.set_data(data)
        self.traj_pref.main(mean=True, need_weights=False)
        res = self.traj_pref.data[f'infer_dec_motion'].squeeze(0) * self.cfg_traj.traj_scale
        return res

    def build_input_dict(self, sequence, frame_id, results_bbs):
        assert frame_id > 0, "no need to construct an input_dict at frame 0"

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
                     'wlh': torch.as_tensor(ref_box.wlh, dtype=torch.float32).cuda()
                     }

        return data_dict, results_bbs[-1], flag
