import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
import torch.nn.functional as F
import os
import random


class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
        super().__init__()
        self.sep_head_dict = sep_head_dict

        for cur_name in self.sep_head_dict:
            output_channels = self.sep_head_dict[cur_name]['out_channels']
            num_conv = self.sep_head_dict[cur_name]['num_conv']

            fc_list = []
            for k in range(num_conv - 1):
                fc_list.append(nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU()
                ))
            fc_list.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1, bias=True))
            fc = nn.Sequential(*fc_list)
            if 'hm' in cur_name:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_normal_(m.weight.data)
                        if hasattr(m, "bias") and m.bias is not None:
                            nn.init.constant_(m.bias, 0)

            self.__setattr__(cur_name, fc)

    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict


class CoInHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.feature_map_stride = self.model_cfg.TARGET_ASSIGNER_CONFIG.get('FEATURE_MAP_STRIDE', None)

        self.class_names = class_names
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
                bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
            ),
            nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
            nn.ReLU(),
        )

        self.heads_list = nn.ModuleList()
        self.separate_head_cfg = self.model_cfg.SEPARATE_HEAD_CFG
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            cur_head_dict = copy.deepcopy(self.separate_head_cfg.HEAD_DICT)
            #xqm
            cur_head_dict['hm'] = dict(out_channels=len(cur_class_names), num_conv=self.model_cfg.NUM_HM_CONV)
            self.heads_list.append(
                SeparateHead(
                    input_channels=self.model_cfg.SHARED_CONV_CHANNEL,
                    sep_head_dict=cur_head_dict,
                    init_bias=-2.19,
                    use_bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
                )
            )


        self.predict_boxes_when_training = predict_boxes_when_training
        self.forward_ret_dict = {}
        self.build_losses()


    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def get_loss(self):
        pred_dicts = self.forward_ret_dict['pred_dicts']
        target_dicts = self.forward_ret_dict['target_dicts']
        feat = self.forward_ret_dict['feat']
        orignal_heatmap = self.forward_ret_dict['orignal_heatmap']

        tb_dict = {}
        loss = 0

        for idx, pred_dict in enumerate(pred_dicts):
            pred_dict['hm'] = self.sigmoid(pred_dict['hm'])
            hm_loss = self.hm_loss_func(pred_dict['hm'], target_dicts['heatmaps'][idx])
            hm_loss *= self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']

            target_boxes = target_dicts['target_boxes'][idx]
            pred_boxes = torch.cat([pred_dict[head_name] for head_name in self.separate_head_cfg.HEAD_ORDER], dim=1)

            reg_loss = self.reg_loss_func(
                pred_boxes, target_dicts['masks'][idx], target_dicts['inds'][idx], target_boxes
            )
            loc_loss = (reg_loss * reg_loss.new_tensor(self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['code_weights'])).sum()
            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']

            loss += hm_loss + loc_loss
            tb_dict['hm_loss_head_%d' % idx] = hm_loss.item()
            tb_dict['loc_loss_head_%d' % idx] = loc_loss.item()


        #In the subsequent experiments, the weights of the loss function were further adjusted.        
        LP_loss = self.LPloss(feat, pred_dicts[0]['hm'], orignal_heatmap)
        loss += LP_loss
        tb_dict['LP_loss_head'] = LP_loss.item()

        x_contrast_loss = self.MCloss(feat, orignal_heatmap)
        loss += x_contrast_loss
        tb_dict['x_contrast_loss'] = x_contrast_loss.item()

        tb_dict['rpn_loss'] = loss.item()
        return loss, tb_dict

    def generate_predicted_boxes(self, batch_size, pred_dicts):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        post_center_limit_range = torch.tensor(post_process_cfg.POST_CENTER_LIMIT_RANGE).cuda().float()

        ret_dict = [{
            'pred_boxes': [],
            'pred_scores': [],
            'pred_labels': [],
        } for k in range(batch_size)]
        for idx, pred_dict in enumerate(pred_dicts):
            batch_hm = pred_dict['hm'].sigmoid()
            batch_center = pred_dict['center']
            batch_center_z = pred_dict['center_z']
            batch_dim = pred_dict['dim'].exp()
            batch_rot_cos = pred_dict['rot'][:, 0].unsqueeze(dim=1)
            batch_rot_sin = pred_dict['rot'][:, 1].unsqueeze(dim=1)
            batch_vel = pred_dict['vel'] if 'vel' in self.separate_head_cfg.HEAD_ORDER else None

            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,
                center=batch_center, center_z=batch_center_z, dim=batch_dim, vel=batch_vel,
                point_cloud_range=self.point_cloud_range, voxel_size=self.voxel_size,
                feature_map_stride=self.feature_map_stride,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                circle_nms=(post_process_cfg.NMS_CONFIG.NMS_TYPE == 'circle_nms'),
                score_thresh=post_process_cfg.SCORE_THRESH,
                post_center_limit_range=post_center_limit_range
            )

            for k, final_dict in enumerate(final_pred_dicts):
                final_dict['pred_labels'] = self.class_id_mapping_each_head[idx][final_dict['pred_labels'].long()]
                if post_process_cfg.NMS_CONFIG.NMS_TYPE != 'circle_nms':
                    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                        box_scores=final_dict['pred_scores'], box_preds=final_dict['pred_boxes'],
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=None
                    )

                    final_dict['pred_boxes'] = final_dict['pred_boxes'][selected]
                    final_dict['pred_scores'] = selected_scores
                    final_dict['pred_labels'] = final_dict['pred_labels'][selected]

                ret_dict[k]['pred_boxes'].append(final_dict['pred_boxes'])
                ret_dict[k]['pred_scores'].append(final_dict['pred_scores'])
                ret_dict[k]['pred_labels'].append(final_dict['pred_labels'])

        for k in range(batch_size):
            ret_dict[k]['pred_boxes'] = torch.cat(ret_dict[k]['pred_boxes'], dim=0)
            ret_dict[k]['pred_scores'] = torch.cat(ret_dict[k]['pred_scores'], dim=0)
            ret_dict[k]['pred_labels'] = torch.cat(ret_dict[k]['pred_labels'], dim=0) + 1

        return ret_dict

    @staticmethod
    def reorder_rois_for_refining(batch_size, pred_dicts):
        num_max_rois = max([len(cur_dict['pred_boxes']) for cur_dict in pred_dicts])
        num_max_rois = max(1, num_max_rois)  # at least one faked rois to avoid error
        pred_boxes = pred_dicts[0]['pred_boxes']

        rois = pred_boxes.new_zeros((batch_size, num_max_rois, pred_boxes.shape[-1]))
        roi_scores = pred_boxes.new_zeros((batch_size, num_max_rois))
        roi_labels = pred_boxes.new_zeros((batch_size, num_max_rois)).long()

        for bs_idx in range(batch_size):
            num_boxes = len(pred_dicts[bs_idx]['pred_boxes'])

            rois[bs_idx, :num_boxes, :] = pred_dicts[bs_idx]['pred_boxes']
            roi_scores[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_scores']
            roi_labels[bs_idx, :num_boxes] = pred_dicts[bs_idx]['pred_labels']
        return rois, roi_scores, roi_labels

    @staticmethod
    def guiyihua(feature):
        #print(feature.shape)
        value_max, ind_max = feature.max(1)
        value_min, ind_min = feature.min(1)
        feature_guiyihua = (feature - value_min.view(-1, 1 ,3).repeat(1, feature.size(1), 1)) / \
        (value_max.view(-1, 1 ,3).repeat(1, feature.size(1), 1) - value_min.view(-1, 1 ,3).repeat(1, feature.size(1), 1))
        
        return feature_guiyihua


    
    def batch_update_hm_labels(self, x, target_dict, thresh=0.6, multi_factor=1):
        #one2onelable
        thresh = 0.9
        heatmap = target_dict['heatmaps']
        hm = heatmap[0]

        if hm.sum() == 0:
            print("sum of hm is zero")
            return batch, 0,  torch.zeros_like(hm).to(hm)

        ##### prepare the feat
        # normalize the feat
        device =x.device
        feat = x.detach()
        bs = feat.size(0)
        chanel = feat.size(1)
     

        # [bs, 64, h, w]
        #2022.10.5
        feat = F.normalize(feat, dim=1, p=2)
        # [bs, 64, h*w] -> [bs, 64, 1, h*w]
        g_feat = feat.flatten(2).unsqueeze(2)
        # [bs, 1, num_class, h*w]
        weight = hm.flatten(2).unsqueeze(1)
        # obtain each gt instance feature
        # [bs, 64, num_class]
        g_feat = (g_feat * weight).sum(dim=-1)

        ##xqm
        g_feat_car, g_feat_ped, g_feat_cyc = torch.split(g_feat, 1, 2)
        weight_car, weight_ped, weight_cyc = torch.split(weight, 1, 2)
        g_feat_car = g_feat_car / (weight_car.sum())
        g_feat_ped = g_feat_ped / (weight_ped.sum())
        g_feat_cyc = g_feat_cyc / (weight_cyc.sum())
        g_feat = torch.cat((g_feat_car, g_feat_ped, g_feat_cyc), 2)

        # re-normalize
        g_feat = F.normalize(g_feat, dim=1, p=2)
        q_feat = feat.flatten(2).permute(0, 2, 1).contiguous()

 

        #feat = F.normalize(feat, dim=1, p=2)
    
        g_feat_kl = feat.flatten(2).unsqueeze(2)
     
        weight_kl = hm.flatten(2).unsqueeze(1)
        # obtain each gt instance feature
        g_feat_kl = (g_feat_kl * weight_kl).sum(dim=-1)

       
        g_feat_car_kl, g_feat_ped_kl, g_feat_cyc_kl = torch.split(g_feat_kl, 1, 2)
        weight_car_kl, weight_ped_kl, weight_cyc_kl = torch.split(weight_kl, 1, 2)
        g_feat_car_kl = g_feat_car_kl / (weight_car_kl.sum())
        g_feat_ped_kl = g_feat_ped_kl / (weight_ped_kl.sum())
        g_feat_cyc_kl = g_feat_cyc_kl / (weight_cyc_kl.sum())
        g_feat_kl = torch.cat((g_feat_car_kl, g_feat_ped_kl, g_feat_cyc_kl), 2)

        q_feat_kl = feat.flatten(2).permute(0, 2, 1).contiguous()

        dist = torch.bmm(q_feat, g_feat)
        #print(dist.shape)

        #compute l2 distance
        q_feat_kl = q_feat_kl.view(bs, -1, chanel, 1).repeat(1,1,1,3)
        g_feat_kl = g_feat_kl.view(bs, 1, chanel, -1).repeat(1,q_feat.size(1),1,1)
        # dist_l2 = q_feat_l2 - g_feat_l2
        # distl2 = 1. - (torch.sqrt(torch.sum(dist_l2 ** 2, axis=2) + 1e-12)*0.01).clamp_(0., 1.)

        #compute kl
        dist_kl = F.kl_div(q_feat_kl, g_feat_kl, reduction='none', log_target=True)  # [m, nsample, d] - kl(pred, gt) to calculate kl = gt * [ log(gt) - log(pred) ]
        distkl = 1 - self.guiyihua(dist_kl.sum(2))  # [m, nsample]
        

        # [bs, num_class, h, w] -> [bs, h, w]
        sum_hm = hm.sum(dim=1)

        mask_query = torch.where(sum_hm > 0, torch.zeros_like(sum_hm), torch.ones_like(sum_hm)).to(feat)
        mask_query = mask_query.flatten(1).unsqueeze(2)
        mask_dist = dist * mask_query #where is instance where is zero, ba yi biao zhu de wu ti kou diao 

        mask_distkl = distkl* mask_query
        

        # [bs, h*w]
        #value, class_ind = mask_dist_final.max(dim=-1)
        value, class_ind = mask_dist.max(dim=-1)
        valuekl, class_indkl = mask_distkl.max(dim=-1)

        # [bs, h*w]
        t_ind = torch.where(value >= thresh)
        t_indkl = torch.where(valuekl >= thresh)
        
        t_class_ind = class_ind[t_ind]
        t_class_indkl = class_indkl[t_indkl]

        ###### assign the dist to pseudo positive instances
        # [bs, num_class, h*w]
        pseudo_hm = torch.zeros_like(hm).flatten(2).to(feat)
        pseudo_hm[(t_ind[0], t_class_ind, t_ind[1])] = mask_dist[(t_ind[0], t_ind[1], t_class_ind)]
        pseudo_hm = pseudo_hm.view(*hm.size())
        
        pseudo_hm_temp = torch.zeros_like(hm).flatten(2).to(feat)
        pseudo_hm_temp[(t_indkl[0], t_class_indkl, t_indkl[1])] = mask_distkl[(t_indkl[0], t_indkl[1], t_class_indkl)]
        pseudo_hm_temp = pseudo_hm_temp.view(*hm.size())

        # scale the pseudo_hm
        pseudo_hm = (multi_factor * pseudo_hm).clamp_(0., 1.)
        pseudo_hm_temp = (multi_factor * pseudo_hm_temp).clamp_(0., 1.)
        pseudo_hm_final = torch.cat((pseudo_hm[:, 0:1, :,:], pseudo_hm_temp[:, 1:3, :,:]),1)
        hm += pseudo_hm_final
        hm.clamp_(0., 1.)

        target_dict['heatmaps'][0] =  hm


        return target_dict

    def LPloss(self, feat, score_hm, heatmap): #hm b,1,w,h

     
        loss = 0
        topk_list = [20, 10, 10]
        for i in range(heatmap.size(1)):
            bs, c, h, w = feat.size()
            hm = heatmap[:, i:i+1, :, :]
            score = score_hm[:, i:i+1, :, :]
    
            topk_t = topk_list[i]
            num_class = hm.size(1)
            device = feat.device
            score = score.detach()

           
            feat = F.normalize(feat, dim=1, p=2)

            # 1.prepare q
            q = torch.zeros(bs, num_class, c).to(device)
            one_ind = torch.where(hm==1.)

            tmp_feat = feat.permute(0,2,3,1).contiguous()
            q[one_ind[0],one_ind[1]] = tmp_feat[one_ind[0],one_ind[2],one_ind[3]]
            class_mask = torch.zeros(bs, num_class).to(device)
            class_mask[one_ind[0],one_ind[1]] = 1.
            if class_mask.sum()==0:
                return torch.zeros(1).to(device), None

            # 2.prepare k0
            k0_feat = feat.flatten(2).unsqueeze(2)
            weight = hm.flatten(2).unsqueeze(1)
            k0_feat = (k0_feat * weight).sum(dim=-1)
            k0_feat = k0_feat / (weight.sum())

            # [bs, num_class, 64]
            k0 = k0_feat.permute(0, 2, 1).contiguous()
            k0 = F.normalize(k0, p=2, dim=-1)

            # 3.prepare positive and negative instances
            hm_sum = hm.sum(dim=1)
            hm_mask = hm_sum == 0
            hm_mask = hm_mask.unsqueeze(dim=1).repeat(1,num_class,1,1)

            score = score * hm_mask
            score_mask = class_mask.view(bs, num_class, 1, 1).expand(-1,-1,h, w)
            score = score * score_mask

            max_values, class_indices = score.max(dim=1)# sanzhangretu 2 yizhangretu
            max_values = max_values.flatten(1)
            class_indices = class_indices.flatten(1)
            topk_values, topk_indices = torch.topk(max_values, k=topk_t, dim=-1)# xuanze qian 512 ge yangben

            k_all_mask = topk_values > 0 # filter the background with value==0
            bs_ind = torch.arange(bs).unsqueeze(1).repeat(1, topk_t).view(-1)
            select_ind = [bs_ind, topk_indices.view(-1)]
            topk_class_indices = class_indices[select_ind].reshape(bs,-1)
            tmp_feat = feat.flatten(2).permute(0,2,1).contiguous()
            k_all = tmp_feat[select_ind].reshape(bs,topk_t, -1)

            # record the positions selected
            pseudo_hm = torch.zeros_like(hm).flatten(2)
            topk_class = class_indices[select_ind]
            values = (k_all_mask * 0.9).view(-1)
            pseudo_hm[select_ind[0], topk_class, select_ind[1]] = values
            pseudo_hm = pseudo_hm.view(bs, num_class, h, -1)


            # 4. prepare k(positive)
            class_inds = torch.arange(num_class).unsqueeze(1).repeat(1, topk_t).to(feat)
            target = -torch.ones(bs, num_class, topk_t).to(feat)
            class_mask_inds = torch.where(class_mask==1)
            target[class_mask_inds] = class_inds[class_mask_inds[1]]
            topk_class_indices = topk_class_indices.unsqueeze(1).repeat(1, num_class, 1)
            k_mask = target == topk_class_indices
            k_mask = k_mask * k_all_mask.unsqueeze(dim=1).expand(-1, 1, -1) #[bs, num_class, topk]


            # 5. combined with k0
            k_all = torch.cat([k_all,k0], dim=1)

            k0_mask = torch.zeros(bs, num_class, num_class).to(device)
            k0_mask[class_mask_inds[0], class_mask_inds[1], class_mask_inds[1]] = 1
            k_mask = torch.cat([k_mask,k0_mask], dim=-1)
            k_all_mask = torch.cat([k_all_mask, class_mask], dim=-1)

            # 6. compute the similarity
            
            k_all = k_all.permute(0, 2, 1).contiguous()
            sim = torch.bmm(q, k_all) / 0.07
            sim = torch.exp(sim)
            sim_sum = (sim * k_all_mask.unsqueeze(1)).sum(dim=-1)
            sim_sum = sim_sum.unsqueeze(2).repeat(1,1,sim.size(2))
            zero_ind = sim_sum == 0
            sim_sum[zero_ind] = 1. # for computation stable

            sim_pos = sim * k_mask
            logit = sim_pos / sim_sum
            zero_ind = logit == 0
            logit[zero_ind] = 1 # for computation stable
            # [bs, num_class]
            log_loss = (torch.log(logit) * k_mask).sum(dim=-1)

            num = k_mask.sum(dim=-1)
            zero_ind = num == 0
            num[zero_ind] = 1
            log_loss = log_loss / num

            count = class_mask.sum()
            loss += log_loss[class_mask_inds].sum() / count
            

        return -loss


    def MCloss(self, feat, hm):


        feat = F.normalize(feat, dim=1, p=2)

        bs = hm.size(0)
        h = hm.size(2)
        w = hm.size(3)

        hm_car = hm[:,0,:,:].reshape(bs, 1, h, w)
        hm_ped = hm[:,1,:,:].reshape(bs, 1, h, w)
        hm_cyc = hm[:,2,:,:].reshape(bs, 1, h, w)
        
        hm_sum = hm.sum(1).reshape(bs, 1, h, w)


        tmp_feat_withmask = feat * hm_sum


        one_ind_car = torch.where(hm_car==1.)
        one_ind_ped = torch.where(hm_ped==1.)
        one_ind_cyc = torch.where(hm_cyc==1.)
        #one_ind_back = torch.where(hm_sum == 0.)
        
        ####(1)acoording hm to find instance feature
        tmp_feat = tmp_feat_withmask.permute(0,2,3,1).contiguous()


        feat_car = tmp_feat[one_ind_car[0],one_ind_car[2],one_ind_car[3]]
        feat_ped = tmp_feat[one_ind_ped[0],one_ind_ped[2],one_ind_ped[3]]
        feat_cyc = tmp_feat[one_ind_cyc[0],one_ind_cyc[2],one_ind_cyc[3]]


        
        feat_car_postive = feat_car.repeat(feat_car.shape[0]-1, 1)
        feat_car_negtive = torch.tensor([]).cuda()
        feat_car_negtive_sub = feat_car.clone()
        for i in range(feat_car.shape[0]-1):
            feat_car_negtive_sub = torch.roll(feat_car_negtive_sub, 1, 0)
            feat_car_negtive = torch.cat((feat_car_negtive, feat_car_negtive_sub), 0)

        num_of_car = feat_car_postive.shape[0]

        feat_ped_postive = feat_ped.repeat(feat_ped.shape[0]-1, 1)
        feat_ped_negtive = torch.tensor([]).cuda()
        feat_ped_negtive_sub = feat_ped.clone()
        for i in range(feat_ped.shape[0]-1):
            feat_ped_negtive_sub = torch.roll(feat_ped_negtive_sub, 1, 0)
            feat_ped_negtive = torch.cat((feat_ped_negtive, feat_ped_negtive_sub), 0)

        num_of_ped = feat_ped_postive.shape[0]

        feat_cyc_postive = feat_cyc.repeat(feat_cyc.shape[0]-1, 1)
        feat_cyc_negtive = torch.tensor([]).cuda()
        feat_cyc_negtive_sub = feat_cyc.clone()
        for i in range(feat_cyc.shape[0]-1):
            feat_cyc_negtive_sub = torch.roll(feat_cyc_negtive_sub, 1, 0)
            feat_cyc_negtive = torch.cat((feat_cyc_negtive, feat_cyc_negtive_sub), 0)


        dims_max = feat_car_postive.shape[0] if feat_car_postive.shape[0] > feat_ped_postive.shape[0] else feat_ped_postive.shape[0]
        dims_max = feat_cyc_postive.shape[0] if feat_cyc_postive.shape[0] > dims_max else dims_max

        feat_car_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_car_postive.shape[0]))(feat_car_postive), 0).reshape(1, -1)
        feat_ped_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_ped_postive.shape[0]))(feat_ped_postive), 0).reshape(1, -1)
        feat_cyc_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_cyc_postive.shape[0]))(feat_cyc_postive), 0).reshape(1, -1)
 

        revert_feat_car_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_car_negtive.shape[0]))(feat_car_negtive), 0).reshape(1, -1)
        revert_feat_ped_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_ped_negtive.shape[0]))(feat_ped_negtive), 0).reshape(1, -1)
        revert_feat_cyc_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_cyc_negtive.shape[0]))(feat_cyc_negtive), 0).reshape(1, -1)
        

        assert feat_car_flatten.shape[1] == feat_ped_flatten.shape[1] == feat_cyc_flatten.shape[1] == revert_feat_car_flatten.shape[1] == revert_feat_ped_flatten.shape[1]\
             == revert_feat_cyc_flatten.shape[1], "Dimesion unmatch!"

  
        q = torch.cat((feat_car_flatten, feat_ped_flatten, feat_cyc_flatten), 0)
        k = torch.cat((revert_feat_car_flatten, revert_feat_ped_flatten, revert_feat_cyc_flatten), 0)


        n = q.size(0)

        logits = torch.mm(q, k.transpose(1, 0))


        logits = logits/ 0.07
        labels = torch.arange(n).cuda().long()
        out = logits.squeeze().contiguous()
        criterion = nn.CrossEntropyLoss().cuda()
        loss = criterion(out, labels)


        return loss


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        x = self.shared_conv(spatial_features_2d)
        
        self.forward_ret_dict['feat'] = x

        pred_dicts = []
        for head in self.heads_list:
            pred_dicts.append(head(x))

        if self.training:
            target_dict = self.assign_targets(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            #orignal heatmap
            self.forward_ret_dict['orignal_heatmap'] = target_dict['heatmaps'][0].clone()
            #update hm
            target_dict_update = self.batch_update_hm_labels(x, target_dict, thresh=0.6, multi_factor=1)

            self.forward_ret_dict['target_dicts'] = target_dict_update

        self.forward_ret_dict['pred_dicts'] = pred_dicts

        if not self.training or self.predict_boxes_when_training:
            pred_dicts = self.generate_predicted_boxes(
                data_dict['batch_size'], pred_dicts
            )

            if self.predict_boxes_when_training:
                rois, roi_scores, roi_labels = self.reorder_rois_for_refining(data_dict['batch_size'], pred_dicts)
                data_dict['rois'] = rois
                data_dict['roi_scores'] = roi_scores
                data_dict['roi_labels'] = roi_labels
                data_dict['has_class_labels'] = True
            else:
                data_dict['final_box_dicts'] = pred_dicts

        return data_dict
