from __future__ import print_function, absolute_import

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseGenerator(Dataset):
    def __init__(self, poses_3d, poses_2d, actions, intrinsics):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)
        
        self._intrinsics = []
        for i in range(len(poses_3d)):
            self._intrinsics.append(np.repeat([intrinsics[i]], poses_3d[i].shape[0], axis=0))
        self._intrinsics = np.vstack(self._intrinsics)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions) == len(self._intrinsics)
        
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]
        out_intrinsics = self._intrinsics[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()
        out_intrinsics = torch.from_numpy(out_intrinsics).float()

        return out_pose_3d, out_pose_2d, out_action, out_intrinsics

    def __len__(self):
        return len(self._actions)

class PoseGenerator_viz(Dataset):
    def __init__(self, poses_3d, poses_2d, actions):
        assert poses_3d is not None

        self._poses_3d = np.concatenate(poses_3d)
        self._poses_2d = np.concatenate(poses_2d)
        self._actions = reduce(lambda x, y: x + y, actions)

        assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        print('Generating {} poses...'.format(len(self._actions)))

    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d, out_action

    def __len__(self):
        return len(self._actions)
