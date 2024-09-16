# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

class Format:
    end = '\033[0m'
    start = '\033[4m'


def init_loss():

    losses = {
    # Cloth: chamfer distance
        "cloth": {"weight": 1e3, "value": 0.0},
    # Stiffness: [RT]_v1 - [RT]_v2 (v1-edge-v2)
        "stiff": {"weight": 1e5, "value": 0.0},
    # Cloth: det(R) = 1
        "rigid": {"weight": 1e5, "value": 0.0},
    # Cloth: edge length
        "edge": {"weight": 0, "value": 0.0},
    # Cloth: normal consistency
        "nc": {"weight": 0, "value": 0.0},
    # Cloth: laplacian smoonth
        "lapla": {"weight": 1e2, "value": 0.0},
    # Body: Normal_pred - Normal_smpl
        "normal": {"weight": 1e0, "value": 0.0},
    # Body: Silhouette_pred - Silhouette_smpl
        "silhouette": {"weight": 1e0, "value": 0.0},
    # Joint: reprojected joints difference
        "joint": {"weight": 5e0, "value": 0.0},
    }

    return losses
