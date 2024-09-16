import numpy as np
import torch
from lib.common.imutils import process_image
from lib.common.render import Render
from lib.dataset.mesh_util import SMPLX
from lib.pixielib.models.SMPLX import SMPLX as PIXIE_SMPLX
from lib.pixielib.pixie import PIXIE
from lib.pixielib.utils.config import cfg as pixie_cfg
from lib.pymafx.core import path_config
from lib.pymafx.models import pymaf_net
from lib.net.geometry import rot6d_to_rotmat

import numpy as np
import torch
from PIL import ImageColor
from pytorch3d.renderer import (
    BlendParams,
    FoVOrthographicCameras,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
    blending,
    look_at_view_transform,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from mmhuman3d.core.renderer.torch3d_renderer.meshes import ParametricMeshes
from mmhuman3d.core.visualization.visualize_smpl import _prepare_colors
from einops import rearrange
from src.utils.util import save_videos_grid
import json
import trimesh
import os

def tensor_to_list(tensor):
    return tensor.squeeze().detach().cpu().numpy().tolist()

@ torch.no_grad()
def save_optimed_video(path, rgb_video, normal_video, smpl_normals):
    # input:
    #     video/smpl_normals: tensor b,c,f,h,w
    rgb_video = rgb_video.detach().cpu()
    normal_video = normal_video.detach().cpu()
    smpl_normals = smpl_normals.detach().cpu()
    videos = torch.cat([rgb_video, normal_video, smpl_normals], dim=0)
    save_videos_grid(videos.repeat(1,1,2,1,1), path, n_rows=3, fps=8)

@ torch.no_grad()
def save_optimed_smpl_param(path, betas,
                    pose, orient,
                    expression, jaw_pose,
                    left_hand_pose,
                    right_hand_pose,
                    trans, scale):
    smplx_param_dict = {
        "betas": tensor_to_list(betas), 
        "global_orient": tensor_to_list(orient),
        "body_pose": tensor_to_list(pose),
        "exp": tensor_to_list(expression),
        "jaw_pose": tensor_to_list(jaw_pose),
        "left_hand_pose": tensor_to_list(left_hand_pose),
        "right_hand_pose": tensor_to_list(right_hand_pose), 
        "trans": tensor_to_list(trans),
        "scale": tensor_to_list(scale),
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(smplx_param_dict, f)

@ torch.no_grad()
def save_optimed_mesh(path, smpl_verts, smpl_faces):
    smpl_mesh = trimesh.Trimesh(vertices=smpl_verts[0].detach().cpu().numpy(), 
                                faces=smpl_faces[0].detach().cpu().numpy(),
                                process=False,
                                maintain_order=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    smpl_mesh.export(path)


class SMPLEstimator:
    def __init__(self, hps_type, device):
        self.hps_type = hps_type
        self.smpl_type = "smplx"
        self.smpl_gender = "neutral"
        self.single = True
        self.device = device
        # smpl related
        self.smpl_data = SMPLX()
        self.SMPLX_object = SMPLX()
        if self.hps_type == "pymafx":
            self.hps = pymaf_net(path_config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
            self.hps.load_state_dict(torch.load(path_config.CHECKPOINT_FILE)["model"], strict=True)
            self.hps.eval()
            pixie_cfg.merge_from_list(["model.n_shape", 10, "model.n_exp", 10])
        elif self.hps_type == "pixie":
            self.hps = PIXIE(config=pixie_cfg, device=self.device)

        self.smpl_model = PIXIE_SMPLX(pixie_cfg.model).to(self.device)
        self.render = Render(size=512, device=self.device)
    
    def estimate_smpl(self, img_pil): 
        arr_dict = process_image(img_pil)
        with torch.no_grad():
            if self.hps_type == "pixie":
                preds_dict = self.hps.forward(arr_dict["img_hps"].to(self.device))
            elif self.hps_type == 'pymafx':
                batch = {k: v.to(self.device) for k, v in arr_dict["img_pymafx"].items()}
                preds_dict, _ = self.hps.forward(batch)

        arr_dict["smpl_faces"] = (
            torch.as_tensor(self.smpl_data.smplx_faces.astype(np.int64)).unsqueeze(0).long().to(
                self.device
            )
        )
        arr_dict["type"] = self.smpl_type

        if self.hps_type == "pymafx":
            output = preds_dict["mesh_out"][-1]
            scale, tranX, tranY = output["pred_cam"].split(1, dim=1)
            arr_dict["betas"] = output["pred_shape"]    #10
            arr_dict["body_pose"] = output["rotmat"][:, 1:22]
            arr_dict["global_orient"] = output["rotmat"][:, 0:1]
            arr_dict["smpl_verts"] = output["smplx_verts"]
            arr_dict["left_hand_pose"] = output["pred_lhand_rotmat"]
            arr_dict["right_hand_pose"] = output["pred_rhand_rotmat"]
            arr_dict['jaw_pose'] = output['pred_face_rotmat'][:, 0:1]
            arr_dict["exp"] = output["pred_exp"]
            # 1.2009, 0.0013, 0.3954

        elif self.hps_type == "pixie":
            arr_dict.update(preds_dict)
            arr_dict["global_orient"] = preds_dict["global_pose"]
            arr_dict["betas"] = preds_dict["shape"]    #200
            arr_dict["smpl_verts"] = preds_dict["vertices"]
            scale, tranX, tranY = preds_dict["cam"].split(1, dim=1)
            # 1.1435, 0.0128, 0.3520

        arr_dict["scale"] = scale.unsqueeze(1)
        arr_dict["trans"] = (
            torch.cat([tranX, tranY, torch.zeros_like(tranX)],
                      dim=1).unsqueeze(1).to(self.device).float()
        )

        # data_dict info (key-shape):
        # scale, tranX, tranY - tensor.float
        # betas - [1,10] / [1, 200]
        # body_pose - [1, 21, 3, 3]
        # jaw_pose - [1, 1, 3, 3]
        # global_orient - [1, 1, 3, 3]
        # smpl_verts - [1, 10475, 3]

        # from rot_mat to rot_6d for better optimization
        N_body, N_pose = arr_dict["body_pose"].shape[:2]
        arr_dict["body_pose"] = arr_dict["body_pose"][:, :, :, :2].reshape(N_body, N_pose, -1)
        arr_dict["global_orient"] = arr_dict["global_orient"][:, :, :, :2].reshape(N_body, 1, -1)

        return arr_dict

    def smpl_forward(self,
                     optimed_betas,
                     optimed_pose,
                     optimed_orient,
                     optimed_trans,
                     expression,
                     jaw_pose,
                     left_hand_pose,
                     right_hand_pose,
                     scale):
        
        N_body, N_pose = optimed_pose.shape[:2]
        optimed_orient_mat = rot6d_to_rotmat(optimed_orient.view(-1, 6)).view(N_body, 1, 3, 3) # B 1 6 -> B 1 3 3
        optimed_pose_mat = rot6d_to_rotmat(optimed_pose.view(-1, 6)).view(N_body, N_pose, 3, 3) # B 21 6 -> B 21 3 3
        smpl_verts, smpl_landmarks, smpl_joints = self.smpl_model(
                    shape_params=optimed_betas, # B 10
                    expression_params=expression, # B 10
                    body_pose=optimed_pose_mat, # B 21 3 3 
                    global_pose=optimed_orient_mat, # B 1 3 3
                    jaw_pose=jaw_pose, # B 1 3 3
                    left_hand_pose=left_hand_pose, # B 15 3 3
                    right_hand_pose=right_hand_pose, # B 15 3 3
                )
        smpl_verts = (smpl_verts + optimed_trans) * scale * torch.tensor([1.0, -1.0, -1.0]).to(self.device) 
        smpl_joints = (smpl_joints + optimed_trans) * scale * torch.tensor([-1.0, 1.0, -1.0]).to(self.device) # 145 3
        smpl_joints_3d = smpl_joints[:, self.smpl_data.smpl_joint_ids_45_pixie, :] # [-1,1]
        
        return smpl_verts, smpl_joints_3d
        
        
        
class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels, fragments, blend_params, znear=-256, zfar=256)

        return images


import matplotlib.pyplot as plt
color_mapping = {
    # tab20b
    10:0, 11:1, 23:2, 27:3, # body
    12:4, 19:5, 3:6, 26:7, 16:7, # left hand
    13:8, 20:9, 15:10, 1:11, 18:11, # right hand
    24:12, 7:13, 9:14, 8:14, # left leg
    2:16, 17:17, 14:18, 22:18, # right leg
    # tab20c
    4:20, 5:20, 6:20, 25:20, # head
    21:21 , # neck
} 

def get_semantic_colors():
    # tab20b
    cmap = plt.cm.get_cmap("tab20b", 20)
    colors = [cmap(i)[:3] for i in range(20)]
    # tab20c
    cmap = plt.cm.get_cmap("tab20c", 20)
    colors.append(cmap(0)[:3])
    colors.append(cmap(1)[:3])
    # black
    colors = [(0.,0.,0.)] + colors
    return torch.Tensor(colors).float()


class SMPLRenderer:
    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.size = size

        # camera setting
        self.dis = 100.0

        self.mesh = None
        self.renderer = None
        self.meshRas = None

    def set_cameras(self, azim_list, elev_list):
        R, T = look_at_view_transform(
            dist = self.dis,
            azim = azim_list,
            elev = elev_list,
        )

        self.R = R.to(self.device)
        self.T = T.to(self.device)
        
        self.cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
        )
        

    def init_renderer(self, cameras, type="normal", bg="gray"):

        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

        if ("normal" in type) or ("semantic" in type) or ("depth" in type):

            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                bin_size=-1, # -1
                faces_per_pixel=1, #30
            ) 
            self.meshRas = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings_mesh)

            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(blend_params=blendparam),
            )

        elif type == "mask":

            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4 - 1.0) * 5e-5, # np.log(1.0 / 1e-4 - 1.0) * 5e-5,
                faces_per_pixel=50, # 50
                bin_size=-1,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=cameras, raster_settings=self.raster_settings_silhouette
            )
            self.renderer = MeshRenderer(
                rasterizer=self.silhouetteRas, shader=SoftSilhouetteShader()
            )


    def load_mesh(self, verts, faces, reverse_normal=False):
        """load mesh into the pytorch3d renderer
        Args:
            verts ([B=1,N,3]): tensor
            faces ([B=1,N,3]): tensor
        """
        ## normal mesh
        self.mesh = Meshes(verts, faces).to(self.device) 
        # normal
        if not reverse_normal:
            self.mesh.textures = TexturesVertex(
                verts_features=(self.mesh.verts_normals_padded() + 1.0) * 0.5
            )
        else:
            self.mesh.textures = TexturesVertex(
                verts_features=(-self.mesh.verts_normals_padded() + 1.0) * 0.5
            ) 
        
        ## semantic mesh
        colors_all = _prepare_colors(palette=['white'],
                                    render_choice='part_silhouette', 
                                    num_person=1, 
                                    num_verts=verts.shape[1], 
                                    model_type='smplx')
        colors_all = colors_all.view(-1, verts.shape[1], 3)
        # color mapping
        color_mapping_tensor = torch.tensor([color_mapping[i] if i in color_mapping else -1 for i in range(max(color_mapping)+1)])
        color_mapping_tensor[0] = 21
        B, N, C = colors_all.shape
        colors_all = color_mapping_tensor[colors_all.view(-1).round().long()].view(B, N, C).float()
        colors_all = colors_all + 1 
        self.semantic_mesh = ParametricMeshes(
            verts=verts,
            faces=faces,
            N_individual_overdide=1,
            model_type="smplx",
            use_nearest=True, # True
            vertex_color=colors_all).to(self.device) 

    
    def get_image(self, type="normal", bg="black"):

        self.init_renderer(self.cameras, type, bg)

        # semantic mesh is different
        if type == "semantic":
            current_mesh = self.semantic_mesh
        elif type == "normal" or type == "mask" or type == "depth":
            current_mesh = self.mesh
        else:
            raise ValueError

        if type == "depth":
            fragments = self.meshRas(current_mesh.extend(len(self.cameras)))
            images = fragments.zbuf[..., 0]

        elif type == "normal": # same renderer for normal & semantic
            images = self.renderer(current_mesh.extend(len(self.cameras)))
            images = images.permute(0, 3, 1, 2) # B,C=4,H,W [0,1]

        elif type == "mask":
            images = self.renderer(current_mesh.extend(len(self.cameras)))[:, :, :, 3] 
            # B,512,512 [0,1]
            
        elif type == "semantic":
            images = self.renderer(current_mesh.extend(len(self.cameras))) # B H W C=4 
            images = images[:,:,:,0] # B H W index
            color_palette = get_semantic_colors().to(self.device)
            B, H, W = images.shape
            images = color_palette[images.view(-1).round().long()].view(B, H, W, 3) # B H W C
            images = images.permute(0, 3, 1, 2) # B C H W
                    
        else:
            print(f"unknown {type}")

        return images

    def render_mask(self, bg="black"): # differentiable mask
        return self.get_image(type="mask", bg=bg) # bg black
    
    def render_normal(self, bg="black"): # differentiable normal, world_space, for normal loss calculation
        # output:
        #     normals: tensor B,C=3,H,W [0,1]
        images = self.get_image(type="normal", bg=bg) # world B,C=4,H,W [0,1]
        images = images[:,:3,:,:]
        return images
    
    
    def project_joints(self, points_3d):
        # input: points_3d tensor B=1 mesh, N points, C=3 [-1,1]
        # output: points_2d tensor B=24 views, N, C=2 [0,1]
        R = self.R.clone().requires_grad_(True) # B=24 3 3
        T = self.T.clone().unsqueeze(2).requires_grad_(True) # B=24 3 1
        
        RT = torch.concatenate([R,T], dim=-1).unsqueeze(1) # B 1 3 4
        ones = torch.ones_like(points_3d[..., 0]).unsqueeze(-1) 
        points_3d = torch.concatenate([points_3d, ones] , dim=-1).unsqueeze(-1) # B=1 N 4 1
        points_2d = torch.matmul(RT, points_3d) # B N 3 1
        points_2d = (points_2d.squeeze(-1)[:,:,:2] + 1.0 ) / 2.0 # B N 2
        
        return points_2d
    
    @torch.no_grad()
    def render_normal_screen_space(self, bg="black", return_mask=False): # normal, screen_space, nvs condition
        # output:
        #     normals: tensor B,C,H,W [0,1]
        #     masks: tensor B,H,W {0,1}
        images = self.get_image(type="normal", bg=bg).detach() # B,C=4,H,W [0,1]
        normals = images[:,:3,:,:] * 2 - 1 # B,3,H,W [-1,1]
        masks = images[:,3,:,:].unsqueeze(1) # B,1,H,W
        masks[masks>0] = 1 # indicate bg & fg
        
        normals = rearrange(normals, "b c h w -> b h w 1 c")
        R = -self.R.clone()*torch.tensor([[1.,1.,1.],[1.,-1.,1.],[1.,1.,1.]]).to(self.device)
        R = rearrange(R, "b c k -> b 1 1 c k ")
        normals = torch.matmul(normals, R) # b h w 1 k convert to screen space
        normals = rearrange(normals, "b h w 1 k -> b k h w")
        normals = (normals + 1) / 2 # fg [0,1]
        normals = normals * masks
        
        if return_mask: # sharp mask (not differentiable)
            return normals, masks.squeeze(1) # BCHW, BHW
        else:
            return normals
    
        
    @torch.no_grad()
    def render_semantic(self, bg="black"): 
        # output:
        #     semantics: tensor B,C,H,W [0,1]
        images = self.get_image(type="semantic", bg=bg).detach() # B,C=4,H,W [0,1]
        images = images[:,:3,:,:]
        return images