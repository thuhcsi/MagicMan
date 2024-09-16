import numpy as np
import torch
import numpy as np
import torch
from PIL import ImageColor, Image
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
from src.utils.util import save_videos_grid
import json
import trimesh
import os
from rembg import remove

def tensor_to_list(tensor):
    return tensor.squeeze().detach().cpu().numpy().tolist()

@ torch.no_grad()
def save_optimed_video(path, video, smpl_normals):
    # input:
    #     video/smpl_normals: tensor b,c,f,h,w
    video = video.detach().cpu()
    smpl_normals = smpl_normals.detach().cpu()
    merge = (video + smpl_normals)/2.0
    videos = torch.cat([video, smpl_normals, merge], dim=0)
    save_videos_grid(videos.repeat(1,1,2,1,1), path, n_rows=3, fps=6)

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

def load_human_nvs_results(mv_dir, ref_path, imSize, view_num=20):
    # For texture refinement
    all_images = []
    all_masks = []
    all_azims = []
    all_elevs = []
    
    image = np.array(Image.open(ref_path).resize(imSize))[:, :, :3]
    image_masked = remove(image) 
    mask = image_masked[:, :, 3]
    all_images.append(image)
    all_masks.append(mask)
    all_azims.append(0.0)
    all_elevs.append(0.0)
    
    for idx in range(1, view_num): 
        azim = idx * 360 // view_num
        elev = 0.0
        rgb_filepath = os.path.join(mv_dir, f"{azim:03d}.png")
        image =np.array(Image.open(rgb_filepath).resize(imSize))[:, :, :3] 
        image_masked = remove(image) 
        mask = image_masked[:, :, 3]
        
        all_images.append(image)
        all_masks.append(mask)
        all_azims.append(azim * 1.0)
        all_elevs.append(elev)

    return np.stack(all_images), np.stack(all_masks), np.stack(all_azims), np.stack(all_elevs)


class cleanShader(torch.nn.Module):
    def __init__(self, blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs):
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(
            texels, fragments, blend_params, znear=-256, zfar=256
        )
        return images


class TexturedMeshRenderer:
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
        ) # B=24,3,3

        self.R = R.to(self.device) 
        self.T = T.to(self.device)
        
        self.cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
        )
        

    def init_renderer(self, cameras, type="mesh", bg="gray"):

        blendparam = BlendParams(1e-4, 1e-8, np.array(ImageColor.getrgb(bg)) / 255.0)

        if ("mesh" in type) or ("depth" in type) or ("rgb" in type):

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


    def load_mesh(self, verts, faces, vert_colors):
        """load mesh into the pytorch3d renderer
        Args:
            verts ([B=1,N,3]): tensor
            faces ([B=1,N,3]): tensor
        """

        self.mesh = Meshes(verts, faces).to(self.device) 

        self.mesh.textures = TexturesVertex(
            verts_features=vert_colors
        )

    
    def get_image(self, type="rgb", bg="black"):

        self.init_renderer(self.cameras, type, bg)

        current_mesh = self.mesh

        if type == "depth":
            fragments = self.meshRas(current_mesh.extend(len(self.cameras)))
            images = fragments.zbuf[..., 0]

        elif type == "rgb":
            images = self.renderer(current_mesh.extend(len(self.cameras)))
            images = images.permute(0, 3, 1, 2) # B,C=4,H,W [0,1]

        elif type == "mask":
            images = self.renderer(current_mesh.extend(len(self.cameras)))[:, :, :, 3] 
            # B,512,512 [0,1]
        else:
            print(f"unknown {type}")

        return images

    def render_mesh(self, bg="black", return_mask=True): 
        # output:
        #     normals: tensor B,C=3,H,W [0,1]
        images = self.get_image(type="rgb", bg=bg)
        mask = images[:, 3, :, :] > 0.5
        images = images[:, :3, :, :]
        
        if return_mask:
            return images, mask
        else:
            return images
    
    
        
        