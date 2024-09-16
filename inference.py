import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logging.getLogger("trimesh").setLevel(logging.ERROR)

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers import AutoencoderKL, DDIMScheduler, MarigoldNormalsPipeline
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
import os
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_magicman import MagicManPipeline
from src.utils.util import get_camera
from src.utils.util import (
    preprocess_image,
    save_image_seq,
)
import sys
sys.path.append("./thirdparties/econ")
from thirdparties.econ.lib.common.smpl_utils import (
    SMPLEstimator, SMPLRenderer,
    save_optimed_video, save_optimed_smpl_param, save_optimed_mesh,
)
from thirdparties.econ.lib.common.imutils import process_video
from thirdparties.econ.lib.common.config import cfg
from thirdparties.econ.lib.common.train_util import init_loss
from contextlib import contextmanager
import time
from tensorboardX import SummaryWriter

@contextmanager
def timer():
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"【Time】{end_time - start_time:.4f} s")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="configs/inference/inference-plus.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input_path", type=str, default="examples/001.jpg")
    parser.add_argument("--output_path", type=str, default="examples/001")
    args = parser.parse_args()

    return args


def init_module(args, config):
    device = args.device
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # VAE
    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)
    
    # image encoder
    image_encoder = None
    
    # reference unet
    reference_unet = UNet2DConditionModel.from_pretrained_2d(
        config.pretrained_unet_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs,
            resolve=True,
        ),
    ).to(dtype=weight_dtype, device=device)

    # denoising unet
    if config.unet_additional_kwargs.use_motion_module:
        mm_path = config.pretrained_motion_module_path
    else:
        mm_path = ""
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_unet_path,
        mm_path,
        unet_additional_kwargs=OmegaConf.to_container(
            config.unet_additional_kwargs,
            resolve=True,
        ),
    ).to(dtype=weight_dtype, device=device)

    # pose guider for normal maps & semantic segmentation maps
    semantic_guider = PoseGuider(**config.pose_guider_kwargs).to(device="cuda")
    normal_guider = PoseGuider(**config.pose_guider_kwargs).to(device="cuda")

    # scheduler
    sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    # random generator
    generator = torch.manual_seed(args.seed)

    # load pretrained weights
    ckpt_path = config.ckpt_path
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_path, f"denoising_unet.pth"),
            map_location="cpu"
        ),
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(ckpt_path, f"reference_unet.pth"),
            map_location="cpu"
        ),
    )
    semantic_guider.load_state_dict(
        torch.load(
            os.path.join(ckpt_path, f"semantic_guider.pth"),
            map_location="cpu",
        ),
    )
    normal_guider.load_state_dict(
        torch.load(
            os.path.join(ckpt_path, f"normal_guider.pth"),
            map_location="cpu",
        ),
    )
    
    return vae, image_encoder, reference_unet, denoising_unet, semantic_guider, normal_guider, scheduler, generator

def init_pipeline(vae, image_encoder,
                  reference_unet, denoising_unet,
                  semantic_guider, normal_guider, 
                  scheduler, unet_attention_mode,
                  weight_dtype, device):
    pipe = MagicManPipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        semantic_guider=semantic_guider,
        normal_guider=normal_guider,
        scheduler=scheduler,
        unet_attention_mode=unet_attention_mode,
    )
    pipe = pipe.to(device, dtype=weight_dtype)
    
    vae.eval()
    if image_encoder is not None:
        image_encoder.eval()
    reference_unet.eval()
    denoising_unet.eval()
    semantic_guider.eval()
    normal_guider.eval()
        
    return pipe

def init_camera(num_views):
    clip_interval = 360 // num_views
    azim_list = []
    elev_list = []
    camera_list = []
    for i in range(num_views):
        azim = -float(i*clip_interval)
        elev = 0.0
        azim_list.append(azim)
        elev_list.append(elev)
    for azim, elev in zip(azim_list, elev_list):
        camera = get_camera(elev, azim)
        camera_list.append(camera)
    cameras = np.stack(camera_list, axis=0) # (f, 4, 4)
    ref_camera = get_camera(0.0, 0.0) # (4, 4)
    return azim_list, elev_list, cameras, ref_camera

def init_ref_normal(rgb_pil, mask_pil, method="marigold", device="cuda:0"):
    if method == "marigold":
        # TODO: modify the marigold ckpt path
        pipe = MarigoldNormalsPipeline.from_pretrained(
            "/apdcephfs_cq10/share_1330077/hexu/NVS/pretrained_weights/marigold-normals-v0-1", # "prs-eth/marigold-normals-v0-1"
            variant="fp16", 
            torch_dtype=torch.float16
        ).to(device)
        normal_np = pipe(rgb_pil, num_inference_steps=25).prediction
        mask_np = np.array(mask_pil)[None,:,:,None]
        
        def normalize_normal_map(normal_np):
            norms = np.linalg.norm(normal_np, axis=-1, keepdims=True)
            normal_np = normal_np / norms
            normal_np = (normal_np + 1.0) / 2.0
            return normal_np
        
        # normalize & mask bg
        normal_np = normalize_normal_map(normal_np)
        normal_np = normal_np * (mask_np>0)
        normal_pil = Image.fromarray((normal_np[0] * 255).astype(np.uint8)).convert("RGB")
        
        del pipe
        torch.cuda.empty_cache()
        
        return normal_pil
    
    else:
        raise NotImplementedError

def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    
    device = args.device
    width, height = args.W, args.H
    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    
    # module initialization 
    (vae, image_encoder,
     reference_unet, denoising_unet, 
     semantic_guider, normal_guider, 
     scheduler, generator)= init_module(args, config) 

    # pipeline initialization
    pipe = init_pipeline(
        vae=vae,
        image_encoder = image_encoder,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        semantic_guider=semantic_guider,
        normal_guider=normal_guider,
        scheduler=scheduler,
        unet_attention_mode=config.unet_attention_mode,
        weight_dtype=weight_dtype,
        device=device)

    # camera initialization
    num_views = config.num_views
    azim_list, elev_list, cameras, ref_camera = init_camera(num_views)

    # SMPL-X estimator initialization
    smpl_estimator = SMPLEstimator(config.hps_type, device)
    
    # renderer initialization
    smpl_renderer = SMPLRenderer(size=512, device=device)
    smpl_renderer.set_cameras(azim_list, elev_list)
    
    # loss initialization
    losses = init_loss()
    
    input_path = args.input_path
    output_path = args.output_path
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    subject = os.path.basename(input_path).split('.')[0]
    
    with timer():
        writer = SummaryWriter(f'./tbruns/{subject}')
        ##0## reference image preparation
        ref_rgb_pil = Image.open(input_path).convert("RGB")
        ref_rgb_pil, ref_mask_pil = preprocess_image(ref_rgb_pil) # remove background & resize & center
        ref_normal_pil = init_ref_normal(ref_rgb_pil, ref_mask_pil, method="marigold", device=device) # initilize reference normal map
        ref_rgb_pil.save(os.path.join(output_path, f"ref_rgb.png"))
        ref_mask_pil.save(os.path.join(output_path, f"ref_mask.png"))
        ref_normal_pil.save(os.path.join(output_path, f"ref_normal.png"))        
        
        ##1## Initialize SMPL-X parameters
        smpl_dict = smpl_estimator.estimate_smpl(ref_rgb_pil)
        optimed_pose = smpl_dict["body_pose"].requires_grad_(True)
        optimed_trans = smpl_dict["trans"].requires_grad_(True)
        optimed_betas = smpl_dict["betas"].requires_grad_(True)
        optimed_orient = smpl_dict["global_orient"].requires_grad_(True)
        optimizer_smpl = torch.optim.Adam([
            optimed_pose, optimed_trans, optimed_betas, optimed_orient
        ], lr=1e-2, amsgrad=True)
        
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=config.patience,
        )

        def tensor2variable(tensor, device):
            return tensor.requires_grad_(True).to(device)
        expression = tensor2variable(smpl_dict["exp"], device)
        jaw_pose = tensor2variable(smpl_dict["jaw_pose"], device)
        left_hand_pose = tensor2variable(smpl_dict["left_hand_pose"], device)
        right_hand_pose = tensor2variable(smpl_dict["right_hand_pose"], device)
        scale = smpl_dict["scale"]
        smpl_faces = smpl_dict["smpl_faces"]
        
        ##2## Initialize NVS w/o SMPL-X
        output = pipe(
            # ref rgb/normal
            ref_rgb_pil,
            ref_normal_pil,
            # cond semantic/normal
            None,
            None,
            cameras,
            ref_camera,
            width,
            height,
            num_views,
            num_inference_steps=config.intermediate_denoising_steps,
            guidance_scale=config.cfg_scale,
            smplx_guidance_scale=0.0,
            guidance_rescale=config.guidance_rescale,
            generator=generator,
        ) # (b=1, c, f, h, w)
        rgb_video = output.rgb_videos
        normal_video = output.normal_videos
        
        ##3## Iteration
        # In each Iter:
        # 3.0 Prepare NVS data as supervision
        # 3.1 Optimization steps
        #     3.1.1 SMPL-X forward to get updated mesh + differentiable rendering
        #     3.1.2 Calculate loss + optimize SMPL-X params
        # 3.2 NVS generation w/ optimized SMPL-X 
        step_count = 0 # global steps
        iterative_optimization_steps = config.iterative_optimization_steps
        for iter, smplx_guidance_scale in enumerate(config.smplx_guidance_scales):
            final_iter = True if (iter == len(config.smplx_guidance_scales) - 1) else False
            
            ##3.0## Prepare NVS data as supervision in the begining of each iteration
            nvs_data = process_video(
                rgb_video[0].detach().cpu().numpy().transpose(1,2,3,0), # (c, f, h, w)->(f, h, w, c)
                normal_video[0].detach().cpu().numpy().transpose(1,2,3,0),
                ) 
            
            ##3.1## Optimization steps
            total_steps = iterative_optimization_steps[iter]
            step_loop = tqdm(range(total_steps))
            for step in step_loop:
                step_count += 1 # tb
                
                optimizer_smpl.zero_grad()
                
                # SMPL-X forward
                smpl_verts, smpl_joints_3d = smpl_estimator.smpl_forward(
                    optimed_betas=optimed_betas,
                    optimed_pose=optimed_pose,
                    optimed_trans=optimed_trans,
                    optimed_orient=optimed_orient,
                    expression=expression,
                    jaw_pose=jaw_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    scale=scale,
                )
                
                # save initialized SMPL-X mesh
                if iter == 0 and step == 0:
                    mesh_path = f"{output_path}/smplx_initialized.obj"
                    save_optimed_mesh(mesh_path, smpl_verts, smpl_faces)
        

                # differentiable rendering
                smpl_renderer.load_mesh(smpl_verts, smpl_faces) # load updated mesh in each step
                # silhouette loss
                smpl_masks = smpl_renderer.render_mask(bg="black") # (B, 512, 512)
                gt_masks = nvs_data["img_mask"].to(device)
                diff_S = torch.abs(smpl_masks - gt_masks)
                losses["silhouette"]["value"] = diff_S.mean(dim=[1,2]) # (B,)
                
                # Loss weights:
                # - normal weight = 1 (only in the intersected part of SMPL-X body and cloth)
                # - mask weight = 0.1 if self-occlusion else 1
                # - joint weight = 50 if loose cloth else 5
                # - front view larger weight
                _, smpl_masks_fake = smpl_renderer.render_normal_screen_space(
                    bg="black", return_mask=True
                    ) # (B, 3, H, W),  (B, H, W) to get SMPL-X body mask
                # self-occlusion detection
                body_overlap = (gt_masks * smpl_masks_fake).sum(dim=[1, 2]) / smpl_masks_fake.sum(dim=[1, 2]) # (B,)
                body_overlap_flag = body_overlap < cfg.body_overlap_thres
                losses["silhouette"]["weight"] = [0.1 if flag else 1.0 for flag in body_overlap_flag]
                # loose cloth detection
                cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_masks.sum(dim=[1, 2])
                cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres # (B,)
                losses["joint"]["weight"] = [50.0 if flag else 5.0 for flag in cloth_overlap_flag]
                
                # normal loss
                if config.use_normal_loss:
                    body_overlap_mask = gt_masks * smpl_masks_fake # (B, 512, 512) intersected mask of body and cloth
                    smpl_normals = smpl_renderer.render_normal(bg="black") # world space
                    gt_normals = nvs_data["img_normal"].to(device) # (B, 3, 512, 512)
                    diff_N = torch.abs(smpl_normals - gt_normals) * body_overlap_mask.unsqueeze(1) 
                    losses["normal"]["value"] = diff_N.mean(dim=[1,2,3]) # (B,)
                    losses["normal"]["weight"] = [1.0 for _ in range(diff_N.shape[0])]
                                    
                # 2d joint loss
                smpl_joints_2d = smpl_renderer.project_joints(smpl_joints_3d)  # (B=1, 45, 3) [-1,1]-> (B=24, 45, 2) [0,1]
                smpl_lmks = smpl_joints_2d[:, smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 1], :] # select 25 joints (B, 25, 2)
                gt_lmks = nvs_data["landmark"][:, smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(device)
                gt_conf = nvs_data["landmark"][:, smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(device)
                occluded_idx = torch.where(body_overlap_flag)[0] # self-occluded frame
                gt_conf[occluded_idx] *= gt_conf[occluded_idx] > 0.50
                diff_J = torch.norm(gt_lmks - smpl_lmks, dim=2) * gt_conf # (B, 25)
                losses['joint']['value'] = diff_J.mean(dim=1) # (B,)

                # Calculate loss and optimize SMPL-X for this step
                smpl_loss = 0.0
                pbar_desc = "Body Fitting -- "
                loss_items = ["normal", "silhouette", "joint"] if config.use_normal_loss else ["silhouette", "joint"]
                for k in loss_items:
                    losses[k]["weight"][0] = losses[k]["weight"][0] * 10.0 # 10 weight for the front view
                    per_loop_loss = (
                        losses[k]["value"] * torch.tensor(losses[k]["weight"]).to(device)
                    ).mean()
                    pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                    smpl_loss += per_loop_loss
                    writer.add_scalar(f"loss/{k}", per_loop_loss, step_count) 
                pbar_desc += f"Total: {smpl_loss:.3f}"
                step_loop.set_description(pbar_desc)
                writer.add_scalar("loss/total", smpl_loss, step_count) 
                smpl_loss.backward()
                optimizer_smpl.step()
                scheduler_smpl.step(smpl_loss)
                
            ##3.2## Update NVS results at the end of each iteration
            cond_normals, cond_masks = smpl_renderer.render_normal_screen_space(bg="black", return_mask=True) 
            cond_semantics = smpl_renderer.render_semantic(bg="black")
            cond_normal_list = []
            cond_semantic_list = []
            for cond_normal in cond_normals:
                normal_np = (cond_normal.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)
                normal_pil = Image.fromarray(normal_np)
                cond_normal_list.append(normal_pil)
            for cond_semantic in cond_semantics:
                semantic_np = (cond_semantic.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)
                semantic_pil = Image.fromarray(semantic_np)
                cond_semantic_list.append(semantic_pil)
            output = pipe(
                # ref rgb/normal
                ref_rgb_pil,
                ref_normal_pil,
                # cond semantic/normal
                cond_semantic_list,
                cond_normal_list,
                cameras,
                ref_camera,
                width,
                height,
                num_views,
                num_inference_steps=config.final_denoising_steps if final_iter else config.intermediate_denoising_steps,
                guidance_scale=config.cfg_scale,
                smplx_guidance_scale=smplx_guidance_scale,
                guidance_rescale=config.guidance_rescale,
                generator=generator,
            ) # (b=1, c, f, h, w)
            rgb_video = output.rgb_videos 
            normal_video = output.normal_videos
        
        ####Iterations End####
        ### Final results
        with torch.no_grad():
            smpl_verts, smpl_joints_3d = smpl_estimator.smpl_forward(
                        optimed_betas=optimed_betas,
                        optimed_pose=optimed_pose,
                        optimed_trans=optimed_trans,
                        optimed_orient=optimed_orient,
                        expression=expression,
                        jaw_pose=jaw_pose,
                        left_hand_pose=left_hand_pose,
                        right_hand_pose=right_hand_pose,
                        scale=scale,
                    )
            smpl_renderer.load_mesh(smpl_verts, smpl_faces)
            smpl_normals = smpl_renderer.render_normal(bg="black")
        ## video
        video_path = f"{output_path}/{subject}.mp4"
        save_optimed_video( 
                        video_path, 
                        rgb_video, normal_video, 
                        smpl_normals.unsqueeze(0).permute(0,2,1,3,4),
                        )
        ## smpl param
        smpl_param_path = f"{output_path}/smplx_refined.json"
        save_optimed_smpl_param(
            path=smpl_param_path, betas=optimed_betas,
            pose=optimed_pose, orient=optimed_orient,
            expression=expression, jaw_pose=jaw_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            trans=optimed_trans, scale=scale)
        ## mesh
        mesh_path = f"{output_path}/smplx_refined.obj"
        save_optimed_mesh(mesh_path, smpl_verts, smpl_faces)
        
        ## nvs images
        save_image_seq(rgb_video, os.path.join(output_path, "rgb")) 
        save_image_seq(normal_video, os.path.join(output_path, "normal"))
        save_image_seq(cond_normals.unsqueeze(0).permute(0,2,1,3,4), os.path.join(output_path, "smplx_normal"))
        save_image_seq(cond_semantics.unsqueeze(0).permute(0,2,1,3,4), os.path.join(output_path, "smplx_semantic"))
        save_image_seq(cond_masks.unsqueeze(0).unsqueeze(0), os.path.join(output_path, "smplx_mask"))
        
        print(f"【End】{input_path}")


if __name__ == "__main__":
    main()

    