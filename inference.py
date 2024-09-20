import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from tensorboardX import SummaryWriter
import os
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_magicman import MagicManPipeline
from src.utils.util import get_camera, preprocess_image, save_image_seq
from diffusers import AutoencoderKL, DDIMScheduler, MarigoldNormalsPipeline
import sys
sys.path.append("./thirdparties/econ")
from thirdparties.econ.lib.common.smpl_utils import (
    SMPLEstimator, SMPLRenderer,
    save_optimed_video, save_optimed_smpl_param, save_optimed_mesh,
)

from thirdparties.econ.lib.common.imutils import process_video
from thirdparties.econ.lib.common.config import cfg
from contextlib import contextmanager
import time
import argparse
import face_recognition
from scipy.spatial import Delaunay
import cv2
from FaceProcess import FaceProcessor
from normalmap_vis import Config, ModelManager, ImageProcessor
import numpy as np

import gc
import psutil
import GPUtil
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class Inference:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device(args.device)
        self.width, self.height = args.W, args.H
        self.weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32
        
        self.init_normalize()

        
        self.subject = os.path.basename(args.input_path).split('.')[0]
        self.writer = SummaryWriter(f'./tbruns/{self.subject}')

    def log_memory_usage(self, step):
        process = psutil.Process()
        gpu = GPUtil.getGPUs()[0]
        
        cpu_memory = process.memory_info().rss / 1e9  # Convert to GB
        gpu_memory = gpu.memoryUsed / 1e3  # Convert to GB
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1e9  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1e9  # Convert to GB
        
        print(f"Step {step} Memory Usage:")
        print(f"  CPU Memory: {cpu_memory:.2f} GB")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        print(f"  Torch Allocated: {allocated:.2f} GB")
        print(f"  Torch Reserved: {reserved:.2f} GB")
        
    def init_modules_pipelines(self):
        self.init_modules()
        self.init_pipeline()
        self.init_camera()
        self.init_smpl()
        self.init_renderer()
        self.init_losses()

        # Convert models to half precision
        self.vae = self.vae.to(dtype=self.weight_dtype)
        self.reference_unet = self.reference_unet.to(dtype=self.weight_dtype)
        self.denoising_unet = self.denoising_unet.to(dtype=self.weight_dtype)
        self.semantic_guider = self.semantic_guider.to(dtype=self.weight_dtype)
        self.normal_guider = self.normal_guider.to(dtype=self.weight_dtype)

        torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)



    def init_normalize(self):
        self.processor = ImageProcessor(normal_model_name="1b",seg_model_name="fg-bg-1b")


    def init_modules(self):
        config = self.config
        device = self.device
        
        self.vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(device, dtype=self.weight_dtype)
        self.image_encoder = None  # Placeholder for image encoder if needed
        
        self.reference_unet = UNet2DConditionModel.from_pretrained_2d(
            config.pretrained_unet_path,
            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs, resolve=True),
        ).to(dtype=self.weight_dtype, device=device)
        
        mm_path = config.pretrained_motion_module_path if config.unet_additional_kwargs.use_motion_module else ""
        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            config.pretrained_unet_path,
            mm_path,
            unet_additional_kwargs=OmegaConf.to_container(config.unet_additional_kwargs, resolve=True),
        ).to(dtype=self.weight_dtype, device=device)
        
        self.semantic_guider = PoseGuider(**config.pose_guider_kwargs).to(device=device)
        self.normal_guider = PoseGuider(**config.pose_guider_kwargs).to(device=device)
        
        sched_kwargs = OmegaConf.to_container(config.noise_scheduler_kwargs)
        self.scheduler = DDIMScheduler(**sched_kwargs)
        
        self.generator = torch.Generator(device=device).manual_seed(self.args.seed)


        self.fp = FaceProcessor()
        
        # Load pretrained weights
        ckpt_path = config.ckpt_path
        self.denoising_unet.load_state_dict(torch.load(os.path.join(ckpt_path, "denoising_unet.pth"), map_location="cpu"))
        self.reference_unet.load_state_dict(torch.load(os.path.join(ckpt_path, "reference_unet.pth"), map_location="cpu"))
        self.semantic_guider.load_state_dict(torch.load(os.path.join(ckpt_path, "semantic_guider.pth"), map_location="cpu"))
        self.normal_guider.load_state_dict(torch.load(os.path.join(ckpt_path, "normal_guider.pth"), map_location="cpu"))
        


  
    def init_pipeline(self):
        self.pipe = MagicManPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            reference_unet=self.reference_unet,
            denoising_unet=self.denoising_unet,
            semantic_guider=self.semantic_guider,
            normal_guider=self.normal_guider,
            scheduler=self.scheduler,
            unet_attention_mode=self.config.unet_attention_mode,
        ).to(self.device, dtype=self.weight_dtype)
        
        self.vae.eval()
        if self.image_encoder is not None:
            self.image_encoder.eval()
        self.reference_unet.eval()
        self.denoising_unet.eval()
        self.semantic_guider.eval()
        self.normal_guider.eval()
        
    def init_camera(self):
        num_views = self.config.num_views
        clip_interval = 360 // num_views
        self.azim_list = [-float(i*clip_interval) for i in range(num_views)]
        self.elev_list = [0.0] * num_views
        self.cameras = np.stack([get_camera(elev, azim) for azim, elev in zip(self.elev_list, self.azim_list)], axis=0)
        self.ref_camera = get_camera(0.0, 0.0)
        
    def init_smpl(self):
        self.smpl_estimator = SMPLEstimator(self.config.hps_type, self.device)
        
    def init_renderer(self):
        self.smpl_renderer = SMPLRenderer(size=512, device=self.device)
        self.smpl_renderer.set_cameras(self.azim_list, self.elev_list)
        
    def init_losses(self):
        self.losses = {
            "silhouette": {"value": None, "weight": None},
            "normal": {"value": None, "weight": None},
            "joint": {"value": None, "weight": None}
        }
        
    def prepare_reference_image(self):
        input_path = self.args.input_path
        output_path = self.args.output_path
        os.makedirs(output_path, exist_ok=True)
        
        self.ref_rgb_pil = Image.open(input_path).convert("RGB")
        self.ref_rgb_pil, self.ref_mask_pil = preprocess_image(self.ref_rgb_pil)
        self.ref_normal_pil = self.init_ref_normal(self.ref_rgb_pil, self.ref_mask_pil)
        
        self.ref_rgb_pil.save(os.path.join(output_path, "ref_rgb.png"))
        self.ref_mask_pil.save(os.path.join(output_path, "ref_mask.png"))
        self.ref_normal_pil.save(os.path.join(output_path, "ref_normal.png"))
 

    # use sapiens - normal  facebook https://about.meta.com/realitylabs/codecavatars/sapiens/
    def init_ref_normal(self, rgb_pil, mask_pil):
        # processor = ImageProcessor()
        normal_map = self.processor.process_image(
            self.args.input_path,
            normal_model_name="0.3b",
            seg_model_name="fg-bg-1b"  
        )
        normal_map_vis = self.processor.visualize_normal_map(normal_map)
        return Image.fromarray(normal_map_vis)

    @staticmethod
    def normalize_normal_map(normal_np):
        norms = np.linalg.norm(normal_np, axis=-1, keepdims=True)
        normal_np = normal_np / (norms + 1e-6)  # Add small epsilon to avoid division by zero
        normal_np = (normal_np + 1.0) / 2.0
        return normal_np
        
    def initialize_nvs(self):
        self.output = self.pipe(
            self.ref_rgb_pil,
            self.ref_normal_pil,
            None,
            None,
            self.cameras,
            self.ref_camera,
            self.width,
            self.height,
            self.config.num_views,
            num_inference_steps=self.config.intermediate_denoising_steps,
            guidance_scale=self.config.cfg_scale,
            smplx_guidance_scale=0.0,
            guidance_rescale=self.config.guidance_rescale,
            generator=self.generator,
        )
        self.rgb_video = self.output.rgb_videos
        self.normal_video = self.output.normal_videos
     


        
    def calculate_losses(self, smpl_masks, gt_masks, smpl_verts, smpl_joints_3d, nvs_data):
        # Ensure all masks have the same dimensions
        target_size = gt_masks.shape[-2:]  # Use gt_masks size as the target
        
        if smpl_masks.shape[-2:] != target_size:
            smpl_masks = F.interpolate(smpl_masks.unsqueeze(1).float(), size=target_size, mode='bilinear').squeeze(1)
        
        # Silhouette loss
        diff_S = torch.abs(smpl_masks - gt_masks)
        self.losses["silhouette"]["value"] = diff_S.mean(dim=[1,2])

        # Self-occlusion detection
        _, smpl_masks_fake = self.smpl_renderer.render_normal_screen_space(bg="black", return_mask=True)
        
        # Ensure smpl_masks_fake has the same size as gt_masks
        if smpl_masks_fake.shape[-2:] != target_size:
            smpl_masks_fake = F.interpolate(smpl_masks_fake.unsqueeze(1).float(), size=target_size, mode='bilinear').squeeze(1)
        
        body_overlap = (gt_masks * smpl_masks_fake).sum(dim=[1, 2]) / smpl_masks_fake.sum(dim=[1, 2])
        body_overlap_flag = body_overlap < cfg.body_overlap_thres
        self.losses["silhouette"]["weight"] = [0.1 if flag else 1.0 for flag in body_overlap_flag]

        # Loose cloth detection
        cloth_overlap = diff_S.sum(dim=[1, 2]) / gt_masks.sum(dim=[1, 2])
        cloth_overlap_flag = cloth_overlap > cfg.cloth_overlap_thres
        self.losses["joint"]["weight"] = [50.0 if flag else 5.0 for flag in cloth_overlap_flag]

        # Normal loss
        if self.config.use_normal_loss:
            body_overlap_mask = gt_masks * smpl_masks_fake
            smpl_normals = self.smpl_renderer.render_normal(bg="black")
            
            # Ensure smpl_normals has the same size as gt_masks
            if smpl_normals.shape[-2:] != target_size:
                smpl_normals = F.interpolate(smpl_normals, size=target_size, mode='bilinear')
            
            gt_normals = nvs_data["img_normal"].to(self.device)
            
            # Ensure gt_normals has the same size
            if gt_normals.shape[-2:] != target_size:
                gt_normals = F.interpolate(gt_normals, size=target_size, mode='bilinear')
            
            diff_N = torch.abs(smpl_normals - gt_normals) * body_overlap_mask.unsqueeze(1)
            self.losses["normal"]["value"] = diff_N.mean(dim=[1,2,3])
            self.losses["normal"]["weight"] = [1.0 for _ in range(diff_N.shape[0])]

        # 2D joint loss
        smpl_joints_2d = self.smpl_renderer.project_joints(smpl_joints_3d)
        smpl_lmks = smpl_joints_2d[:, self.smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 1], :]
        gt_lmks = nvs_data["landmark"][:, self.smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 0], :2].to(self.device)
        gt_conf = nvs_data["landmark"][:, self.smpl_estimator.SMPLX_object.ghum_smpl_pairs[:, 0], -1].to(self.device)
        occluded_idx = torch.where(body_overlap_flag)[0]
        gt_conf[occluded_idx] *= gt_conf[occluded_idx] > 0.50
        diff_J = torch.norm(gt_lmks - smpl_lmks, dim=2) * gt_conf
        self.losses['joint']['value'] = diff_J.mean(dim=1)

        # Add identity loss (if implemented)
        if hasattr(self, 'ref_embedding') and self.ref_embedding is not None:
            identity_loss = self.fp.identity_consistency_loss(self.rgb_video[0, :, 0], self.ref_embedding)
            self.losses['identity'] = {'value': torch.tensor(identity_loss, device=self.device), 'weight': self.config.identity_weight}
        else:
            self.losses['identity'] = {'value': torch.tensor(0.0, device=self.device), 'weight': 0.0}

        
    def calculate_total_loss(self):
        total_loss = 0.0
        pbar_desc = "Body Fitting -- "
        loss_items = ["normal", "silhouette", "joint"] if self.config.use_normal_loss else ["silhouette", "joint"]
        for k in loss_items:
            if self.losses[k]["value"] is not None:
                self.losses[k]["weight"][0] *= 10.0  # 10x weight for the front view
                per_loop_loss = (self.losses[k]["value"] * torch.tensor(self.losses[k]["weight"]).to(self.device)).mean()
                pbar_desc += f"{k}: {per_loop_loss:.3f} | "
                total_loss += per_loop_loss
                self.writer.add_scalar(f"loss/{k}", per_loop_loss, self.step_count)
        pbar_desc += f"Total: {total_loss:.3f}"
        tqdm.write(pbar_desc)
        self.writer.add_scalar("loss/total", total_loss, self.step_count)
        return total_loss
        
    def update_nvs(self, final_iter=False):
        with torch.no_grad():
            self.cond_normals, self.cond_masks = self.smpl_renderer.render_normal_screen_space(bg="black", return_mask=True)
            self.cond_semantics = self.smpl_renderer.render_semantic(bg="black")
        
    
        cond_normal_list = []
        for cond_normal in self.cond_normals:
            img = Image.fromarray((cond_normal.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0))
            normal_map = self.processor.process_the_image(
                img,
                normal_model_name="0.3b",
                seg_model_name="fg-bg-1b"
            )
            normal_map_vis = self.processor.visualize_normal_map(normal_map)
            cond_normal_list.append(Image.fromarray(normal_map_vis))
        
        cond_semantic_list = [Image.fromarray((cond_semantic.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)) 
                            for cond_semantic in self.cond_semantics]
        
        self.output = self.pipe(
            self.ref_rgb_pil,
            self.ref_normal_pil,
            cond_semantic_list,
            cond_normal_list,
            self.cameras,
            self.ref_camera,
            self.width,
            self.height,
            self.config.num_views,
            num_inference_steps=self.config.final_denoising_steps if final_iter else self.config.intermediate_denoising_steps,
            guidance_scale=self.config.cfg_scale,
            smplx_guidance_scale=self.current_smplx_guidance_scale,
            guidance_rescale=self.config.guidance_rescale,
            generator=self.generator,
        )
        self.rgb_video = self.output.rgb_videos
        self.normal_video = self.output.normal_videos

    def save_results(self):
        output_path = self.args.output_path
        
        # Load the last saved intermediate results
        last_iteration = len(self.config.smplx_guidance_scales) - 1
        rgb_video = np.load(f"{output_path}/rgb_video_{last_iteration}.npy")
        normal_video = np.load(f"{output_path}/normal_video_{last_iteration}.npy")
        smpl_normals = np.load(f"{output_path}/cond_normals_{last_iteration}.npy")

        # Save video
        video_path = f"{output_path}/{self.subject}.mp4"
        save_optimed_video(
            video_path,
            rgb_video,
            normal_video,
            smpl_normals
        )

        # Save NVS images
        save_image_seq(rgb_video, os.path.join(output_path, "rgb"))
        save_image_seq(normal_video, os.path.join(output_path, "normal"))
        save_image_seq(smpl_normals, os.path.join(output_path, "smplx_normal"))
        save_image_seq(np.load(f"{output_path}/cond_semantics_{last_iteration}.npy"), os.path.join(output_path, "smplx_semantic"))
        save_image_seq(np.load(f"{output_path}/cond_masks_{last_iteration}.npy"), os.path.join(output_path, "smplx_mask"))

    def run(self):
        # self.prepare_reference_image()

        
        # Initialize SMPL-X parameters
        smpl_dict = self.smpl_estimator.estimate_smpl(self.ref_rgb_pil)

        self.log_memory_usage("Initial")
        self.initialize_nvs()
        self.log_memory_usage("After NVS Init")

        self.log_memory_usage("After SMPL Estimation")
        self.optimed_pose = smpl_dict["body_pose"].requires_grad_(True)
        self.optimed_trans = smpl_dict["trans"].requires_grad_(True)
        self.optimed_betas = smpl_dict["betas"].requires_grad_(True)
        self.optimed_orient = smpl_dict["global_orient"].requires_grad_(True)
        
        self.optimizer_smpl = torch.optim.Adam([
            self.optimed_pose, self.optimed_trans, self.optimed_betas, self.optimed_orient
        ], lr=1e-2, amsgrad=True)
        
        
        self.scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5,
            patience=self.config.patience,
        )

        self.expression = self.tensor2variable(smpl_dict["exp"])
        self.jaw_pose = self.tensor2variable(smpl_dict["jaw_pose"])
        self.left_hand_pose = self.tensor2variable(smpl_dict["left_hand_pose"])
        self.right_hand_pose = self.tensor2variable(smpl_dict["right_hand_pose"])
        self.scale = smpl_dict["scale"]
        self.smpl_faces = smpl_dict["smpl_faces"]

        self.step_count = 0
        for iter, smplx_guidance_scale in enumerate(self.config.smplx_guidance_scales):
            self.current_smplx_guidance_scale = smplx_guidance_scale
            final_iter = iter == len(self.config.smplx_guidance_scales) - 1
            
            nvs_data = self.process_nvs_data()
            self.log_memory_usage(f"Iteration {iter} - After NVS Data Processing")
            smpl_verts, smpl_joints_3d = self.adaptive_refinement(nvs_data)
            self.log_memory_usage(f"Iteration {iter} - After Adaptive Refinement")
            
            if final_iter:
                self.smpl_verts = smpl_verts 
                self.save_intermediate_results(iter)
                reconstructed_mesh = self.multi_scale_reconstruction(
                    smpl_verts, self.smpl_faces, self.rgb_video, self.normal_video,
                    scales=self.config.reconstruction_scales
                )
                self.smpl_verts = reconstructed_mesh
                self.log_memory_usage(f"Iteration {iter} - After Multi-Scale Reconstruction")
            
            with torch_gc_context():
                self.update_nvs(final_iter)
                self.save_intermediate_results(iter)

                self.log_memory_usage(f"Iteration {iter} - After NVS Update")
        
        self.save_final_results()
        self.log_memory_usage("Final")
        print(f"【End】{self.args.input_path}")


    def save_final_results(self):
        output_path = self.args.output_path
        
        # Save SMPL parameters
        smpl_param_path = f"{output_path}/smplx_refined.json"
        save_optimed_smpl_param(
            path=smpl_param_path,
            betas=self.optimed_betas,
            pose=self.optimed_pose,
            orient=self.optimed_orient,
            expression=self.expression,
            jaw_pose=self.jaw_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            trans=self.optimed_trans,
            scale=self.scale
        )

        # Save mesh
        mesh_path = f"{output_path}/smplx_refined.obj"
        save_optimed_mesh(mesh_path, self.smpl_verts, self.smpl_faces)

        print(f"Final results saved to {output_path}")
        
    def tensor2variable(self, tensor):
        return tensor.requires_grad_(True).to(self.device)

    def process_nvs_data(self):
        return process_video(
            self.rgb_video[0].detach().cpu().numpy().transpose(1,2,3,0),
            self.normal_video[0].detach().cpu().numpy().transpose(1,2,3,0),
        )


    def multi_scale_reconstruction(self, smpl_verts, smpl_faces, rgb_video, normal_video, scales=[1]):
        print(f"multi_scale_reconstruction: Initial shapes - smpl_verts: {smpl_verts.shape}, rgb_video: {rgb_video.shape}, normal_video: {normal_video.shape}")
        B, T, C, H, W = rgb_video.shape
        for scale in scales:
            print(f"Processing scale: {scale}")
            # Reshape to 4D for interpolation
            rgb_video_reshaped = rgb_video.view(B * T, C, H, W)
            normal_video_reshaped = normal_video.view(B * T, C, H, W)
            
            # Downsampling
            rgb_video_downsampled = F.interpolate(rgb_video_reshaped, scale_factor=1/scale, mode='bilinear')
            normal_video_downsampled = F.interpolate(normal_video_reshaped, scale_factor=1/scale, mode='bilinear')
            
            # Reshape back to 5D
            _, _, new_H, new_W = rgb_video_downsampled.shape
            rgb_video_downsampled = rgb_video_downsampled.view(B, T, C, new_H, new_W)
            normal_video_downsampled = normal_video_downsampled.view(B, T, C, new_H, new_W)
            
            print(f"Downsampled shapes - rgb: {rgb_video_downsampled.shape}, normal: {normal_video_downsampled.shape}")
            
            reconstructed_mesh = self.perform_reconstruction(smpl_verts, smpl_faces, rgb_video_downsampled, normal_video_downsampled)
            print(f"Reconstructed mesh shape: {reconstructed_mesh.shape}")
            
            smpl_verts = self.upsample_mesh(reconstructed_mesh, scale)
            print(f"Upsampled mesh shape: {smpl_verts.shape}")
        
        return reconstructed_mesh


    def save_intermediate_results(self, iteration):
        print("save_intermediate_results:")
        output_path = self.args.output_path
        np.save(f"{output_path}/smpl_verts_{iteration}.npy", self.smpl_verts.detach().cpu().numpy())
        np.save(f"{output_path}/rgb_video_{iteration}.npy", self.rgb_video.detach().cpu().numpy())
        np.save(f"{output_path}/normal_video_{iteration}.npy", self.normal_video.detach().cpu().numpy())
        np.save(f"{output_path}/cond_normals_{iteration}.npy", self.cond_normals.detach().cpu().numpy())
        np.save(f"{output_path}/cond_semantics_{iteration}.npy", self.cond_semantics.detach().cpu().numpy())
        np.save(f"{output_path}/cond_masks_{iteration}.npy", self.cond_masks.detach().cpu().numpy())
        
    def perform_reconstruction(self, smpl_verts, smpl_faces, rgb_video, normal_video):
        print(f"perform_reconstruction: Input shapes - smpl_verts: {smpl_verts.shape}, rgb_video: {rgb_video.shape}, normal_video: {normal_video.shape}")
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Use mixed precision
            features = torch.cat([rgb_video, normal_video], dim=1).to(self.device)
            print(f"Concatenated features shape: {features.shape}")
            
            if not hasattr(self, 'conv3d'):
                self.conv3d = nn.Conv3d(features.shape[1], 32, kernel_size=3, padding=1).to(self.device)
            
            features = self.conv3d(features)
            print(f"Features after Conv3D shape: {features.shape}")
            
            projected_features = self.project_features_to_3d(features, smpl_verts)
            print(f"Projected features shape: {projected_features.shape}")
            
            # Add a linear layer to reduce the feature dimensions from 32 to 3
            if not hasattr(self, 'feature_reducer'):
                self.feature_reducer = nn.Linear(32, 3).to(self.device)
            
            reduced_features = self.feature_reducer(projected_features)
            print(f"Reduced features shape: {reduced_features.shape}")
            
            refined_verts = smpl_verts + reduced_features
            print(f"Refined verts shape: {refined_verts.shape}")

        # Clear unnecessary variables
        # del features, projected_features, reduced_features
        # torch.cuda.empty_cache()
        
        return refined_verts

    def project_features_to_3d(self, features, smpl_verts):
        B, C, D, H, W = features.shape
        V = smpl_verts.shape[1]
        print(f"project_features_to_3d: Input shapes - features: {features.shape}, smpl_verts: {smpl_verts.shape}")
        
        features_flat = features.view(B, C, -1)
        print(f"Flattened features shape: {features_flat.shape}")
        
        # Process in chunks to avoid OOM
        chunk_size = 1000  # Adjust this value based on your available memory
        num_chunks = (V + chunk_size - 1) // chunk_size
        print(f"Processing in {num_chunks} chunks of size {chunk_size}")
        
        aggregated = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, V)
            
            chunk_projected = features_flat.unsqueeze(2).expand(-1, -1, end_idx - start_idx, -1)
            print(f"Chunk {i+1}/{num_chunks} projected shape: {chunk_projected.shape}")
            
            chunk_aggregated = chunk_projected.mean(dim=-1)
            print(f"Chunk {i+1}/{num_chunks} aggregated shape: {chunk_aggregated.shape}")
            
            aggregated.append(chunk_aggregated)
        
        aggregated = torch.cat(aggregated, dim=2)
        print(f"Final aggregated shape: {aggregated.shape}")
        
        result = aggregated.permute(0, 2, 1)
        print(f"Final result shape: {result.shape}")
        return result  # Shape: (B, V, C)

    def upsample_mesh(self, verts, scale):
        verts_np = verts.detach().cpu().numpy()
        faces = Delaunay(verts_np[:, :2]).simplices
        
        new_verts = verts_np.copy()
        new_faces = []
        
        for face in faces:
            v1, v2, v3 = verts_np[face]
            
            v12 = (v1 + v2) / 2
            v23 = (v2 + v3) / 2
            v31 = (v3 + v1) / 2
            
            new_verts = np.vstack([new_verts, v12, v23, v31])
            
            n = len(new_verts)
            new_faces.extend([
                [face[0], n-3, n-1],
                [face[1], n-2, n-3],
                [face[2], n-1, n-2],
                [n-3, n-2, n-1]
            ])
        
        return torch.tensor(new_verts, device=self.device), torch.tensor(new_faces, device=self.device)

    

    def adaptive_refinement(self, nvs_data, max_iterations=4, threshold=1e-4):
        print("adaptive_refinement...")
        prev_loss = float('inf')
        
        for i in range(max_iterations):
            print("i:", i)
            with torch.no_grad():
                smpl_verts, smpl_joints_3d = self.smpl_estimator.smpl_forward(
                    optimed_betas=self.optimed_betas,
                    optimed_pose=self.optimed_pose,
                    optimed_trans=self.optimed_trans,
                    optimed_orient=self.optimed_orient,
                    expression=self.expression,
                    jaw_pose=self.jaw_pose,
                    left_hand_pose=self.left_hand_pose,
                    right_hand_pose=self.right_hand_pose,
                    scale=self.scale,
                )
            
            self.smpl_renderer.load_mesh(smpl_verts, self.smpl_faces)
            smpl_masks = self.smpl_renderer.render_mask(bg="black")
            gt_masks = nvs_data["img_mask"].to(self.device)
            
            self.calculate_losses(smpl_masks, gt_masks, smpl_verts, smpl_joints_3d, nvs_data)
            
            total_loss = self.calculate_total_loss()
            
            if abs(prev_loss - total_loss.item()) < threshold:
                print(f"Converged after {i+1} iterations")
                break
            
            self.optimizer_smpl.zero_grad()
            total_loss.backward()
            self.optimizer_smpl.step()
            self.scheduler_smpl.step(total_loss)
            
            prev_loss = total_loss.item()

            self.update_nvs()

            # Cleanup
            del smpl_masks, gt_masks, total_loss
            torch.cuda.empty_cache()
            gc.collect()

        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

        return smpl_verts, smpl_joints_3d
    

@contextmanager
def timer():
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"【Time】{end_time - start_time:.4f} s")

@contextmanager
def torch_gc_context():
    try:
        yield
    finally:
        torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config", type=str, default="configs/inference/inference-plus.yaml")
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--input_path", type=str, default="examples/ref_rgb.png")
    parser.add_argument("--output_path", type=str, default="examples/image_0")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    
    inference = Inference(args, config)
    inference.prepare_reference_image()
    inference.init_modules_pipelines()
    inference.run()

    try:
        inference.save_results()
    except Exception as e:
        print(f"Error occurred while saving results: {e}")
        print("You can try to run the save_results method separately later.")

if __name__ == "__main__":
    main()