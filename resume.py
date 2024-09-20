import numpy as np
import torch
import os
from omegaconf import OmegaConf
from inference import Inference, parse_args

class ResumeInference(Inference):
    def __init__(self, args, config):
        super().__init__(args, config)

    def load_intermediate_results(self, iteration):
        output_path = './examples/image_0'
        self.smpl_verts = torch.from_numpy(np.load(f"{output_path}/smpl_verts_{iteration}.npy")).to(self.device)
        self.rgb_video = torch.from_numpy(np.load(f"{output_path}/rgb_video_{iteration}.npy")).to(self.device)
        self.normal_video = torch.from_numpy(np.load(f"{output_path}/normal_video_{iteration}.npy")).to(self.device)
        self.cond_normals = torch.from_numpy(np.load(f"{output_path}/cond_normals_{iteration}.npy")).to(self.device)
        self.cond_semantics = torch.from_numpy(np.load(f"{output_path}/cond_semantics_{iteration}.npy")).to(self.device)
        self.cond_masks = torch.from_numpy(np.load(f"{output_path}/cond_masks_{iteration}.npy")).to(self.device)

    def resume_from_intermediate(self):
        # Load the last iteration results
        last_iteration = 0 #len(self.config.smplx_guidance_scales) - 1
        self.load_intermediate_results(last_iteration)


        # Initialize SMPL-X parameters
        smpl_dict = self.smpl_estimator.estimate_smpl(self.ref_rgb_pil)
        self.smpl_faces = smpl_dict["smpl_faces"]
        
        # Perform multi-scale reconstruction
        reconstructed_mesh = self.multi_scale_reconstruction(
            self.smpl_verts, self.smpl_faces, self.rgb_video, self.normal_video,
            scales=self.config.reconstruction_scales
        )
        self.smpl_verts = reconstructed_mesh
        self.log_memory_usage(f"After Multi-Scale Reconstruction")

        # Update NVS
        with torch_gc_context():
            self.update_nvs(final_iter=True)
            self.save_intermediate_results(last_iteration + 1)
            self.log_memory_usage(f"After Final NVS Update")

        # Save final results
        self.save_final_results()
        self.log_memory_usage("Final")
        print(f"【End】{self.args.input_path}")

def main():
    args = parse_args()
    
    config = OmegaConf.load(args.config)
    
    inference = ResumeInference(args,config)
    inference.prepare_reference_image()
    inference.init_modules_pipelines()
    inference.resume_from_intermediate()

    try:
        inference.save_results()
    except Exception as e:
        print(f"Error occurred while saving results: {e}")
        print("You can try to run the save_results method separately later.")

if __name__ == "__main__":
    main()