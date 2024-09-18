import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class Config:
    SEG_CHECKPOINTS_DIR = '/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/seg/checkpoints/sapiens_1b/'  
    CHECKPOINTS_DIR = '/media/oem/12TB/sapiens/pretrain/sapiens_lite_host/torchscript/normal/checkpoints/sapiens_1b/'

    CHECKPOINTS = {
        "0.3b": "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2",
        "0.6b": "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2",
        "1b": "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2",
        "2b": "sapiens_2b_normal_render_people_epoch_70_torchscript.pt2",
    }
    SEG_CHECKPOINTS = {
        "fg-bg-1b": "sapiens_1b_seg_foreground_epoch_8_torchscript.pt2",
        "no-bg-removal": None,
        "part-seg-1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
    }

class ModelManager:
    @staticmethod
    def load_model(checkpoint_name: str, is_seg_model: bool = False):
        if checkpoint_name is None:
            return None
        if is_seg_model:
            checkpoint_path = os.path.join(Config.SEG_CHECKPOINTS_DIR, checkpoint_name)
        else:
            checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, checkpoint_name)
        model = torch.jit.load(checkpoint_path)
        model.eval()
        model.to("cuda")
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        return torch.nn.functional.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)

class ImageProcessor:
    def __init__(self):
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
        ])

    def process_image(self, image_path: str, normal_model_name: str, seg_model_name: str):
        image = Image.open(image_path).convert("RGB")
        normal_model = ModelManager.load_model(Config.CHECKPOINTS[normal_model_name])
        input_tensor = self.transform_fn(image).unsqueeze(0).to("cuda")

        normal_output = ModelManager.run_model(normal_model, input_tensor, image.height, image.width)
        normal_map = normal_output.squeeze().cpu().numpy().transpose(1, 2, 0)

        if seg_model_name != "no-bg-removal":
            seg_model = ModelManager.load_model(Config.SEG_CHECKPOINTS[seg_model_name], is_seg_model=True)
            seg_output = ModelManager.run_model(seg_model, input_tensor, image.height, image.width)
            seg_mask = (seg_output.argmax(dim=1) > 0).float().cpu().numpy()[0]
            normal_map[seg_mask == 0] = np.nan

        return normal_map

    @staticmethod
    def visualize_normal_map(normal_map):
        normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)
        normal_map_vis = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
        return normal_map_vis

def main():
    parser = argparse.ArgumentParser(description="Normal Estimation CLI")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to save the output normal map visualization")
    parser.add_argument("output_npy", help="Path to save the output normal map as .npy file")
    parser.add_argument("--normal_model", choices=list(Config.CHECKPOINTS.keys()), default="1b", help="Normal model size")
    parser.add_argument("--seg_model", choices=list(Config.SEG_CHECKPOINTS.keys()), default="fg-bg-1b", help="Segmentation model for background removal")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    processor = ImageProcessor()
    normal_map = processor.process_image(args.input_image, args.normal_model, args.seg_model)

    # Save the normal map as .npy file
    np.save(args.output_npy, normal_map)

    # Visualize and save the normal map as an image
    normal_map_vis = processor.visualize_normal_map(normal_map)
    Image.fromarray(normal_map_vis).save(args.output_image)

    print(f"Normal map visualization saved to: {args.output_image}")
    print(f"Normal map data saved to: {args.output_npy}")

if __name__ == "__main__":
    main()

    # python normalmap_vis.py '/home/oem/Desktop/OLD/image_1.png' test.png test.pny

