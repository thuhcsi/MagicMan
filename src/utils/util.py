import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import importlib
import os.path as osp
import shutil
import sys
from pathlib import Path
from rembg import remove
from rembg.session_factory import new_session
import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
import imageio
from torchvision import transforms


def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
    
def pil_list_to_tensor(pil_list):
    transform = transforms.Compose(
        [transforms.Resize((512, 512)), transforms.ToTensor()]
    )
    tensor_list = []
    for pil in pil_list:
        tensor_list.append(transform(pil))
    return torch.stack(tensor_list, dim=0).transpose(0, 1).unsqueeze(0)

def preprocess_image(img_pil, ratio=1.85/2.0, resolution=512):
    img = np.array(img_pil) # H,W,C=3 
    # remove background
    img_rembg = remove(img, post_process_mask=True, session=new_session("u2net")) 
    # resize & center human
    ret, mask = cv2.threshold(img_rembg[..., -1], 0, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = int(max_size / ratio) 
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    padded_image[
        center - h // 2 : center - h // 2 + h,
        center - w // 2 : center - w // 2 + w,
    ] = img_rembg[y : y + h, x : x + w]
    # resize image
    rgba = Image.fromarray(padded_image).resize((resolution, resolution), Image.LANCZOS)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    rgb_pil = Image.fromarray((rgb * 255).astype(np.uint8))
    # mask
    image = (rgba_arr * 255).astype(np.uint8)
    color_mask = image[..., -1]
    image = (rgb * 255).astype(np.uint8)
    invalid_color_mask = color_mask < 255*0.5
    threshold =  np.ones_like(image[:,:,0]) * 250
    invalid_white_mask = (image[:, :, 0] > threshold) & (image[:, :, 1] > threshold) & (image[:, :, 2] > threshold)
    invalid_color_mask_final = invalid_color_mask & invalid_white_mask
    color_mask = (1 - invalid_color_mask_final) > 0
    mask_pil = Image.fromarray((color_mask * 255).astype(np.uint8))
    
    return rgb_pil, mask_pil

def save_image_seq(video, save_dir):
    # input:
    #     video: torch.Tensor[b c f h w]
    #     save_dir: str
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    image_seq = video.squeeze(0).detach().cpu().numpy().transpose(1,2,3,0) # f h w c
    if image_seq.shape[-1]==1:
        image_seq = image_seq[...,0] # f h w for gray image
    num_frames = image_seq.shape[0]
    angle_step = 360 // num_frames
    for i in range(image_seq.shape[0]):
        image = Image.fromarray((image_seq[i]*255).astype(np.uint8))
        angle = i * angle_step
        save_path = os.path.join(save_dir, f"{angle:03d}.png")
        image.save(save_path)

def get_camera(elevation, azimuth):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    
    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 0, 1])
    
    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward) # z
    right = np.cross(-up, forward) 
    right /= np.linalg.norm(right) # x
    new_up = np.cross(forward, right) # y
    new_up /= np.linalg.norm(new_up) # y
    cam2world = np.eye(4)
    cam2world[:3, 0] = right
    cam2world[:3, 1] = new_up
    cam2world[:3, 2] = forward
    cam2world[:3, 3] = camera_pos
    return cam2world


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    import av

    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        writer = imageio.get_writer(path, fps=fps)
        for img in pil_images:
            img = np.array(img)
            writer.append_data(img)
        writer.close()
        # codec = "libx264"
        # container = av.open(path, "w")
        # stream = container.add_stream(codec, rate=fps)

        # stream.width = width
        # stream.height = height

        # for pil_image in pil_images:
        #     # pil_image = Image.fromarray(image_arr).convert("RGB")
        #     av_frame = av.VideoFrame.from_image(pil_image)
        #     container.mux(stream.encode(av_frame))
        # container.mux(stream.encode())
        # container.close()

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps

