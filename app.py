'''
conda activate zero123
cd stable-diffusion
python gradio_new.py 0
'''
import os, sys
from huggingface_hub import snapshot_download
code_dir = snapshot_download("One-2-3-45/code", token=os.environ['TOKEN'])
# code_dir = "../code"
sys.path.append(code_dir)

elev_est_dir = os.path.join(code_dir, "one2345_elev_est/")
sys.path.append(elev_est_dir)

# sparseneus_dir = os.path.join(code_dir, "SparseNeuS_demo_v1/")
# sys.path.append(sparseneus_dir)

import subprocess
subprocess.run(["sh", os.path.join(elev_est_dir, "install.sh")], cwd=elev_est_dir)
# export TORCH_CUDA_ARCH_LIST="7.0;7.2;8.0;8.6"
# export IABN_FORCE_CUDA=1
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.2;8.0;8.6"
os.environ["IABN_FORCE_CUDA"] = "1"
subprocess.run(["pip", "install", "inplace_abn"]) 

import inspect
import shutil
import torch
import fire
import gradio as gr
import numpy as np
# import plotly.express as px
import plotly.graph_objects as go
# import rich
import sys
from functools import partial

from lovely_numpy import lo
# from omegaconf import OmegaConf
import cv2
from PIL import Image
import trimesh
import tempfile
from zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from sam_utils import sam_init, sam_out, sam_out_nosave
from utils import image_preprocess_nosave, gen_poses
from one2345_elev_est.tools.estimate_wild_imgs import estimate_elev
from rembg import remove

_GPU_INDEX = 0

_TITLE = 'One-2-3-45: Any Single Image to 3D Mesh in 45 Seconds without Per-Shape Optimization'

# This demo allows you to generate novel viewpoints of an object depicted in an input image using a fine-tuned version of Stable Diffusion.
_DESCRIPTION = '''
We reconstruct a 3D textured mesh from a single image by initially predicting multi-view images and then lifting them to 3D.
'''

_USER_GUIDE = "Please upload an image in the top left block (or choose an example above) and click **Run Generation**." 
_BBOX_1 = "Predicting bounding box for the input image..."
_BBOX_2 = "Bounding box adjusted. Continue adjusting or **Run Generation**."
_BBOX_3 = "Bounding box predicted. Adjust it using sliders or **Run Generation**."
_SAM = "Preprocessing the input image... (safety check, SAM segmentation, *etc*.)"
_GEN_1 = "Predicting multi-view images... (may take \~23 seconds) <br> Images will be shown in the bottom right blocks."
_GEN_2 = "Predicting nearby views and generating mesh... (may take \~48 seconds) <br> Mesh will be shown below."
_DONE = "Done! Mesh is shown below. <br> If it is not satisfactory, please select **Retry view** checkboxes for inaccurate views and click **Regenerate selected view(s)** at the bottom."
_REGEN_1 = "Selected view(s) are regenerated. You can click **Regenerate nearby views and mesh**. <br> Alternatively, if the regenerated view(s) are still not satisfactory, you can repeat the previous step (select the view and regenerate)."
_REGEN_2 = "Regeneration done. <br> Mesh is shown below."

class CameraVisualizer:
    def __init__(self, gradio_plot):
        self._gradio_plot = gradio_plot
        self._fig = None
        self._polar = 0.0
        self._azimuth = 0.0
        self._radius = 0.0
        self._raw_image = None
        self._8bit_image = None
        self._image_colorscale = None

    def polar_change(self, value):
        self._polar = value
        # return self.update_figure()

    def azimuth_change(self, value):
        self._azimuth = value
        # return self.update_figure()

    def radius_change(self, value):
        self._radius = value
        # return self.update_figure()

    def encode_image(self, raw_image, elev=90):
        '''
        :param raw_image (H, W, 3) array of uint8 in [0, 255].
        '''
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot

        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        self._raw_image = raw_image
        self._8bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # self._8bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        self._image_colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]
        self._elev = elev
        # return self.update_figure()

    def update_figure(self):
        fig = go.Figure()

        if self._raw_image is not None:
            (H, W, C) = self._raw_image.shape

            x = np.zeros((H, W))
            (y, z) = np.meshgrid(np.linspace(-1.0, 1.0, W), np.linspace(1.0, -1.0, H) * H / W)
            
            angle_deg = self._elev-90
            angle = np.radians(90-self._elev)
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
            # Assuming x, y, z are the original 3D coordinates of the image
            coordinates = np.stack((x, y, z), axis=-1)  # Combine x, y, z into a single array
            # Apply the rotation matrix
            rotated_coordinates = np.matmul(coordinates, rotation_matrix)
            # Extract the new x, y, z coordinates from the rotated coordinates
            x, y, z = rotated_coordinates[..., 0], rotated_coordinates[..., 1], rotated_coordinates[..., 2]


            print('x:', lo(x))
            print('y:', lo(y))
            print('z:', lo(z))

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=self._8bit_image,
                cmin=0,
                cmax=255,
                colorscale=self._image_colorscale,
                showscale=False,
                lighting_diffuse=1.0,
                lighting_ambient=1.0,
                lighting_fresnel=1.0,
                lighting_roughness=1.0,
                lighting_specular=0.3))

            scene_bounds = 3.5
            base_radius = 2.5
            zoom_scale = 1.5  # Note that input radius offset is in [-0.5, 0.5].
            fov_deg = 50.0
            edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

            input_cone = calc_cam_cone_pts_3d(
                angle_deg, 0.0, base_radius, fov_deg)  # (5, 3).
            output_cone = calc_cam_cone_pts_3d(
                self._polar, self._azimuth, base_radius + self._radius * zoom_scale, fov_deg)  # (5, 3).
            output_cones = []
            for i in range(1,4):
                output_cones.append(calc_cam_cone_pts_3d(
                    angle_deg, i*90, base_radius + self._radius * zoom_scale, fov_deg))
            delta_deg = 30 if angle_deg <= -15 else -30
            for i in range(4):
                output_cones.append(calc_cam_cone_pts_3d(
                    angle_deg+delta_deg, 30+i*90, base_radius + self._radius * zoom_scale, fov_deg))

            cones = [(input_cone, 'rgb(174, 54, 75)', 'Input view (Predicted view 1)')]
            for i in range(len(output_cones)):
                cones.append((output_cones[i], 'rgb(32, 77, 125)', f'Predicted view {i+2}'))

            for idx, (cone, clr, legend) in enumerate(cones):

                for (i, edge) in enumerate(edges):
                    (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
                    (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
                    (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
                    fig.add_trace(go.Scatter3d(
                        x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                        line=dict(color=clr, width=3),
                        name=legend, showlegend=(i == 1) and (idx <= 1)))

                # Add label.
                if cone[0, 2] <= base_radius / 2.0:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] - 0.05], showlegend=False,
                        mode='text', text=legend, textposition='bottom center'))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=[cone[0, 0]], y=[cone[0, 1]], z=[cone[0, 2] + 0.05], showlegend=False,
                        mode='text', text=legend, textposition='top center'))

            # look at center of scene
            fig.update_layout(
                # width=640,
                # height=480,
                # height=400,
                height=360,
                autosize=True,
                hovermode=False,
                margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                showlegend=False,
                legend=dict(
                    yanchor='bottom',
                    y=0.01,
                    xanchor='right',
                    x=0.99,
                ),
                scene=dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=1.0),
                    camera=dict(
                        eye=dict(x=base_radius - 1.6, y=0.0, z=0.6),
                        center=dict(x=0.0, y=0.0, z=0.0),
                        up=dict(x=0.0, y=0.0, z=1.0)),
                    xaxis_title='',
                    yaxis_title='',
                    zaxis_title='',
                    xaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    yaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks=''),
                    zaxis=dict(
                        range=[-scene_bounds, scene_bounds],
                        showticklabels=False,
                        showgrid=True,
                        zeroline=False,
                        showbackground=True,
                        showspikes=False,
                        showline=False,
                        ticks='')))

        self._fig = fig
        return fig
    

def stage1_run(models, device, cam_vis, tmp_dir,
               input_im, scale, ddim_steps, rerun_all=[],
               *btn_retrys):
    is_rerun = True if cam_vis is None else False

    stage1_dir = os.path.join(tmp_dir, "stage1_8")
    if not is_rerun:
        os.makedirs(stage1_dir, exist_ok=True)
        output_ims = predict_stage1_gradio(models['turncam'], input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
        stage2_steps = 50 # ddim_steps
        zero123_infer(models['turncam'], tmp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
        elev_output = estimate_elev(tmp_dir)
        gen_poses(tmp_dir, elev_output)
        show_in_im1 = np.asarray(input_im, dtype=np.uint8)
        cam_vis.encode_image(show_in_im1, elev=elev_output)
        new_fig = cam_vis.update_figure()

        flag_lower_cam = elev_output <= 75
        if flag_lower_cam:
            output_ims_2 = predict_stage1_gradio(models['turncam'], input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
        else:
            output_ims_2 = predict_stage1_gradio(models['turncam'], input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
        return (elev_output, new_fig, *output_ims, *output_ims_2)
    else:
        rerun_idx = [i for i in range(len(btn_retrys)) if btn_retrys[i]]
        elev_output = estimate_elev(tmp_dir)
        if elev_output > 75:
            rerun_idx_in = [i if i < 4 else i+4 for i in rerun_idx]
        else:
            rerun_idx_in = rerun_idx
        for idx in rerun_idx_in:
            if idx not in rerun_all:
                rerun_all.append(idx)
        print("rerun_idx", rerun_all)
        output_ims = predict_stage1_gradio(models['turncam'], input_im, save_path=stage1_dir, adjust_set=rerun_idx_in, device=device, ddim_steps=ddim_steps, scale=scale)
        outputs = [gr.update(visible=True)] * 8
        for idx, view_idx in enumerate(rerun_idx):
            outputs[view_idx] = output_ims[idx]
        reset = [gr.update(value=False)] * 8
        return (rerun_all, *reset, *outputs)
    
def stage2_run(models, device, tmp_dir,
               elev, scale, rerun_all=[], stage2_steps=50):
    # print("elev", elev)
    flag_lower_cam = int(elev["label"]) <= 75
    is_rerun = True if rerun_all else False
    if not is_rerun:
        if flag_lower_cam:
            zero123_infer(models['turncam'], tmp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
        else:
            zero123_infer(models['turncam'], tmp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        print("rerun_idx", rerun_all)
        zero123_infer(models['turncam'], tmp_dir, indices=rerun_all, device=device, ddim_steps=stage2_steps, scale=scale)
    
    dataset = tmp_dir
    main_dir_path = os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe())))
    os.chdir(os.path.join(code_dir, 'SparseNeuS_demo_v1/'))

    bash_script = f'CUDA_VISIBLE_DEVICES={_GPU_INDEX} python exp_runner_generic_blender_val.py --specific_dataset_name {dataset} --mode export_mesh --conf confs/one2345_lod0_val_demo.conf  --is_continue'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(tmp_dir, f"meshes_val_bg/lod0/mesh_00340000_gradio_lod0.ply")
    mesh_path = os.path.join(tmp_dir, "mesh.obj")
    # Read the textured mesh from .ply file
    mesh = trimesh.load_mesh(ply_path)
    axis = [1, 0, 0]
    angle = np.radians(90)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(rotation_matrix)
    axis = [0, 0, 1]
    angle = np.radians(180)
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
    mesh.apply_transform(rotation_matrix)
    # flip x
    mesh.vertices[:, 0] = -mesh.vertices[:, 0]
    mesh.faces = np.fliplr(mesh.faces)
    # Export the mesh as .obj file with colors
    mesh.export(mesh_path, file_type='obj', include_color=True)

    if not is_rerun:
        return (mesh_path)
    else:
        return (mesh_path, [], gr.update(visible=False), gr.update(visible=False))

def nsfw_check(models, raw_im, device='cuda'):
    safety_checker_input = models['clip_fe'](raw_im, return_tensors='pt').to(device)
    (_, has_nsfw_concept) = models['nsfw'](
        images=np.ones((1, 3)), clip_input=safety_checker_input.pixel_values)
    print('has_nsfw_concept:', has_nsfw_concept)
    if np.any(has_nsfw_concept):
        print('NSFW content detected.')
        # Define the image size and background color
        image_width = image_height = 256
        background_color = (255, 255, 255)  # White
        # Create a blank image
        image = Image.new("RGB", (image_width, image_height), background_color)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(image)
        text = "Potential NSFW content was detected."
        text_color = (255, 0, 0)
        text_position = (10, 123)  
        draw.text(text_position, text, fill=text_color)
        text = "Please try again with a different image."
        text_position = (10, 133) 
        draw.text(text_position, text, fill=text_color)
        return image
    else:
        print('Safety check passed.')
        return False

def preprocess_run(predictor, models, raw_im, preprocess, *bbox_sliders):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    check_results = nsfw_check(models, raw_im, device=predictor.device)
    if check_results:
        return check_results
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), *bbox_sliders)
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=preprocess, rescale=True)
    return input_256

def calc_cam_cone_pts_3d(polar_deg, azimuth_deg, radius_m, fov_deg):
    '''
    :param polar_deg (float).
    :param azimuth_deg (float).
    :param radius_m (float).
    :param fov_deg (float).
    :return (5, 3) array of float with (x, y, z).
    '''
    polar_rad = np.deg2rad(polar_deg)
    azimuth_rad = np.deg2rad(azimuth_deg)
    fov_rad = np.deg2rad(fov_deg)
    polar_rad = -polar_rad  # NOTE: Inverse of how used_x relates to x.

    # Camera pose center:
    cam_x = radius_m * np.cos(azimuth_rad) * np.cos(polar_rad)
    cam_y = radius_m * np.sin(azimuth_rad) * np.cos(polar_rad)
    cam_z = radius_m * np.sin(polar_rad)

    # Obtain four corners of camera frustum, assuming it is looking at origin.
    # First, obtain camera extrinsics (rotation matrix only):
    camera_R = np.array([[np.cos(azimuth_rad) * np.cos(polar_rad),
                          -np.sin(azimuth_rad),
                          -np.cos(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(azimuth_rad) * np.cos(polar_rad),
                          np.cos(azimuth_rad),
                          -np.sin(azimuth_rad) * np.sin(polar_rad)],
                         [np.sin(polar_rad),
                          0.0,
                          np.cos(polar_rad)]])
    # print('camera_R:', lo(camera_R).v)

    # Multiply by corners in camera space to obtain go to space:
    corn1 = [-1.0, np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn2 = [-1.0, -np.tan(fov_rad / 2.0), np.tan(fov_rad / 2.0)]
    corn3 = [-1.0, -np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn4 = [-1.0, np.tan(fov_rad / 2.0), -np.tan(fov_rad / 2.0)]
    corn1 = np.dot(camera_R, corn1)
    corn2 = np.dot(camera_R, corn2)
    corn3 = np.dot(camera_R, corn3)
    corn4 = np.dot(camera_R, corn4)

    # Now attach as offset to actual 3D camera position:
    corn1 = np.array(corn1) / np.linalg.norm(corn1, ord=2)
    corn_x1 = cam_x + corn1[0]
    corn_y1 = cam_y + corn1[1]
    corn_z1 = cam_z + corn1[2]
    corn2 = np.array(corn2) / np.linalg.norm(corn2, ord=2)
    corn_x2 = cam_x + corn2[0]
    corn_y2 = cam_y + corn2[1]
    corn_z2 = cam_z + corn2[2]
    corn3 = np.array(corn3) / np.linalg.norm(corn3, ord=2)
    corn_x3 = cam_x + corn3[0]
    corn_y3 = cam_y + corn3[1]
    corn_z3 = cam_z + corn3[2]
    corn4 = np.array(corn4) / np.linalg.norm(corn4, ord=2)
    corn_x4 = cam_x + corn4[0]
    corn_y4 = cam_y + corn4[1]
    corn_z4 = cam_z + corn4[2]

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4]

    return np.array([xs, ys, zs]).T

def save_bbox(dir, x_min, y_min, x_max, y_max):
    box = np.array([x_min, y_min, x_max, y_max])
    # save the box to a file
    bbox_path = os.path.join(dir, "bbox.txt")
    np.savetxt(bbox_path, box)

def on_coords_slider(image, x_min, y_min, x_max, y_max, color=(88, 191, 131, 255)):
    """Draw a bounding box annotation for an image."""
    print("on_coords_slider, drawing bbox...")
    image_size = image.size
    if max(image_size) > 180:
        image.thumbnail([180, 180], Image.Resampling.LANCZOS)
        shrink_ratio = max(image.size) / max(image_size)
        x_min = int(x_min * shrink_ratio)
        y_min = int(y_min * shrink_ratio)
        x_max = int(x_max * shrink_ratio)
        y_max = int(y_max * shrink_ratio)
    print("on_coords_slider, image_size:", np.array(image).shape)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, int(max(max(image.shape) / 400*2, 2)))
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA) # image[:, :, ::-1]

def save_img(image):
    image.thumbnail([512, 512], Image.Resampling.LANCZOS)
    width, height = image.size
    image_rem = image.convert('RGBA')
    image_nobg = remove(image_rem, alpha_matting=True)
    arr = np.asarray(image_nobg)[:,:,-1]
    x_nonzero = np.nonzero(arr.sum(axis=0))
    y_nonzero = np.nonzero(arr.sum(axis=1))
    x_min = int(x_nonzero[0].min())
    y_min = int(y_nonzero[0].min())
    x_max = int(x_nonzero[0].max())
    y_max = int(y_nonzero[0].max())
    image_mini = image.copy()
    image_mini.thumbnail([180, 180], Image.Resampling.LANCZOS)
    shrink_ratio = max(image_mini.size) / max(width, height)
    x_min_shrink = int(x_min * shrink_ratio)
    y_min_shrink = int(y_min * shrink_ratio)
    x_max_shrink = int(x_max * shrink_ratio)
    y_max_shrink = int(y_max * shrink_ratio)

    return [on_coords_slider(image_mini, x_min_shrink, y_min_shrink, x_max_shrink, y_max_shrink),
            gr.update(value=x_min, maximum=width),
            gr.update(value=y_min, maximum=height),
            gr.update(value=x_max, maximum=width),
            gr.update(value=y_max, maximum=height)]


def run_demo(
        device_idx=_GPU_INDEX,
        ckpt='zero123-xl.ckpt'):

    print('sys.argv:', sys.argv)
    if len(sys.argv) > 1:
        print('old device_idx:', device_idx)
        device_idx = int(sys.argv[1])
        print('new device_idx:', device_idx)

    device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
    models = init_model(device, os.path.join(code_dir, ckpt))
    # model = models['turncam']
    # sampler = DDIMSampler(model)

    # init sam model
    predictor = sam_init(device_idx)

    with open('instructions_12345.md', 'r') as f:
        article = f.read()

    # NOTE: Examples must match inputs
    example_folder = os.path.join(os.path.dirname(__file__), 'demo_examples')
    example_fns = os.listdir(example_folder)
    example_fns.sort()
    examples_full = [os.path.join(example_folder, x) for x in example_fns if x.endswith('.png')]


    # Compose demo layout & data flow.
    css="#model-3d-out {height: 400px;}"
    with gr.Blocks(title=_TITLE, css=css) as demo:
        gr.Markdown('# ' + _TITLE)
        gr.Markdown(_DESCRIPTION)

        with gr.Row(variant='panel'):
            with gr.Column(scale=0.85):
                image_block = gr.Image(type='pil', image_mode='RGBA', label='Input image', tool=None)
                with gr.Row():
                    bbox_block = gr.Image(type='pil', label="Bounding box", interactive=False).style(height=300)
                    sam_block = gr.Image(type='pil', label="SAM output", interactive=False)
                max_width = max_height = 256
                # with gr.Row():
                #     gr.Markdown('After uploading the image, a bounding box will be generated automatically. If the result is not satisfactory, you can also use the slider below to manually select the object.')
                with gr.Row():
                    x_min_slider = gr.Slider(
                        label="X min",
                        interactive=True,
                        value=0,
                        minimum=0,
                        maximum=max_width,
                        step=1,
                    )
                    y_min_slider = gr.Slider(
                        label="Y min",
                        interactive=True,
                        value=0,
                        minimum=0,
                        maximum=max_height,
                        step=1,
                    )
                with gr.Row():
                    x_max_slider = gr.Slider(
                        label="X max",
                        interactive=True,
                        value=max_width,
                        minimum=0,
                        maximum=max_width,
                        step=1,
                    )
                    y_max_slider = gr.Slider(
                        label="Y max",
                        interactive=True,
                        value=max_height,
                        minimum=0,
                        maximum=max_height,
                        step=1,
                    )
                    bbox_sliders = [x_min_slider, y_min_slider, x_max_slider, y_max_slider]

                
            with gr.Column(scale=1.15):
                gr.Examples(
                    examples=examples_full,  # NOTE: elements must match inputs list!
                    # fn=save_img,
                    fn=lambda x: x,
                    inputs=[image_block],
                    # outputs=[image_block, bbox_block, *bbox_sliders],
                    outputs=[image_block],
                    cache_examples=False,
                    run_on_click=True,
                    label='Examples (click one of the images below to start)',
                )
                preprocess_chk = gr.Checkbox(
                    True, label='Reduce image contrast (mitigate shadows on the backside)')

                with gr.Accordion('Advanced options', open=False):
                    scale_slider = gr.Slider(0, 30, value=3, step=1,
                                             label='Diffusion guidance scale')
                    steps_slider = gr.Slider(5, 200, value=75, step=5,
                                             label='Number of diffusion inference steps')

                # with gr.Row():
                run_btn = gr.Button('Run Generation', variant='primary')
                # guide_title = gr.Markdown(_GUIDE_TITLE, visible=True)
                guide_text = gr.Markdown(_USER_GUIDE, visible=True)

                # with gr.Row():
                    # height does not work [a bug]
                mesh_output = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="One-2-3-45's Textured Mesh", elem_id="model-3d-out") #.style(height=800)
        
        with gr.Row(variant='panel'):
            with gr.Column(scale=0.85):
                with gr.Row():
                    # with gr.Column(scale=8):
                    elev_output = gr.Label(label='Estimated elevation / polar angle of the input image (degree, w.r.t. the Z axis)')
                    # with gr.Column(scale=1):
                    #     theta_output = gr.Image(value="./theta_mini.png", interactive=False, show_label=False).style(width=100)
                vis_output = gr.Plot(
                    label='Camera poses of the input view (red) and predicted views (blue)')
                
            with gr.Column(scale=1.15):
                gr.Markdown('Predicted multi-view images')
                with gr.Row():
                    view_1 = gr.Image(interactive=False, show_label=False).style(height=200)
                    view_2 = gr.Image(interactive=False, show_label=False).style(height=200)
                    view_3 = gr.Image(interactive=False, show_label=False).style(height=200)
                    view_4 = gr.Image(interactive=False, show_label=False).style(height=200)
                with gr.Row():
                    btn_retry_1 = gr.Checkbox(label='Retry view 1')
                    btn_retry_2 = gr.Checkbox(label='Retry view 2')
                    btn_retry_3 = gr.Checkbox(label='Retry view 3')
                    btn_retry_4 = gr.Checkbox(label='Retry view 4')
                with gr.Row():
                    view_5 = gr.Image(interactive=False, show_label=False).style(height=200)
                    view_6 = gr.Image(interactive=False, show_label=False).style(height=200)
                    view_7 = gr.Image(interactive=False, show_label=False).style(height=200)
                    view_8 = gr.Image(interactive=False, show_label=False).style(height=200)
                with gr.Row():
                    btn_retry_5 = gr.Checkbox(label='Retry view 5')
                    btn_retry_6 = gr.Checkbox(label='Retry view 6')
                    btn_retry_7 = gr.Checkbox(label='Retry view 7')
                    btn_retry_8 = gr.Checkbox(label='Retry view 8')
                # regen_btn = gr.Button('Regenerate selected views and mesh', variant='secondary', visible=False)
                with gr.Row():
                    regen_view_btn = gr.Button('1. Regenerate selected view(s)', variant='secondary', visible=False)
                    regen_mesh_btn = gr.Button('2. Regenerate nearby views and mesh', variant='secondary', visible=False)

        update_guide = lambda GUIDE_TEXT: gr.update(value=GUIDE_TEXT)

        views = [view_1, view_2, view_3, view_4, view_5, view_6, view_7, view_8]
        btn_retrys = [btn_retry_1, btn_retry_2, btn_retry_3, btn_retry_4, btn_retry_5, btn_retry_6, btn_retry_7, btn_retry_8]
        
        rerun_idx = gr.State([])
        tmp_dir = gr.State('./demo_tmp/tmp_dir')

        def refresh(tmp_dir):
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            tmp_dir = tempfile.TemporaryDirectory(dir=os.path.join(os.path.dirname(__file__), 'demo_tmp'))
            print("create tmp_dir", tmp_dir.name)
            clear = [gr.update(value=[])] + [None] * 5 + [gr.update(visible=False)] * 2 + [None] * 8 + [gr.update(value=False)] * 8
            return (tmp_dir.name, *clear)
        
        placeholder = gr.Image(visible=False)
        tmp_func = lambda x: False if not x else gr.update(visible=False)
        disable_func = lambda x: gr.update(interactive=False)
        enable_func = lambda x: gr.update(interactive=True)
        image_block.change(fn=refresh,
                           inputs=[tmp_dir],
                           outputs=[tmp_dir, rerun_idx, bbox_block, sam_block, elev_output, vis_output, mesh_output, regen_view_btn, regen_mesh_btn, *views, *btn_retrys]
                           ).success(disable_func, inputs=run_btn, outputs=run_btn
                           ).success(fn=tmp_func, inputs=[image_block], outputs=[placeholder]
                           ).success(fn=partial(update_guide, _BBOX_1), outputs=[guide_text]
                           ).success(fn=save_img,
                                     inputs=[image_block],
                                     outputs=[bbox_block, *bbox_sliders]
                           ).success(fn=partial(update_guide, _BBOX_3), outputs=[guide_text]
                           ).success(enable_func, inputs=run_btn, outputs=run_btn)


        for bbox_slider in bbox_sliders:
            bbox_slider.release(fn=on_coords_slider,
                               inputs=[image_block, *bbox_sliders],
                               outputs=[bbox_block]
                               ).success(fn=partial(update_guide, _BBOX_2), outputs=[guide_text])

        cam_vis = CameraVisualizer(vis_output)

        gr.Markdown(article)

        # Define the function to be called when any of the btn_retry buttons are clicked
        def on_retry_button_click(*btn_retrys):
            any_checked = any([btn_retry for btn_retry in btn_retrys])
            print('any_checked:', any_checked, [btn_retry for btn_retry in btn_retrys])
            # return regen_btn.update(visible=any_checked)
            if any_checked:
                return (gr.update(visible=True), gr.update(visible=True))
            else:
                return (gr.update(), gr.update())
            # return regen_view_btn.update(visible=any_checked), regen_mesh_btn.update(visible=any_checked)
        # make regen_btn visible when any of the btn_retry is checked
        for btn_retry in btn_retrys:
            # Add the event handlers to the btn_retry buttons
            # btn_retry.change(fn=on_retry_button_click, inputs=[*btn_retrys], outputs=regen_btn)
            btn_retry.change(fn=on_retry_button_click, inputs=[*btn_retrys], outputs=[regen_view_btn, regen_mesh_btn])

        

        run_btn.click(fn=partial(update_guide, _SAM), outputs=[guide_text]
                      ).success(fn=partial(preprocess_run, predictor, models), 
                                inputs=[image_block, preprocess_chk, *bbox_sliders], 
                                outputs=[sam_block]
                      ).success(fn=partial(update_guide, _GEN_1), outputs=[guide_text]
                      ).success(fn=partial(stage1_run, models, device, cam_vis),
                                inputs=[tmp_dir, sam_block, scale_slider, steps_slider],
                                outputs=[elev_output, vis_output, *views]
                      ).success(fn=partial(update_guide, _GEN_2), outputs=[guide_text]
                      ).success(fn=partial(stage2_run, models, device),
                                inputs=[tmp_dir, elev_output, scale_slider],
                                outputs=[mesh_output]
                      ).success(fn=partial(update_guide, _DONE), outputs=[guide_text])
    

        regen_view_btn.click(fn=partial(stage1_run, models, device, None),
                             inputs=[tmp_dir, sam_block, scale_slider, steps_slider, rerun_idx, *btn_retrys],
                             outputs=[rerun_idx, *btn_retrys, *views]
                            ).success(fn=partial(update_guide, _REGEN_1), outputs=[guide_text])
        regen_mesh_btn.click(fn=partial(stage2_run, models, device),
                             inputs=[tmp_dir, elev_output, scale_slider, rerun_idx],
                             outputs=[mesh_output, rerun_idx, regen_view_btn, regen_mesh_btn]
                            ).success(fn=partial(update_guide, _REGEN_2), outputs=[guide_text])


    demo.launch(enable_queue=True, share=False, max_threads=80, auth=("admin", "7wQ@>1ga}NNmdLh-N]0*"))


if __name__ == '__main__':

    fire.Fire(run_demo)