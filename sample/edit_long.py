from utils.fixseed import fixseed
import os
import json
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from pathlib import Path
import pandas as pd
from functools import partial
from datetime import datetime


def in_between_fn(input_motions, model_kwargs, start, end):
    model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool, device=input_motions.device)  # True means use gt motion
    model_kwargs['y']['inpainting_mask'][:, :, :,start:end] = False  # do inpainting in those frames
    return model_kwargs


def body_mask_fn(input_motions, model_kwargs, start, end, mask):
    model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool, device=input_motions.device)  # True means use gt motion
    frame_mask = torch.tensor(mask, dtype=torch.bool, device=input_motions.device)
    num_frames_to_generate = end - start
    assert num_frames_to_generate > 0
    model_kwargs['y']['inpainting_mask'][..., start:end] = frame_mask.unsqueeze(0).unsqueeze(
        -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], num_frames_to_generate)
    return model_kwargs


def batch_linear_interpolation(point1, point2, window_size):
    alphas = np.zeros((window_size, *point1.shape))
    for d in range(3):
        alphas[..., d] = np.linspace(point1[:, d], point2[:, d], window_size)
    return alphas


def sample_lerp(sample, start_frame, end_frame):
    start, end = start_frame, end_frame
    window = (end - start) // 2
    interp_start = max(start - window, 0)
    p1_before = sample[..., interp_start].squeeze()
    p2_before = sample[..., interp_start + window].squeeze()
    vec_before = batch_linear_interpolation(p1_before, p2_before, window)

    p1_after = sample[..., end-1].squeeze()
    interp_end = min(end-1 + window, sample.shape[-1]-1)
    p2_after = sample[..., interp_end].squeeze()
    vec_after = batch_linear_interpolation(p1_after, p2_after, interp_end - (end-1))

    sample[..., interp_start : interp_start + window] = torch.Tensor(vec_before).unsqueeze(0).permute(0, 2, 3, 1)
    sample[..., end-1 : interp_end] = torch.Tensor(vec_after).unsqueeze(0).permute(0, 2, 3, 1)
    return sample


def main():
    INPAINTING_DICT = {
        "in_between": in_between_fn,
        # lower_body, upper_body defined the opposite in the original code.
        # keeping this for compatability
        "lower_body": partial(body_mask_fn, mask=humanml_utils.HML_LOWER_BACK_UNMASK),
        "upper_body": partial(body_mask_fn, mask=humanml_utils.HML_UPPER_BACK_UNMASK),
        # disable lower_body and upper_body at the moment
        "lower_back": partial(body_mask_fn, mask=humanml_utils.HML_LOWER_BACK_UNMASK),
        "upper_back": partial(body_mask_fn, mask=humanml_utils.HML_UPPER_BACK_UNMASK),
        "knees": partial(body_mask_fn, mask=humanml_utils.HML_KNEE_UNMASK),
        "hips": partial(body_mask_fn, mask=humanml_utils.HML_HIP_UNMASK),
    }

    args = edit_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if args.dataset in ['kit', 'humanml', "humanfeedback"]:
        max_frames = 196
    elif args.dataset == "moyo":
        max_frames = 196
    else:
        max_frames = 60

    if args.dataset == "kit":
        fps = 12.5
    elif args.dataset == "moyo":
        fps = 20
    else:
        fps = 20
    dist_util.setup_dist(args.device)
    run_time = datetime.now().strftime("%Y%m%d_%H%M")

    viz_dir = Path(args.output_dir) / "viz" / "vecs_12"
    viz_dir.mkdir(exist_ok=True, parents=True)
    npy_dir = Path(args.output_dir) / "human_feedback" / "vecs_12"
    npy_dir.mkdir(exist_ok=True, parents=True)

    # if out_path == '':
    #     out_path = os.path.join(os.path.dirname(args.model_path),
    #                             'edit_{}_{}_{}_seed{}_{}'.format(name, niter, args.edit_mode, args.seed, run_time))
    #     if args.text_condition != '':
    #         out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')
    out_path = viz_dir

    print('Loading dataset...')
    # assert args.num_samples <= args.batch_size, \
        # f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    # args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    args.batch_size = 1
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split=args.split_file,
                              hml_mode='train')  # in train mode, you get both text and motion.

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    iterator = iter(data)

    for _ in range(args.num_samples):
        try:
            input_motions, model_kwargs, filename = next(iterator)
        except StopIteration:
            break

        input_motions = input_motions.to(dist_util.dev())
        filename = filename[0]

        # add inpainting mask according to args
        assert max_frames == input_motions.shape[-1]
        gt_frames_per_sample = {}
        model_kwargs['y']['inpainted_motion'] = input_motions

        all_motions, all_text, all_edit_modes = [[] for _ in range(3)]
        sample_timeline_path = Path(args.timeline_path) / f"{filename}.json"
        with open(sample_timeline_path, "r") as fp:
            timeline = json.load(fp)

        num_repetitions = len(timeline)
        for i, rep in enumerate(timeline):
            text = rep["text"]
            start_time = rep["start"]
            end_time = rep["end"]
            start = int(fps * start_time)
            end = int(fps * end_time)
            inpainting = rep["body_part"].lower()

            texts = [text]
            edit_modes = [inpainting]
            model_kwargs['y']['text'] = texts
            if text == '':
                args.guidance_param = 0.  # Force unconditioned generation

            inpainting_fn = INPAINTING_DICT[inpainting]
            model_kwargs = inpainting_fn(input_motions=input_motions, 
                                        model_kwargs=model_kwargs, 
                                        start=start, 
                                        end=end)

            # add CFG scale to batch
            model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop

            sample = sample_fn(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

                sample = sample_lerp(sample, start, end)

            all_text += model_kwargs['y']['text']
            all_edit_modes += edit_modes
            all_motions.append(sample.cpu().numpy())

        print(f"created {len(all_motions) * args.batch_size} samples")

        print(f"saving visualizations to [{str(out_path)}]...")
        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
            input_motions = recover_from_ric(input_motions, n_joints)
            input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()

        # Visualizations
        caption = 'Input Motion'
        length = int(model_kwargs['y']['lengths'].cpu().numpy())
        motion = input_motions.squeeze().transpose(2, 0, 1)[:length]
        save_file = f"input_motion_{filename}.mp4"
        animation_save_path = str(out_path / save_file)
        rep_files = [animation_save_path]
        print(f'[({filename}) "{caption}" | -> {save_file}]')
        plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                    dataset=args.dataset, fps=fps, vis_mode='gt')
        
        # Save generated motions
        sample_savedir = npy_dir / filename
        sample_savedir.mkdir(exist_ok=True, parents=True)

        for rep_i in range(num_repetitions):
            caption = all_text[rep_i]
            edit_mode = all_edit_modes[rep_i]
            if caption == '':
                caption = 'Edit [{}] unconditioned'.format(edit_mode)
            else:
                caption = 'Edit [{}]: {}'.format(edit_mode, caption)
            motion = all_motions[rep_i].squeeze().transpose(2, 0, 1)[:length]
            save_file = f"sample{filename}_rep{rep_i}.mp4"
            animation_save_path = str(out_path / save_file)
            rep_files.append(animation_save_path)
            print(f'[({filename}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
                        dataset=args.dataset, fps=fps, vis_mode=args.edit_mode)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

            # save each rep
            np.save(sample_savedir / f"{filename}_{rep_i}.npy", motion)

        all_rep_save_file = out_path / f"{filename}.mp4"
        ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
        hstack_args = f' -filter_complex hstack=inputs={num_repetitions+1}'
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
        os.system(ffmpeg_rep_cmd)
        print(f'[({filename}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    print(f'[Done] Results are at [{out_path.resolve()}]')
        
        
if __name__ == "__main__":
    main()
