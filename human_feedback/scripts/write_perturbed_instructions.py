import re
import json
import random
from pathlib import Path
import numpy as np
from human_feedback.data_utils import get_gpt_sentences


def parse_sentence(sentence):
    text, tmp_body_part = re.split(r'[(\[]', sentence)
    body_part = re.split(r'[)\]]', tmp_body_part)[0]
    text = text.rstrip()
    return text, body_part


def main():
    fps = 20 # target fps
    current_fps = 30
    subsample_factor = current_fps / fps
    num_frames = int(196*subsample_factor)
    min_seq_len = 3 # in sec

    dataset_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/new_joint_vecs")
    gpt_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/gpt")
    output_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MOYO/human_feedback/perturb_t2m")
    output_dir.mkdir(exist_ok=True, parents=True)

    for npy_path in dataset_dir.rglob("*.npy"):
        npy_stem = npy_path.stem
        full_motion = np.load(npy_path)
        for i in range(0, len(full_motion) - num_frames, num_frames):
            curr_motion = full_motion[i: i+num_frames]
            # MOYO is in 40fps, HumanML3D 20fps, MotionX 30fps
            curr_motion = curr_motion[::subsample_factor]
            if len(curr_motion) <= int(fps*min_seq_len):
                # skip short motions
                continue
            
            gpt_path = gpt_dir / f"{npy_stem}.json"
            sentences = get_gpt_sentences(gpt_path)
            sentences = [x for x in sentences if len(x.strip())>0]
            sentences = [x.split(f"{i+1}. ")[-1] for i, x in enumerate(sentences)]
            instructions = [parse_sentence(x) for x in sentences]

            seq_len_sec = len(curr_motion) / fps
            data = []
            for text, body_part in instructions:
                # choose start, end randomly
                pert_motion_len = random.uniform(min_seq_len, seq_len_sec)
                start = random.uniform(0, seq_len_sec - pert_motion_len)
                end = start + pert_motion_len
                data.append({"text": text,
                             "body_part": body_part,
                             "start": start,
                             "end": end})
            
            motion_filename = f"{npy_stem}_{i}_{i+num_frames}"
            pert_instructions_filepath = output_dir / f"{motion_filename}.json"
            with open(pert_instructions_filepath, "w") as fp:
                json.dump(data, fp)


if __name__ == "__main__":
    main()