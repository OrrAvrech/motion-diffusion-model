import json
import random
import difflib
from shutil import copy
from pathlib import Path
from human_feedback.data_utils import get_gpt_sentences, parse_sentence
from human_feedback.motion_utils import subsample_motion
import numpy as np


PREDEFINED_BODY_PARTS = ["lower_body",
                         "upper_body",
                         "knees",
                         "hips"]

def find_closest_body_part(body_part_in: str):
    body_part_out = difflib.get_close_matches(body_part_in, PREDEFINED_BODY_PARTS, n=1)[0]
    return body_part_out


def main():
    fps = 20 # target fps
    current_fps = 30
    subsample_factor = current_fps / fps
    num_frames = int(196*subsample_factor)
    min_seq_len = 2 # in sec
    copy_text = False

    dataset_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX")
    vecs_dir = dataset_dir / "vector_263"
    motion_output_dir = dataset_dir / "human_feedback" / "new_joint_vecs_split"
    motion_output_dir.mkdir(exist_ok=True, parents=True)

    gpt_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/instructions/gpt")
    inst_output_dir = Path("/proj/vondrick2/orr/motion-diffusion-model/dataset/MotionX/human_feedback/instructions/perturb_t2m")
    inst_output_dir.mkdir(exist_ok=True, parents=True)

    text_dir = dataset_dir / "human_feedback" / "seq_texts"
    new_text_dir = dataset_dir / "human_feedback" / "seq_texts_split"
    split_filepath = dataset_dir / "vecs_12.txt"

    filenames = []
    skip_count = 0
    for npy_path in vecs_dir.rglob("*.npy"):
        full_motion = np.load(npy_path)
        diff = len(full_motion) - num_frames
        end_frame = diff if diff > 0 else len(full_motion)
        for i in range(0, end_frame, num_frames):
            curr_end = min(end_frame, i+num_frames)
            curr_motion = full_motion[i: curr_end]
            # MOYO is in 40fps, HumanML3D 20fps, MotionX 30fps
            sub_motion = subsample_motion(curr_motion, subsample_factor)
            if len(sub_motion) <= int(fps*min_seq_len):
                 # skip short motions
                continue

            motion_filename = f"{npy_path.stem}_{i}_{curr_end}.npy"
            relative_path_npy = npy_path.relative_to(vecs_dir).parent

            # write perturbed instructions
            # try:
            #     gpt_path = gpt_dir / relative_path_npy /  f"{npy_path.stem}.json"
            #     sentences = get_gpt_sentences(gpt_path)
            #     sentences = [x for x in sentences if len(x.strip())>0]
            #     sentences = [x.split(f"{i+1}. ")[-1] for i, x in enumerate(sentences)]
            #     instructions = [parse_sentence(x) for x in sentences]

            #     seq_len_sec = len(sub_motion) / fps
            #     data = []
            #     for text, body_part in instructions:
            #         body_part = find_closest_body_part(body_part_in=body_part)
            #         # choose start, end randomly
            #         pert_motion_len = random.uniform(min_seq_len, seq_len_sec)
            #         start = random.uniform(0, seq_len_sec - pert_motion_len)
            #         end = start + pert_motion_len
            #         data.append({"text": text,
            #                     "body_part": body_part,
            #                     "start": start,
            #                     "end": end})
            # except (ValueError, IndexError):
            #     print(f"skip {gpt_path}")
            #     skip_count += 1
            #     continue

            # save split motion
            # save_dir = motion_output_dir / relative_path_npy
            # save_dir.mkdir(exist_ok=True, parents=True)
            # np.save(save_dir / motion_filename, sub_motion)
            
            # save preturb instructions
            # save_dir = inst_output_dir / relative_path_npy
            # save_dir.mkdir(exist_ok=True, parents=True)
            # pert_instructions_filepath = save_dir / f"{Path(motion_filename).stem}.json"
            # with open(pert_instructions_filepath, "w") as fp:
            #     json.dump(data, fp)
            
            filenames.append(str(relative_path_npy / motion_filename))

            if copy_text is True:
                src_text, dst_text = text_dir / relative_path_npy, new_text_dir / relative_path_npy
                dst_text.mkdir(exist_ok=True, parents=True)
                copy(src_text / f"{npy_path.stem}.txt", dst_text / f"{Path(motion_filename).stem}.txt")
    
    with open(split_filepath, "w") as fp:
        [fp.write(f"{x}\n") for x in filenames]

    print(f"Done! Skipped samples count: {skip_count}")


if __name__ == "__main__":
    main()