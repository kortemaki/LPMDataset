from __future__ import unicode_literals

from subprocess import call
import os
import json

import pandas as pd
from tqdm import tqdm

BASE_PATH = os.environ['VIDEO_DIR']
FFMPEG = os.environ['FFMPEG']

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(BASE_PATH, 'raw_video_links.csv'))

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        speaker = row['speaker']
        link = row['link']
        video_id = row['video_id'].replace(".mp4", "")

        save_dir = os.path.join(BASE_PATH, f'figures/{speaker}')
        video_path = f"{save_dir}/{video_id}/{video_id}.mp4"

        video_split = row['Answer.startTimeList'].split("|")[1:]
        src_dir_path = os.path.dirname(video_path)
        tgt_dir_path = src_dir_path
        os.makedirs(tgt_dir_path, exist_ok=True)

        run = f"yt-dlp -i -o {save_dir}/{video_id}/{video_id}.%(ext)s  -v {link}"
        res1 = call(run, shell=True)

        #extract frames
        for i, (timestamp_next, timestamp_prev) in zip(
            video_split or [row['seconds']],
            [0] + video_split[:-1]
        ):
            time_string = str(max(timestamp_next - 1, (timestamp_prev + timestamp_next)/2))
            dir_path = os.path.dirname(video_path)
            output_save = os.path.join(tgt_dir_path, "slide_{}.png".format(str(i).zfill(3)))
            frame_capture = f"{FFMPEG} -ss {time_string} -i {video_path} -frames:v 1 {output_save}"
            subprocess.call(frame_capture, shell = True)

        # delete video
        os.remove(video_path)
