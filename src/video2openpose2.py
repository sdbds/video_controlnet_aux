import gradio as gr
from controlnet_aux import (
    OpenposeDetector,
    DWposeDetector,
    AnimalposeDetector,
    DenseposeDetector,
)
import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *
import argparse
import torch
import re

class_map = {
    "openpose": {"class": OpenposeDetector, "download_path": "/path/to/openpose"},
    "dw": {"class": DWposeDetector, "download_path": "/path/to/dw"},
    "animal": {"class": AnimalposeDetector, "download_path": "/path/to/animal"},
    "densepose": {"class": DenseposeDetector, "download_path": "/path/to/densepose"},
}


def main(
    input_path="vid2pose/sample_videos/input_video.mp4",
    output_path="./outputs/",
    preprocesser_model="dwpose",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (
        preprocesser_model
        and preprocesser_model in class_map
        and not preprocesser_model in globals()
        and hasattr(class_map[preprocesser_model]["class"], "from_pretrained")
    ):
        if "torchscript" in preprocesser_model:
            globals()[preprocesser_model] = class_map[preprocesser_model][
                "class"
            ].from_pretrained(
                class_map[preprocesser_model]["download_path"],
                torchscript_device=device,
            )
        else:
            globals()[preprocesser_model] = class_map[preprocesser_model][
                "class"
            ].from_pretrained(class_map[preprocesser_model]["download_path"])
            globals()[preprocesser_model].to(device)

    def regex(string):
        return re.findall(r"\d+", str(string))[-1]

    def get_frames(video_in):
        frames = []
        # resize the video
        clip = VideoFileClip(video_in)

        # check fps
        video_path = os.path.join(output_path, "video_resized.mp4")
        if clip.fps > 30:
            print("vide rate is over 30, resetting to 30")
            clip_resized = clip.resize(height=512)
            clip_resized.write_videofile(video_path, fps=30)
        else:
            print("video rate is OK")
            clip_resized = clip.resize(height=512)
            clip_resized.write_videofile(video_path, fps=clip.fps)

        print("video resized to 512 height")

        # Opens the Video file with CV2
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print("video fps: " + str(fps))
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break
            path = os.path.join(output_path, "raw" + str(i) + ".jpg")
            cv2.imwrite(path, frame)
            frames.append(path)
            i += 1

        cap.release()
        cv2.destroyAllWindows()
        print("broke the video into frames")

        return frames, fps

    def get_openpose_filter(i):
        image = Image.open(i)

        # image = np.array(image)

        if preprocesser_model.__contains__("full"):
            image = globals()[preprocesser_model](
                image, include_hand=True, include_face=True
            )
        elif preprocesser_model.__contains__("hand"):
            image = globals()[preprocesser_model](image, include_hand=True)
        elif preprocesser_model.__contains__("face"):
            image = globals()[preprocesser_model](image, include_face=True)
        else:
            image = globals()[preprocesser_model](image)
        # image = Image.fromarray(image)
        path = os.path.join(output_path, "openpose_frame_" + regex(i) + ".jpeg")
        image.save(path)
        return path

    def create_video(frames, fps, type):
        print("building video result")
        clip = ImageSequenceClip(frames, fps=fps)
        path = os.path.join(output_path, type + "_result.mp4")
        clip.write_videofile(path, fps=fps)

        return path

    def convertG2V(imported_gif):
        clip = VideoFileClip(imported_gif.name)
        path = os.path.join(output_path, "my_gif_video.mp4")
        clip.write_videofile(path)
        return path

    def infer(video_in):
        # 1. break video into frames and get FPS
        break_vid = get_frames(video_in)
        frames_list = break_vid[0]
        fps = break_vid[1]
        # n_frame = int(trim_value*fps)
        n_frame = len(frames_list)

        if n_frame >= len(frames_list):
            print("video is shorter than the cut value")
            n_frame = len(frames_list)

        # 2. prepare frames result arrays
        result_frames = []
        print("set stop frames to: " + str(n_frame))

        for i in frames_list[0 : int(n_frame)]:
            openpose_frame = get_openpose_filter(i)
            result_frames.append(openpose_frame)
            print("frame " + i + "/" + str(n_frame) + ": done;")

        final_vid = create_video(result_frames, fps, "openpose")

        files = [final_vid]

        return final_vid, files

    title = """
<div style="text-align: center; max-width: 500px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
            margin-bottom: 10px;
        "
        >
        <h1 style="font-weight: 600; margin-bottom: 7px;">
            Video to OpenPose
        </h1>
        </div>
    </div>
"""

    with gr.Blocks() as demo:
        with gr.Column():
            gr.HTML(title)
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(
                        source="upload",
                        type="filepath",
                        value=input_path if not input_path.endswith(".gif") else None,
                    )
                    gif_input = gr.File(
                        label="import a GIF instead",
                        file_types=[".gif"],
                        value=input_path if input_path.endswith(".gif") else None,
                    )
                    gif_input.change(
                        fn=convertG2V, inputs=gif_input, outputs=video_input
                    )
                    submit_btn = gr.Button("Submit")

                with gr.Column():
                    video_output = gr.Video()
                    file_output = gr.Files()

        submit_btn.click(
            fn=infer, inputs=[video_input], outputs=[video_output, file_output]
        )

    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="vid2pose/sample_videos/input_video.mp4"
    )
    parser.add_argument("-o", "--output_path", type=str, default="./outputs/")
    parser.add_argument("--preprocesser_model", type=str, default="dwpose")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args.input_path, args.output_path, args.preprocesser_model)
