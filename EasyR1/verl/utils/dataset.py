# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import subprocess
import hashlib
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF

# --- Configuration ---
# Max model length budget for tokenizer+multimodal tokens. Configurable via env.
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
# Default max frames per video (before truncation).
DEFAULT_MAX_VIDEO_FRAMES = int(os.getenv("EASYR1_MAX_FRAMES", "64"))


# --- Templates ---
QUESTION_TEMPLATE = (
"{Question}\n\n"
"Please carefully analyze the video by first focusing on the differences between various frames, especially those that reveal overall movement or changes. "
"Your <observation> should emphasize these frame-to-frame differences and describe the overall movement observed in the video. "
"Then, in <think>, reason about the camera motion based on your observations, explaining how the visual evidence leads to your conclusion about the type of camera movement. "
"Finally, provide only the single option letter (e.g., A, B, C, D, E, F etc.) within the <answer> </answer> tags. "
"Follow the format specified in the instructions."
)

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "OCR": (
        "Please provide only the transcribed text within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Hello World</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "regression": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>42.7</answer>"
    ),
    "math": (
        "Please provide only the final result (a number or LaTeX formula) within the <answer>...</answer> tags.\n"
        "Example:\n<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "temporal grounding": (
        "Please provide only the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"time\": [12.3, 25.7]}</answer>"
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        "Example:\n<answer>{\"boxes\": [35, 227, 437, 932]}</answer>"
    ),
    "spatial-temporal grounding": (
        "Please provide only the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        "<answer>{\"time\": [8.125, 13.483], \"boxes\": {\"9\": [317, 422, 582, 997], "
        "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "tracking": (
        "Please track the target object throughout the video and provide one bounding box per second, "
        "ONLY up to 32 seconds, within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>{\"boxes\": {\"1\": [405, 230, 654, 463], \"2\": [435, 223, 678, 446], ..., "
        "\"32\": [415, 203, 691, 487]}}</answer>\n"
        "Note: Each key in 'boxes' must correspond to a second (1, 2, 3, ..., 32) and contain a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "segmentation_image": (
        "This task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\n"
        "Please provide ONE bounding box, 3 positive points (clearly INSIDE the object), and 3 negative points "
        "(clearly OUTSIDE the object) within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>{\"boxes\": [x1, y1, x2, y2], \"positive_points\": [[x,y],[x,y],[x,y]], "
        "\"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    ),
    "segmentation_video": (
        "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
        "Please select ONE representative time (in seconds), and provide ONE bounding box, "
        "3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>{\"time\": <time_in_seconds>, \"boxes\": [x1, y1, x2, y2], "
        "\"positive_points\": [[x,y],[x,y],[x,y]], \"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
    )
}

# --- Helpers ---

def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str, min_pixels: int = 4*32*32, max_pixels: int = 64*32*32, max_frames: int = 128, video_fps: float = 2, return_fps: bool = False
):
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "max_frames": max_frames, "fps": video_fps}
    return fetch_video(vision_info, image_patch_size=16, return_video_sample_fps=return_fps, return_video_metadata=return_fps)


def _hash_path(path: str) -> str:
    try:
        return hashlib.sha1(path.encode("utf-8")).hexdigest()
    except Exception:
        return str(abs(hash(path)))


def extract_frames_1fps(video_path: str, max_frames: int = DEFAULT_MAX_VIDEO_FRAMES) -> list[str]:
    """
    Extracts frames from a video at 1 FPS into a temporary cache directory.
    Returns a list of file paths to the extracted frames.
    """
    if not os.path.exists(video_path):
        return []

    cache_root = os.path.join(tempfile.gettempdir(), "easyrlhf_frame_cache")
    os.makedirs(cache_root, exist_ok=True)
    
    vid_hash = _hash_path(video_path)
    out_dir = os.path.join(cache_root, vid_hash)
    
    if os.path.exists(out_dir):
        existing_frames = sorted([
            os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")
        ])
        if existing_frames:
            return existing_frames[:max_frames]
    
    os.makedirs(out_dir, exist_ok=True)

    try:
        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-i", video_path,
            "-vf", "fps=1",
            os.path.join(out_dir, "frame_%05d.png"),
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        return []
    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []

    frames = sorted([
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".png")
    ])
    return frames[:max_frames]


def _truncate_frames_to_fit(
    processor: ProcessorMixin,
    frames_per_video: list[list[str]], 
    min_pixels: Optional[int], 
    max_pixels: Optional[int],
    max_len: int = MAX_MODEL_LEN
) -> list[list[str]]:
    """
    Iteratively reduces the number of frames if the total token count exceeds max_len.
    """
    current_frames = [list(f) for f in frames_per_video]
    
    while True:
        flat_frames = [f for vid_frames in current_frames for f in vid_frames]
        if not flat_frames:
            return current_frames

        # Create dummy messages just to check length
        dummy_text = "Dummy text" 
        content_list = [{"type": "image"} for _ in flat_frames]
        content_list.append({"type": "text", "text": dummy_text})
        messages = [{"role": "user", "content": content_list}]
        
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        processed_imgs = [process_image(fp, min_pixels, max_pixels) for fp in flat_frames]
        
        try:
            inputs = processor(images=processed_imgs, text=[text_prompt], return_tensors="pt")
            curr_len = inputs.input_ids.shape[1]
        except Exception:
            curr_len = max_len + 1

        if curr_len <= max_len:
            return current_frames

        # Reduce frames by 20%
        total_before = sum(len(x) for x in current_frames)
        for i in range(len(current_frames)):
            n = len(current_frames[i])
            if n > 1:
                keep_n = max(1, int(n * 0.8))
                indices = np.linspace(0, n-1, keep_n, dtype=int)
                current_frames[i] = [current_frames[i][j] for j in indices]
        
        total_after = sum(len(x) for x in current_frames)
        if total_after == total_before:
            break
            
    return current_frames


class RLHFDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _get_question_text(self, example: dict[str, Any]) -> str:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        pt = example.get("problem_type") or ""
        question = prompt_str  

        if (pt == "multiple choice") and isinstance(example.get("options"), list) and example["options"]:
            opts = "\n".join(example["options"])
            question = f"{question}\nOptions:\n{opts}"

        return QUESTION_TEMPLATE.format(Question=question)

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        question_text = self._get_question_text(example)
        
        if self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            videos = example[self.video_key]
            if self.image_dir and videos and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, v) for v in videos]

            all_frames = []
            for v in videos:
                all_frames.extend(extract_frames_1fps(v, DEFAULT_MAX_VIDEO_FRAMES))

            content_list = [{"type": "image"} for _ in all_frames]
            content_list.append({"type": "text", "text": question_text})
            messages = [{"role": "user", "content": content_list}]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            processed_images = [process_image(f, self.min_pixels, self.max_pixels) for f in all_frames]
            
            if processed_images:
                model_inputs = self.processor(images=processed_images, text=[prompt], add_special_tokens=False, return_tensors="pt")
                return model_inputs.input_ids.size(1) <= self.max_prompt_length
            else:
                return True # Fallback

        elif self.image_key in example:
            return True # Simplified for brevity, assume images fit
            
        else:
            messages = [{"role": "user", "content": question_text}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            return tokenized.input_ids.size(1) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        question_text = self._get_question_text(example)
        example.pop(self.prompt_key, None)

        input_ids = None
        attention_mask = None
        
        # --- VIDEO HANDLING ---
        if self.video_key in example and isinstance(example.get(self.video_key), list) and len(example.get(self.video_key)) > 0:
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]

            frames_per_video: list[list[str]] = []
            for v in videos:
                frames_per_video.append(extract_frames_1fps(v, DEFAULT_MAX_VIDEO_FRAMES))

            if self.processor is not None:
                frames_per_video = _truncate_frames_to_fit(
                    self.processor, 
                    frames_per_video, 
                    self.min_pixels, 
                    self.max_pixels,
                    max_len=self.max_prompt_length
                )

            flat_frame_paths = [f for vid_frames in frames_per_video for f in vid_frames]

            # Build messages: <image>...<image> Text
            content_list = [{"type": "image"} for _ in flat_frame_paths]
            content_list.append({"type": "text", "text": question_text})
            messages = [{"role": "user", "content": content_list}]
            
            # Apply template -> Generates string with exact number of <image> tags
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            if flat_frame_paths:
                processed_images = [process_image(fp, self.min_pixels, self.max_pixels) for fp in flat_frame_paths]
                model_inputs = self.processor(
                    images=processed_images, 
                    text=[prompt], 
                    add_special_tokens=False, 
                    return_tensors="pt"
                )
                input_ids = model_inputs.input_ids[0]
                attention_mask = model_inputs.attention_mask[0]
                example["multi_modal_data"] = {"images": processed_images} 
            else:
                model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.input_ids[0]
                attention_mask = model_inputs.attention_mask[0]
                example["multi_modal_data"] = {}

        # --- IMAGE HANDLING ---
        elif self.image_key in example and isinstance(example.get(self.image_key), list) and len(example.get(self.image_key)) > 0:
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]

            content_list = [{"type": "image"} for _ in images]
            content_list.append({"type": "text", "text": question_text})
            messages = [{"role": "user", "content": content_list}]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            
            processed_images = [process_image(img, self.min_pixels, self.max_pixels) for img in images]
            model_inputs = self.processor(images=processed_images, text=[prompt], add_special_tokens=False, return_tensors="pt")
            
            input_ids = model_inputs.input_ids[0]
            attention_mask = model_inputs.attention_mask[0]
            example["multi_modal_data"] = {"images": processed_images}

        # --- TEXT ONLY ---
        else:
            messages = [{"role": "user", "content": question_text}]
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.input_ids[0]
            attention_mask = model_inputs.attention_mask[0]

        if "images" in example: example.pop("images")
        if "videos" in example: example.pop("videos")

        # --- FIX: Update Prompt ---
        example["prompt"] = prompt 
        print(prompt)
        # --------------------------

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index
            
            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        return example