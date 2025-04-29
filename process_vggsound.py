#!/usr/bin/env python
import sys
import os

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])

import argparse
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from decord import VideoReader, cpu
import torchvision.transforms as transforms

from fairscale.nn.model_parallel import initialize as fs_init
import torch.distributed as dist

from util.misc import setup_for_distributed
from util.misc import default_tensor_type
from model.meta import MetaModel
from data.conversation_lib import conv_templates, SeparatorStyle
from data.finetune_dataset import make_audio_features
from data import video_utils

# Read audio classes from CSV
CLASSES = pd.read_csv("../../data/audio_classes.csv")["display_name"].tolist()

# Image transformation from the original script
T_random_resized_crop = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            size=(224, 224),
            scale=(0.9, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=3,
            antialias=None,
        ),  # 3 is bicubic
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def load_model(args):
    """
    Load the OneLLM model
    """
    # Initialize distributed setup
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    world_size = len(args.gpu_ids)
    rank = 0  # Just use the first GPU for inference
    gpu_id = args.gpu_ids[rank]

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )
    fs_init.initialize_model_parallel(world_size)
    torch.cuda.set_device(gpu_id)

    torch.manual_seed(1)
    np.random.seed(1)

    # Set the print behavior
    setup_for_distributed(rank == 0)

    # Load model with appropriate data type
    target_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel(
            args.llama_type, args.llama_config, tokenizer_path=args.tokenizer_path
        )

    print("Loading pretrained weights...")
    checkpoint = torch.load(args.pretrained_path, map_location="cpu")
    msg = model.load_state_dict(checkpoint, strict=False)
    print("Load result:\n", msg)

    model.cuda()
    model.eval()
    print(f"Model loaded successfully")

    return model, target_dtype


def load_audio(audio_path):
    """Load and process audio for the model"""
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]  # [1, 128, 1024]
    return fbank


def load_video(video_path):
    """Load and process video for the model"""
    video_feats = video_utils.load_and_transform_video_data(
        video_path, video_path, clip_duration=1, clips_per_video=5
    )
    return video_feats[:, :, 0]


def get_video_list(csv_path):
    """
    Reads video IDs from a CSV file.
    Assumes CSV with two columns: video_id and label. If the video_id does not
    end with '.mp4', it appends '.mp4'.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
        return []

    df = pd.read_csv(csv_path, names=["video_id", "label"], header=None)
    video_ids = df["video_id"].tolist()
    video_ids = [vid if vid.endswith(".mp4") else vid + ".mp4" for vid in video_ids]
    return video_ids


def write_predictions_csv(predictions, responses, output_csv):
    """
    Writes the predictions dictionary to a CSV file.
    The CSV will have columns: video_id, suggestions, and response.
    """
    df_table = {
        i: {
            "video_id": vid,
            "suggestions": list(predictions[vid]),
            "response": responses[vid],
        }
        for i, vid in enumerate(predictions.keys())
    }
    df = pd.DataFrame.from_dict(df_table, orient="index")
    df.to_csv(output_csv, index=False)
    tqdm.write(f"Predictions CSV saved to {output_csv}")


@torch.inference_mode()
def process_video(
    model,
    target_dtype,
    dataset_path,
    video_id,
    max_gen_len=1024,
    temperature=0.2,
    top_p=0.75,
    modality="av",
    prompt="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
    prompt_mode="single",
):
    """
    Process a single video and detect classes using OneLLM model.
    Returns a list of detected classes and the model's response.
    """
    # Set up paths
    video_path = os.path.join(dataset_path, "video", video_id)
    audio_path = None

    if modality == "av" or modality == "audio":
        audio_path = os.path.join(
            dataset_path, "audio", video_id.replace(".mp4", ".wav")
        )

    # Load inputs based on modality
    inputs = None

    if modality == "v":
        try:
            inputs = load_video(video_path)
            inputs = inputs[None].cuda().to(target_dtype)
        except Exception as e:
            tqdm.write(f"Error loading video {video_id}: {e}")
            return [], f"Error: {str(e)}"

    elif modality == "a":
        try:
            inputs = load_audio(audio_path)
            inputs = inputs.cuda().to(target_dtype)
        except Exception as e:
            tqdm.write(f"Error loading audio {audio_path}: {e}")
            return [], f"Error: {str(e)}"

    elif modality == "av":
        raise NotImplementedError("Audio-visual modality not implemented")
        
    # Create conversation template
    conv = conv_templates["v1"].copy()
    modality_map = {
        "av": "video",
        "a": "audio",
        "v": "video",
    }
    detected = []
    response = ""

    # Process with either single prompt for all classes or individual prompts per class
    if prompt_mode == "single":
        # Format prompt with all classes
        formatted_prompt = prompt.format(cl=", ".join(CLASSES))
        conv.append_message(conv.roles[0], formatted_prompt)
        conv.append_message(conv.roles[1], "")

        # Generate response
        responses = []
        for stream_response in model.stream_generate(
            conv.get_prompt(),
            inputs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            modal=modality_map[modality],
        ):
            # Handle end of content
            conv_sep = (
                conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
            )
            end_pos = stream_response["text"].find(conv_sep)
            if end_pos != -1:
                stream_response["text"] = stream_response["text"][:end_pos].rstrip()
                stream_response["end_of_content"] = True

            responses.append(stream_response["text"])

            if stream_response.get("end_of_content", False):
                break

        response = responses[-1] if responses else ""

        # Check for detected classes
        for cl in CLASSES:
            if cl.lower() in response.lower():
                detected.append(cl)

    elif prompt_mode == "multi":
        all_responses = []

        for cl in tqdm(CLASSES, desc="Processing classes", leave=False):
            # Reset conversation for each class
            conv = conv_templates["v1"].copy()

            # Format prompt for this class
            formatted_prompt = prompt.format(cl=cl)
            conv.append_message(conv.roles[0], formatted_prompt)
            conv.append_message(conv.roles[1], "")

            # Generate response
            class_responses = []
            for stream_response in model.stream_generate(
                conv.get_prompt(),
                inputs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                modal=modality_map[modality],
            ):
                # Handle end of content
                conv_sep = (
                    conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
                )
                end_pos = stream_response["text"].find(conv_sep)
                if end_pos != -1:
                    stream_response["text"] = stream_response["text"][:end_pos].rstrip()
                    stream_response["end_of_content"] = True

                class_responses.append(stream_response["text"])

                if stream_response.get("end_of_content", False):
                    break

            class_response = class_responses[-1] if class_responses else ""

            if "yes" in class_response.lower():
                detected.append(cl)

            all_responses.append(f"{cl}: {class_response}")

        response = ",".join(all_responses)

    else:
        raise ValueError(
            f"Invalid prompt mode: {prompt_mode}. Supported modes: 'single', 'multi'"
        )

    # Return the unique set of detected classes and response
    return list(set(detected)), response


def main():
    parser = argparse.ArgumentParser(
        description="Process videos and generate predictions using OneLLM model"
    )

    # Model configuration
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[0],
        help="GPU IDs to use for processing",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="config/llama2/tokenizer.model",
        help="Path to tokenizer model file",
    )
    parser.add_argument(
        "--llama_type", type=str, default="onellm", help="LLaMA model type"
    )
    parser.add_argument(
        "--llama_config",
        type=str,
        default="config/llama2/7B.json",
        help="Path to LLaMA config file",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="weights/consolidated.00-of-01.pth",
        help="Path to pretrained model weights",
    )
    parser.add_argument(
        "--master_port", type=int, default=23862, help="Port for PyTorch distributed"
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="127.0.0.1",
        help="Address for PyTorch distributed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        default="fp16",
        help="Data type for model weights and inference",
    )

    # Dataset and output configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/tmp/vggsound",
        help="Path to VGGSound dataset directory",
    )
    parser.add_argument(
        "--video_csv",
        type=str,
        default="../../data/train.csv",
        help="CSV file with video IDs to process",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="../../data/onellm_predictions.csv",
        help="Output CSV file for predictions",
    )

    # Generation parameters
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.75, help="Top-p for generation"
    )
    parser.add_argument(
        "--max_gen_len", type=int, default=1024, help="Maximum generation length"
    )

    # Processing configuration
    parser.add_argument("--page", type=int, default=1, help="Page number to process")
    parser.add_argument(
        "--per_page",
        type=int,
        default=1000,
        help="Number of videos to process per page",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="av",
        choices=["av", "a", "v"],
        help="Modality to use: 'av' for audio-visual, 'audio' for audio-only, 'video' for video-only",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Do you hear or see '{cl}' class in this video? Answer only with yes or no.",
        help="Prompt template for class detection",
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="single",
        choices=["single", "multi"],
        help="Prompt mode: 'single' for one prompt with all classes, 'multi' for individual prompts per class",
    )

    args = parser.parse_args()

    # Load model
    print("Loading OneLLM model...")
    model, target_dtype = load_model(args)

    # Get list of videos to process
    video_list = get_video_list(args.video_csv)
    if not video_list:
        print("No videos found to process.")
        return

    # Process only a subset (page) of videos
    page_videos = video_list[
        args.page * args.per_page : (args.page + 1) * args.per_page
    ]

    # Update output CSV filename to include page and prompt mode
    args.output_csv = args.output_csv.replace(
        ".csv", f"_{args.prompt_mode}_page_{args.page}.csv"
    )

    predictions = {}
    responses = {}

    # Process each video
    for video_id in tqdm(page_videos, desc="Processing Videos"):
        try:
            detected_classes, response = process_video(
                model=model,
                target_dtype=target_dtype,
                dataset_path=args.dataset_path,
                video_id=video_id,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
                modality=args.modality,
                prompt=args.prompt,
                prompt_mode=args.prompt_mode,
            )

            predictions[video_id] = detected_classes
            responses[video_id] = response

            # Write predictions to CSV periodically
            write_predictions_csv(predictions, responses, args.output_csv)

        except Exception as e:
            tqdm.write(f"Error processing video {video_id}: {e}")
            continue


if __name__ == "__main__":
    main()
