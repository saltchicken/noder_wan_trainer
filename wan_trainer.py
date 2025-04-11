from typing import Union, List
import asyncio
import os
import toml

import base64
from datetime import datetime
from io import BytesIO
from PIL import Image


class RealWanTrainer(Node):
    async def run(
        self,
        captioned_images: Union[CaptionedImage, List[CaptionedImage]] = None,
        captioned_videos: Union[CaptionedVideo, List[CaptionedVideo]] = None,
    ) -> str:
        # TODO: Handle if nothing is passed to the input

        # Create output directory if it doesn't exist
        base_output_dir = os.path.join(self.output_dir)
        output_dir = self.widgets[0]
        # TODO: Need to set the default values for path_to_wan_video and diffusion_pipe_dir with env variables
        path_to_wan_video = self.widgets[
            1
        ]  # { "type": "text", "value": "/home/saltchicken/.local/diffusion-pipe/models/Wan2.1-T2V-14B-480P"}
        diffusion_pipe_dir = self.widgets[
            2
        ]  # { "type": "text", "value": "/home/saltchicken/.local/diffusion-pipe"}
        status = self.widgets[3]  # {"type": "textarea", "value": ""}
        output_dir_dataset_input = os.path.join(base_output_dir, output_dir, "input")

        if os.path.exists(os.path.join(base_output_dir, output_dir)):
            # await self.set_status("Output path already exists. Please choose a different path.")
            print("output path already exists.")
            # TODO: Better handling for directories that already exist
            return None

        print("Creating output directories since it doesn't exist")
        os.makedirs(os.path.join(base_output_dir, output_dir), exist_ok=True)
        os.makedirs(output_dir_dataset_input)

        # Handle images if present
        if captioned_images is not None:
            images = (
                [captioned_images]
                if isinstance(captioned_images, CaptionedImage)
                else captioned_images
            )
            save_captioned_images(images, output_dir_dataset_input)

        # Handle videos if present
        if captioned_videos is not None:
            videos = (
                [captioned_videos]
                if isinstance(captioned_videos, CaptionedVideo)
                else captioned_videos
            )
            save_captioned_videos(videos, output_dir_dataset_input)

        dataset_config = create_dataset_toml(output_dir_dataset_input)

        # Write the TOML file
        data_set_toml_path = os.path.join(base_output_dir, output_dir, "dataset.toml")
        with open(os.path.join(data_set_toml_path), "w") as f:
            toml.dump(dataset_config, f)

        config = create_wan_video_toml(
            output_dir=os.path.join(base_output_dir, output_dir, "output"),
            dataset_path=data_set_toml_path,
            model_ckpt_path=path_to_wan_video,
        )

        video_toml_path = os.path.join(base_output_dir, output_dir, "wan_video.toml")
        with open(os.path.join(video_toml_path), "w") as f:
            toml.dump(config, f)

        conda_exec = os.path.join(os.environ.get("CONDA_EXE", "conda"))

        custom_env = os.environ.copy()
        custom_env["NCCL_P2P_DISABLE"] = "1"
        custom_env["NCCL_IB_DISABLE"] = "1"

        conda_env = "diffusion-pipe"
        print("RUNNING")
        full_command = f"{conda_exec} run --live-stream -n {conda_env} deepspeed --num_gpus=1 {diffusion_pipe_dir}/train.py --deepspeed --config {video_toml_path}"
        print(full_command)
        print("RUNNING")
        command_list = full_command.split()
        await run_command(command_list)

        return output_dir_dataset_input


def save_captioned_images(captioned_images, output_dir):
    for idx, img_data in enumerate(captioned_images):
        base64_data = img_data.image.split(",")[1]
        image_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_data))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}_{idx}"

        image_filename = f"{filename}.png"
        output_path = os.path.join(output_dir, image_filename)
        img.save(output_path)

        text_filename = f"{filename}.txt"
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, "w") as f:
            f.write(img_data.caption)


def save_captioned_videos(captioned_videos, output_dir):
    for idx, video_data in enumerate(captioned_videos):
        base64_data = video_data.video.split(",")[1]
        video_bytes = base64.b64decode(base64_data)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}_{idx}"

        video_filename = f"{filename}.mp4"
        video_path = os.path.join(output_dir, video_filename)
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        text_filename = f"{filename}.txt"
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, "w") as f:
            f.write(video_data.caption)


async def run_command(command_list):
    try:
        process = await asyncio.create_subprocess_exec(
            *command_list,  # Unpack the list into arguments
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
        )

        while True:
            line_bytes = await process.stdout.readline()

            if not line_bytes:
                break

            # Decode the bytes to string (assuming utf-8) and strip whitespace
            line_content = line_bytes.decode("utf-8", errors="replace").strip()
            print(f"Received: {line_content}")
            # await self.update_widget("status", line_content)

            if "error" in line_content.lower():
                print(f"ACTION: Found 'error'!")
                # process.terminate()
                # try:
                #     # Wait briefly for graceful termination
                #     await asyncio.wait_for(process.wait(), timeout=2.0)
                # except asyncio.TimeoutError:
                #     print("ACTION: Process did not terminate gracefully, killing.")
                #     process.kill()
                # break

            await asyncio.sleep(0.01)  # Yield control briefly

        await process.wait()
        print(f"--- Command finished with exit code: {process.returncode} ---")

        if process.returncode != 0:
            print(
                f"Warning: Command exited with non-zero status ({process.returncode})."
            )

    except FileNotFoundError:
        # Check if 'conda' itself wasn't found
        if command_list[0] == "conda":
            print(
                "Error: 'conda' command not found. Is Conda installed and in your PATH?",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: Command or script not found within the environment: {command_list}",
                file=sys.stderr,
            )
        # Ensure process is cleaned up if creation failed partially (unlikely here, but good practice)
        if process and process.returncode is None:
            process.kill()

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        # Ensure process is cleaned up if an error occurred during execution
        if process and process.returncode is None:
            process.kill()


def create_dataset_toml(
    directory_path: str,
    resolutions: List[int] = [256],
    min_ar: float = 0.5,
    max_ar: float = 2.0,
    num_ar_buckets: int = 7,
    frame_buckets: List[int] = [1, 33],
    num_repeats: int = 5,
) -> dict:
    """
    Creates a dataset.toml file for training configuration.

    Args:
        output_path: Path where to save the TOML file
        directory_path: Path to the input directory
        resolutions: List of resolution values
        min_ar: Minimum aspect ratio
        max_ar: Maximum aspect ratio
        num_ar_buckets: Number of aspect ratio buckets
        frame_buckets: List of frame bucket values
        num_repeats: Number of repeats for directory

    Returns:
        dict: TOML file
    """

    dataset_config = {
        "resolutions": resolutions,
        "min_ar": min_ar,
        "max_ar": max_ar,
        "num_ar_buckets": num_ar_buckets,
        "frame_buckets": frame_buckets,
        "directory": [{"path": directory_path, "num_repeats": num_repeats}],
    }

    return dataset_config


def create_wan_video_toml(
    output_dir: str,
    dataset_path: str,
    epochs: int = 100,
    micro_batch_size: int = 1,
    pipeline_stages: int = 1,
    gradient_accumulation_steps: int = 4,
    gradient_clipping: float = 1.0,
    warmup_steps: int = 100,
    eval_every_n_epochs: int = 1,
    save_every_n_epochs: int = 5,
    checkpoint_every_n_minutes: int = 30,
    model_ckpt_path: str = "/path/to/Wan2.1-T2V-14B-480P",
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> dict:
    """
    Creates a wan_video.toml file for training configuration.

    Args:
        output_path: Path where to save the TOML file
        output_dir: Directory for output files
        dataset_path: Path to the dataset.toml file
        epochs: Number of training epochs
        micro_batch_size: Micro batch size per GPU
        pipeline_stages: Number of pipeline stages
        gradient_accumulation_steps: Number of gradient accumulation steps
        gradient_clipping: Gradient clipping value
        warmup_steps: Number of warmup steps
        eval_every_n_epochs: Evaluation frequency in epochs
        save_every_n_epochs: Model saving frequency in epochs
        checkpoint_every_n_minutes: Checkpoint saving frequency in minutes
        model_ckpt_path: Path to the model checkpoint
        learning_rate: Learning rate for training
        weight_decay: Weight decay for optimizer

    Returns:
        dict: TOML file
    """

    config = {
        "output_dir": output_dir,
        "dataset": dataset_path,
        "epochs": epochs,
        "micro_batch_size_per_gpu": micro_batch_size,
        "pipeline_stages": pipeline_stages,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "warmup_steps": warmup_steps,
        "eval_every_n_epochs": eval_every_n_epochs,
        "eval_before_first_step": True,
        "eval_micro_batch_size_per_gpu": micro_batch_size,
        "eval_gradient_accumulation_steps": 1,
        "save_every_n_epochs": save_every_n_epochs,
        "checkpoint_every_n_minutes": checkpoint_every_n_minutes,
        "activation_checkpointing": True,
        "partition_method": "parameters",
        "save_dtype": "bfloat16",
        "caching_batch_size": 1,
        "steps_per_print": 1,
        "video_clip_mode": "single_middle",
        "model": {
            "type": "wan",
            "ckpt_path": model_ckpt_path,
            "dtype": "bfloat16",
            "transformer_dtype": "float8",
            "timestep_sample_method": "logit_normal",
        },
        "adapter": {"type": "lora", "rank": 32, "dtype": "bfloat16"},
        "optimizer": {
            "type": "adamw_optimi",
            "lr": learning_rate,
            "betas": [0.9, 0.99],
            "weight_decay": weight_decay,
            "eps": 1e-8,
        },
    }

    return config
