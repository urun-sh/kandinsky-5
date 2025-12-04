import argparse
import time
import warnings
import logging

import torch

from kandinsky.utils import set_hf_token
from kandinsky import get_T2V_pipeline, get_I2V_pipeline, get_T2I_pipeline, get_I2I_pipeline

import lovely_tensors as lt
lt.monkey_patch()

def validate_args(args):
    size = (args.width, args.height)
    if "i2i" in args.config:
        return
    elif "t2i" in args.config:
        supported_sizes = [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)]
    else:
        supported_sizes = [(512, 512), (512, 768), (768, 512), (1280, 768), (768, 1280), (1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)]
    if not size in supported_sizes:
        raise NotImplementedError(
            f"Provided size of video is not supported: {size}")


def disable_warnings():
    warnings.filterwarnings("ignore")
    logging.getLogger("torch").setLevel(logging.ERROR)
    torch._logging.set_logs(
        dynamo=logging.ERROR,
        dynamic=logging.ERROR,
        aot=logging.ERROR,
        inductor=logging.ERROR,
        guards=False,
        recompiles=False
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a video using Kandinsky 5"
    )
    parser.add_argument(
        '--local-rank',
        type=int,
        help='local rank'
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/k5_pro_t2v_5s_sft_sd.yaml",
        help="The config file of the model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The bear plays balalaika.",
        help="The prompt to generate video"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./assets/test_image.jpg",
        help="An image to generate image/video from."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        help="Negative prompt for classifier-free guidance"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        choices=[512, 640, 768, 896, 1152, 1024, 1280],
        help="Width of the video in pixels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        choices=[512, 640, 768, 896, 1152, 1024, 1280],
        help="Height of the video in pixels"
    )
    parser.add_argument(
        "--video_duration",
        type=int,
        default=5,
        help="Duratioin of the video in seconds"
    )
    parser.add_argument(
        "--expand_prompt",
        type=int,
        default=0,
        help="Whether to use prompt expansion."
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=None,
        help="The sampling steps number."
    )
    parser.add_argument(
        "--guidance_weight",
        type=float,
        default=None,
        help="Guidance weight."
    )
    parser.add_argument(
        "--scheduler_scale",
        type=float,
        default=5.0,
        help="Scheduler scale."
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Name of the resulting file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1137,
        help="Seed for the random number generator"
    )

    parser.add_argument(
        "--offload",
        action='store_true',
        default=False,
        help="Offload models to save memory or not"
    )
    parser.add_argument(
        "--magcache",
        action='store_true',
        default=False,
        help="Using MagCache (for 50 steps models only)"
    )
    parser.add_argument(
        "--qwen_quantization",
        action='store_true',
        default=False,
        help="Use quantized Qwen2.5-VL model (4-bit quantization)"
    )
    parser.add_argument(
        "--attention_engine",
        type=str,
        default="auto",
        help="Name of the full attention algorithm to use for <=5 second generation",
        choices=["flash_attention_2", "flash_attention_3", "sdpa", "sage", "auto"]
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="token to download restricted models like FLUX.1-dev VAE",
    )
    args = parser.parse_args()

    if args.hf_token:
        set_hf_token(args.hf_token)

    return args


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    disable_warnings()
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)
    device_map = {"dit": "cuda:0", "vae": "cuda:0",
                  "text_embedder": "cuda:0"}

    if "t2i" in args.config:
        pipe = get_T2I_pipeline(
            device_map=device_map,
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
        )
    elif "i2i" in args.config:
        pipe = get_I2I_pipeline(
            device_map=device_map,
            resolution=1024,
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
        )
    elif "i2v" in args.config:
        pipe = get_I2V_pipeline(
            device_map=device_map,
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
        )
    else:
        pipe = get_T2V_pipeline(
            device_map=device_map,
            conf_path=args.config,
            offload=args.offload,
            magcache=args.magcache,
            quantized_qwen=args.qwen_quantization,
            attention_engine=args.attention_engine,
        )

    if args.output_filename is None:
        args.output_filename = "./" + args.prompt.replace(" ", "_")
        if len(args.output_filename) > 32:
            args.output_filename = args.output_filename[:32]
        if "t2i" in args.config or "i2i" in args.config:
            args.output_filename = args.output_filename + ".png"
        else:
            args.output_filename = args.output_filename + ".mp4"

    import copy
    import time

    import torch

    def print_menu():
        print("\n" + "=" * 40)
        print(" Kandinsky TUI Generator ")
        print("=" * 40)
        print("Enter your generation settings below:")
        print("  (Press Enter to keep default/current values)")
        print("-" * 40)

    # Helper function to broadcast python objects (strings, ints, etc) to all ranks
    def broadcast_obj(obj, src=0):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # Convert obj to string for broadcast (most robust for arbitrary python objects)
            import pickle
            obj_bytes = pickle.dumps(obj)
            obj_size = torch.tensor([len(obj_bytes)], dtype=torch.int64, device="cuda")
            torch.distributed.broadcast(obj_size, src)
            # Prepare buffer
            buf = torch.empty(obj_size.item(), dtype=torch.uint8, device="cuda")
            if torch.distributed.get_rank() == src:
                buf[:] = torch.tensor(list(obj_bytes), dtype=torch.uint8, device="cuda")
            torch.distributed.broadcast(buf, src)
            if torch.distributed.get_rank() != src:
                obj_bytes = bytes(buf.cpu().tolist())
                obj = pickle.loads(obj_bytes)
        return obj

    def get_input(prompt_str, current_val, _type=str, rank0=True):
        if rank0:
            prompt_full = f"{prompt_str} [{current_val}]: "
            inp = input(prompt_full)
            if inp.strip() == "":
                result = current_val
            else:
                try:
                    result = _type(inp)
                except Exception as e:
                    print(f"Invalid input ({e}), using current value.")
                    result = current_val
        else:
            result = None  # will receive from rank 0
        return broadcast_obj(result, src=0)

    # Determine distributed setup
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        is_rank0 = rank == 0
    else:
        rank = 0
        is_rank0 = True

    # For safety, as args will be mutated
    current_args = copy.deepcopy(args)

    while True:
        if is_rank0:
            print_menu()
        # Prompt for all relevant arguments interactively (input only on rank0 and broadcast!)
        current_args.prompt = get_input("Prompt", current_args.prompt, str, rank0=is_rank0)

        if (
            "t2i" in current_args.config
            or "i2i" in current_args.config
            or not ("t2i" in current_args.config or "i2i" in current_args.config or "i2v" in current_args.config)
        ):
            current_args.width = get_input("Width", current_args.width, int, rank0=is_rank0)
            current_args.height = get_input("Height", current_args.height, int, rank0=is_rank0)
        if "i2i" in current_args.config or "i2v" in current_args.config:
            current_args.image = get_input("Image path", current_args.image, str, rank0=is_rank0)
        if "i2v" in current_args.config or not ("t2i" in current_args.config or "i2i" in current_args.config or "i2v" in current_args.config):
            current_args.video_duration = get_input("Video duration (s)", current_args.video_duration, int, rank0=is_rank0)
        current_args.sample_steps = get_input("Sample steps", current_args.sample_steps, int, rank0=is_rank0)
        current_args.guidance_weight = get_input("Guidance weight", current_args.guidance_weight, float, rank0=is_rank0)
        current_args.scheduler_scale = get_input("Scheduler scale", current_args.scheduler_scale, float, rank0=is_rank0)
        current_args.expand_prompt = get_input(
            "Expand prompt (True/False)", current_args.expand_prompt,
            lambda x: x.lower() in ["true", "1", "yes", "y"], rank0=is_rank0
        )
        current_args.seed = get_input("Seed", current_args.seed, int, rank0=is_rank0)
        out_fn = get_input("Output filename", current_args.output_filename, str, rank0=is_rank0)
        current_args.output_filename = out_fn

        if is_rank0:
            print("\nGenerating... Please wait.")
        torch.distributed.barrier() if torch.distributed.is_available() and torch.distributed.is_initialized() else None
        # Only for timing print, use rank0
        if is_rank0:
            start_time = time.perf_counter()

        if "t2i" in current_args.config:
            x = pipe(
                current_args.prompt,
                width=current_args.width,
                height=current_args.height,
                num_steps=current_args.sample_steps,
                guidance_weight=current_args.guidance_weight,
                scheduler_scale=current_args.scheduler_scale,
                expand_prompts=current_args.expand_prompt,
                save_path=current_args.output_filename,
                seed=current_args.seed,
            )
        elif "i2i" in current_args.config:
            x = pipe(
                current_args.prompt,
                image=current_args.image,
                num_steps=current_args.sample_steps,
                guidance_weight=current_args.guidance_weight,
                scheduler_scale=current_args.scheduler_scale,
                expand_prompts=current_args.expand_prompt,
                save_path=current_args.output_filename,
                seed=current_args.seed,
            )
        elif "i2v" in current_args.config:
            x = pipe(
                current_args.prompt,
                image=current_args.image,
                time_length=current_args.video_duration,
                num_steps=current_args.sample_steps,
                guidance_weight=current_args.guidance_weight,
                scheduler_scale=current_args.scheduler_scale,
                expand_prompts=current_args.expand_prompt,
                save_path=current_args.output_filename,
                seed=current_args.seed,
            )
        else:
            x = pipe(
                current_args.prompt,
                time_length=current_args.video_duration,
                width=current_args.width,
                height=current_args.height,
                num_steps=current_args.sample_steps,
                guidance_weight=current_args.guidance_weight,
                scheduler_scale=current_args.scheduler_scale,
                expand_prompts=current_args.expand_prompt,
                save_path=current_args.output_filename,
                seed=current_args.seed,
            )

        if is_rank0:
            elapsed = time.perf_counter() - start_time
            print(f"\n\033[92mTIME ELAPSED: {elapsed:.2f} seconds\033[0m")
            print(f"\033[94mGenerated file saved to: {current_args.output_filename}\033[0m")
            print("\nWould you like to generate another file? (Y/n)")
            again = input("> ").strip().lower()
        else:
            again = None
        again = broadcast_obj(again, src=0)
        if again and again != "y" and again != "yes":
            if is_rank0:
                print("Goodbye!")
            break

