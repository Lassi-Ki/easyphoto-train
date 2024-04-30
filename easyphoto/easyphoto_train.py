import os
import subprocess
import sys
import numpy as np
from glob import glob
from PIL import Image, ImageOps
from easyphoto.easyphoto_config import (
        cache_log_file_path,
        easyphoto_models_path,
        models_path,
        user_id_outpath_samples,
        validation_prompt,
)
from easyphoto.easyphoto_config import (data_dir, get_bucket_and_key,
                                        generated_lora_s3uri, s3_client,
                                        generated_ref_s3uri)


config_sdxl = os.path.join(data_dir, "sd_xl_base.yaml")
python_executable_path = sys.executable


def easyphoto_train_forward(
    sd_model_checkpoint: str,
    user_id: str,
    unique_id: str,
    train_mode_choose: str,
    resolution: int,
    val_and_checkpointing_steps: int,
    max_train_steps: int,
    steps_per_photos: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    dataloader_num_workers: int,
    learning_rate: float,
    rank: int,
    network_alpha: int,
    instance_images: list,
    skin_retouching_bool: bool,
    training_prefix_prompt: str,
    crop_ratio: float,
    *args,
):
    print(f'lora training v2: {unique_id} start...')

    # -------------------------------------------------------------------------------------
    # ----------------------------  阶段一  环境准备  ----------------------------------------
    # -------------------------------------------------------------------------------------
    if len(instance_images) == 0:
        return "instance_images list is empty."
    if int(rank) < int(network_alpha):
        return "The network alpha {} must not exceed rank {}. " "It will result in an unintended LoRA.".format(network_alpha, rank)
    if int(resolution) < 1024:
        return "The resolution for SDXL Training needs to be 1024."

    cache_outpath_samples = user_id_outpath_samples

    # Template address
    training_templates_path = os.path.join(easyphoto_models_path, "training_templates")
    # Raw data backup
    original_backup_path = os.path.join(cache_outpath_samples, unique_id, "original_backup")
    # Reference backup of face
    ref_image_path = os.path.join(cache_outpath_samples, unique_id, "ref_image.jpg")
    # Training data retention
    user_path = os.path.join(cache_outpath_samples, unique_id, "processed_images")
    images_save_path = os.path.join(cache_outpath_samples, unique_id, "processed_images", "train")
    json_save_path = os.path.join(cache_outpath_samples, unique_id, "processed_images", "metadata.jsonl")
    # Training weight saving
    weights_save_path = os.path.join(cache_outpath_samples, unique_id, "user_weights")
    webui_save_path = os.path.join(models_path, f"Lora/{unique_id}.safetensors")
    webui_load_path = os.path.join(models_path, f"Stable-diffusion", sd_model_checkpoint)
    sd_save_path = os.path.join(easyphoto_models_path, "stable-diffusion-xl/stabilityai_stable_diffusion_xl_base_1.0")

    os.makedirs(original_backup_path, exist_ok=True)
    os.makedirs(user_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(webui_save_path)), exist_ok=True)

    max_train_steps = int(min(len(instance_images) * int(steps_per_photos), int(max_train_steps)))
    print("max_train_steps: ", max_train_steps)

    for index, user_image in enumerate(instance_images):
        image = Image.open(user_image)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.save(os.path.join(original_backup_path, str(index) + ".jpg"))

    # -------------------------------------------------------------------------------------
    # --------------------------  阶段二  训练图像预处理  -------------------------------------
    # -------------------------------------------------------------------------------------
    preprocess_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess.py")
    command = [
        f"{python_executable_path}",
        f"{preprocess_path}",
        f"--images_save_path={images_save_path}",
        f"--json_save_path={json_save_path}",
        f"--validation_prompt={validation_prompt}",
        f"--inputs_dir={original_backup_path}",
        f"--ref_image_path={ref_image_path}",
        f"--crop_ratio={crop_ratio}",
    ]
    if skin_retouching_bool:
        command += ["--skin_retouching_bool"]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")

    # check preprocess results
    train_images = glob(os.path.join(images_save_path, "*.jpg"))
    if len(train_images) == 0:
        return "Failed to obtain preprocessed images, please check the preprocessing process."
    if not os.path.exists(json_save_path):
        return "Failed to obtain preprocessed metadata.jsonl, please check the preprocessing process."

    # -------------------------------------------------------------------------------------
    # ---------------------------  阶段三  XL Lora训练  -------------------------------------
    # -------------------------------------------------------------------------------------
    train_kohya_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_kohya/train_lora_sd_XL.py")
    if not os.path.exists(os.path.dirname(cache_log_file_path)):
        os.makedirs(os.path.dirname(cache_log_file_path), exist_ok=True)
    sdxl_model_dir = os.path.join(easyphoto_models_path, "stable-diffusion-xl")
    pretrained_vae_model_name_or_path = os.path.join(sdxl_model_dir, "madebyollin_sdxl_vae_fp16_fix")
    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TRANSFORMERS_CACHE"] = sdxl_model_dir
    random_seed = np.random.randint(1, 1e6)
    command = [
        f"{python_executable_path}",
        "-m",
        "accelerate.commands.launch",
        "--mixed_precision=fp16",
        "--main_process_port=3456",
        f"{train_kohya_path}",
        f"--pretrained_model_name_or_path={sd_save_path}",
        f"--pretrained_model_ckpt={webui_load_path}",
        f"--train_data_dir={user_path}",
        "--caption_column=text",
        f"--resolution={resolution}",
        "--random_flip",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        f"--dataloader_num_workers={dataloader_num_workers}",
        f"--max_train_steps={max_train_steps}",
        f"--checkpointing_steps={val_and_checkpointing_steps}",
        f"--learning_rate={learning_rate}",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--train_text_encoder",
        f"--seed={random_seed}",
        f"--rank={rank}",
        f"--network_alpha={network_alpha}",
        f"--validation_prompt={validation_prompt}",
        f"--validation_steps={val_and_checkpointing_steps}",
        f"--output_dir={weights_save_path}",
        f"--logging_dir={weights_save_path}",
        "--enable_xformers_memory_efficient_attention",
        "--mixed_precision=fp16",
        f"--template_dir={training_templates_path}",
        "--template_mask",
        "--merge_best_lora_based_face_id",
        f"--merge_best_lora_name={unique_id}",
        f"--cache_log_file={cache_log_file_path}",
    ]
    command += [f"--original_config={config_sdxl}"]
    command += [f"--pretrained_vae_model_name_or_path={pretrained_vae_model_name_or_path}"]

    try:
        subprocess.run(command, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing the command: {e}")

    best_weight_path = os.path.join(weights_save_path, "pytorch_lora_weights.safetensors")
    if not os.path.exists(best_weight_path):
        print("Failed to obtain Lora after training, please check the training process.")

    # -------------------------------------------------------------------------------------
    # ----------------------------  阶段四  训练结果上传  ------------------------------------
    # -------------------------------------------------------------------------------------
    try:
        post_lora(best_weight_path, user_id, unique_id)
        post_ref(ref_image_path, user_id, unique_id)
    except Exception as e:
        print(f"Error uploading the LoRA to S3: {e}")

    return "The training has been completed."


def post_lora(lora_path, user_id, unique_id):
    bucket, key = get_bucket_and_key(generated_lora_s3uri)
    if key.endswith('/'):
        key = key[:-1]
    key += "/" + user_id
    s3_client.put_object(
        Body=open(lora_path, 'rb'),
        Bucket=bucket,
        Key=f'{key}/{unique_id}.safetensors'
    )
    print(f"Upload Lora successfully to S3: {key}/{unique_id}.safetensors")


def post_ref(ref_path, user_id, unique_id):
    bucket, key = get_bucket_and_key(generated_ref_s3uri)
    if key.endswith('/'):
        key = key[:-1]
    key += "/" + user_id + "/" + unique_id
    s3_client.put_object(
        Body=open(ref_path, 'rb'),
        Bucket=bucket,
        Key=f'{key}/ref_image.jpg'
    )
    print(f"Upload Ref successfully to S3: {key}/ref_image.jpg")
