import argparse
import torch
import os
from easyphoto.easyphoto_train import easyphoto_train_forward
from easyphoto.easyphoto_down import download_dataset_from_s3, down_sd_model
from easyphoto.easyphoto_train import models_path
from easyphoto.easyphoto_config import easyphoto_models_path


def download_model_from_s3():
    down_sd_model("", os.path.join(models_path, f"Stable-diffusion"))
    down_sd_model("", os.path.join(os.path.join(easyphoto_models_path, "stable-diffusion-xl"),
                                                      "madebyollin_sdxl_vae_fp16_fix"))


def training():
    # download_model_from_s3()
    # user_path = f'./datasets/{opt.user_id}/{opt.unique_id}'
    # if opt.s3Url != '':
    #     download_dataset_from_s3(opt.s3Url, user_path)
    #     print(f'download dataset from s3: {opt.s3Url} success.')
    # else:
    #     print(f'The dataset is empty.')
    #     return f'The dataset is empty.'

    img_list = os.listdir("/opt/ml/input/data/images/")
    instance_images = []
    for idx, img_path in enumerate(img_list):
        img_path = os.path.join("/opt/ml/input/data/images/", img_path)
        instance_images.append(img_path)
    print("instance_images: ", instance_images)

    try:
        message = easyphoto_train_forward(
            user_id=opt.user_id,
            unique_id=opt.unique_id,
            train_mode_choose=opt.train_mode_choose,
            resolution=opt.resolution,
            val_and_checkpointing_steps=opt.val_and_checkpointing_steps,
            max_train_steps=opt.max_train_steps,
            steps_per_photos=opt.steps_per_photos,
            train_batch_size=opt.train_batch_size,
            gradient_accumulation_steps=opt.gradient_accumulation_steps,
            dataloader_num_workers=opt.dataloader_num_workers,
            learning_rate=opt.learning_rate,
            rank=opt.rank,
            network_alpha=opt.network_alpha,
            instance_images=instance_images,
            skin_retouching_bool=opt.skin_retouching_bool,
            training_prefix_prompt=opt.training_prefix_prompt,
            crop_ratio=opt.crop_ratio
        )
    except Exception as e:
        torch.cuda.empty_cache()
        message = f"Train error, error info:{str(e)}"
        print(message)
    return {"message": message}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--sd_model_checkpoint', type=str, default="sd_xl_base_1.0.safetensors")
    parser.add_argument('--user_id', type=str, default="test")
    parser.add_argument('--unique_id', type=str, default="test_00")
    parser.add_argument('--train_mode_choose', type=str, default="Train Human Lora")
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--val_and_checkpointing_steps', type=int, default=100)
    parser.add_argument('--max_train_steps', type=int, default=400)
    parser.add_argument('--steps_per_photos', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--dataloader_num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--network_alpha', type=int, default=16)
    parser.add_argument('--skin_retouching_bool', type=bool, default=True)
    parser.add_argument('--training_prefix_prompt', type=str, default="")
    parser.add_argument('--crop_ratio', type=float, default=3)

    opt = parser.parse_args()

    training()
