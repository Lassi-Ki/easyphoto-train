import os
import requests
import boto3
from tqdm import tqdm
from easyphoto.easyphoto_config import data_path, easyphoto_models_path, models_path, tryon_gallery_dir


# download path
controlnet_extensions_path = os.path.join(data_path, "extensions", "sd-webui-controlnet")
controlnet_extensions_builtin_path = os.path.join(data_path, "extensions-builtin", "sd-webui-controlnet")
models_annotator_path = os.path.join(data_path, "models")
if os.path.exists(controlnet_extensions_path):
    controlnet_annotator_cache_path = os.path.join(controlnet_extensions_path, "annotator/downloads/openpose")
elif os.path.exists(controlnet_extensions_builtin_path):
    controlnet_annotator_cache_path = os.path.join(controlnet_extensions_builtin_path, "annotator/downloads/openpose")
else:
    controlnet_annotator_cache_path = os.path.join(models_annotator_path, "annotator/downloads/openpose")

download_urls = {
    "sdxl": [
        # sdxl
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors",
        "https://pai-aigc-photog.oss-cn-hangzhou.aliyuncs.com/webui/madebyollin-sdxl-vae-fp16-fix.safetensors",
    ],
}

save_filenames = {
    "sdxl": [
        os.path.join(easyphoto_models_path, "stable-diffusion-xl/madebyollin_sdxl_vae_fp16_fix/diffusion_pytorch_model.safetensors"),
        os.path.join(models_path, f"VAE/madebyollin-sdxl-vae-fp16-fix.safetensors"),
    ],
}


def urldownload_progressbar(url, file_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()


def check_files_exists_and_download(check_hash, download_mode="base"):
    urls, filenames = download_urls[download_mode], save_filenames[download_mode]

    for url, filename in zip(urls, filenames):
        if type(filename) is str:
            filename = [filename]

        exist_flag = False
        for _filename in filename:
            if not check_hash:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
            else:
                if os.path.exists(_filename):
                    exist_flag = True
                    break
        if exist_flag:
            continue

        os.makedirs(os.path.dirname(filename[0]), exist_ok=True)
        urldownload_progressbar(url, filename[0])


def download_dataset_from_s3(s3uri, path):
    if not os.path.exists(path):
        os.makedirs(path)
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    print("bucket:", bucket)
    print("key:", key)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.filter(Prefix=key):
        if obj.key[-1] == '/':
            continue
        target = os.path.join(path, os.path.relpath(obj.key, key))
        print("target:", target)
        if not target.endswith(('.jpg', '.jpeg', '.png')):
            continue
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        bucket.download_file(obj.key, target)


def down_sd_model(s3uri, path):
    if not os.path.exists(path):
        os.makedirs(path)

    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)

    bucket.download_file(key, os.path.join(path, os.path.basename(key)))
    if "vae" in s3uri:
        print("download vae model successfully.")
    else:
        print("download xl base model successfully.")


def down_easyphoto_model():
    check_files_exists_and_download(True, "sdxl")
