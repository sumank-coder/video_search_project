import argparse
import glob
import os
from PIL import Image
import sys
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import aiohttp
import asyncio
import subprocess
import numpy as np
import io
import aiofiles
import shutil
import cv2

class options:
  def __init__(self):
    options.img_dir = "gic/static/new_images"
    options.out_dir = os.getcwd()
    options.format = "txt"
    options.nucleus = False
    options.q_factor = 1.0
    options.min_length =22
    options.torch_device = "cpu"

SIZE = 384
BLIP_MODEL_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth'

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--img_dir",
        type=str,
        nargs="?",
        const=True,
        default="gic/static/new_images",
        help="directory with images to be captioned",
    ),
    parser.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        const=True,
        default=os.getcwd(),
        help="directory to put captioned images",
    ),
    parser.add_argument(
        "--format",
        type=str,
        nargs="?",
        const=True,
        default="txt",
        help="'filename', 'mrwho', 'txt', or 'caption'",
    ),
    parser.add_argument(
        "--nucleus",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="use nucleus sampling instead of beam",
    ),
    parser.add_argument(
        "--q_factor",
        type=float,
        nargs="?",
        const=True,
        default=1.0,
        help="adjusts the likelihood of a word being repeated",
    ),
    parser.add_argument(
        "--min_length",
        type=int,
        nargs="?",
        const=True,
        default=22,
        help="adjusts the likelihood of a word being repeated",
    ),
    parser.add_argument(
        "--torch_device",
        type=str,
        nargs="?",
        const=False,
        default="cpu",
        help="specify a different torch device, e.g. 'cpu'",
    ),

    return parser

def load_image(raw_image, device):
    transform = transforms.Compose([
        #transforms.CenterCrop(SIZE),
        transforms.Resize((SIZE, SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image

def get_out_file_name(out_dir, base_name, ext):
    return os.path.join(out_dir, f"{base_name}{ext}")

async def main(opt,vid_file_name):
    print("starting")
    import models.blip

    sample = False
    if opt.nucleus:
        sample = True

    input_dir = opt.img_dir
    print("input_dir: ", input_dir)

    config_path = "scripts/BLIP/configs/med_config.json"

    cache_folder = ".cache"
    model_cache_path = ".cache/model_base_caption_capfilt_large.pth"

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    if not os.path.exists(model_cache_path):
        print(f"Downloading model to {model_cache_path}... please wait")

        async with aiohttp.ClientSession() as session:
            async with session.get(BLIP_MODEL_URL) as res:
                with open(model_cache_path, 'wb') as f:
                    async for chunk in res.content.iter_chunked(1024):
                        f.write(chunk)
        print(f"Model cached to: {model_cache_path}")
    else:
        print(f"Model already cached to: {model_cache_path}")

    blip_decoder = models.blip.blip_decoder(pretrained=model_cache_path, image_size=SIZE, vit='base', med_config=config_path)
    blip_decoder.eval()

    print(f"loading model to {opt.torch_device}")

    blip_decoder = blip_decoder.to(torch.device(opt.torch_device))

    ext = ('.jpg', '.jpeg', '.png', '.webp', '.tif', '.tga', '.tiff', '.bmp', '.gif')

    i = 0

    full_video_caption =""

    for idx, img_file_name in enumerate(glob.iglob(os.path.join(opt.img_dir, "*.*"))):
        if img_file_name.endswith(ext):
            caption = None
            file_ext = os.path.splitext(img_file_name)[1]
            if (file_ext in ext):
                async with aiofiles.open(img_file_name, "rb") as input_file:
                    print("working image: ", img_file_name)

                    image_bin = await input_file.read()
                    image = Image.open(io.BytesIO(image_bin))

                    if not image.mode == "RGB":
                        image = image.convert("RGB")

                    image = load_image(image, device=torch.device(opt.torch_device))

                    if opt.nucleus:
                        captions = blip_decoder.generate(image, sample=True, top_p=opt.q_factor)
                    else:
                        captions = blip_decoder.generate(image, sample=sample, num_beams=16, min_length=opt.min_length, \
                            max_length=48, repetition_penalty=opt.q_factor)

                    caption = captions[0]

                    if opt.format in ["mrwho","joepenna"]:
                        prefix = f"{i:05}@"
                        i += 1
                        caption = prefix+caption
                    elif opt.format == "filename":
                        postfix = f"_{i}"
                        i += 1
                        caption = caption+postfix
                    full_video_caption += (' '+caption)

    if opt.format in ["txt", "text", "caption"]:
        out_base_name = os.path.splitext(os.path.basename(img_file_name))[0]

    if opt.format in ["txt", "text"]:
        out_file = get_out_file_name(opt.out_dir, "image_search_db", ".txt")

    if opt.format in ["caption"]:
        out_file = get_out_file_name(opt.out_dir, out_base_name, ".caption")

    if opt.format in ["txt", "text", "caption"]:
        print("writing caption to: ", out_file)
        async with aiofiles.open(out_file, "a") as out_file:
            txtstr = ""
            head, tail = os.path.split(vid_file_name)
            txtstr += "videos/" + str(tail) + " ::: " + str(full_video_caption) + "\n"
            await out_file.write(txtstr)

    if opt.format in ["filename", "mrwho", "joepenna"]:
        caption = caption.replace("/", "").replace("\\", "")  # must clean slashes using filename
        out_file = get_out_file_name(opt.out_dir, caption, file_ext)
        async with aiofiles.open(out_file, "wb") as out_file:
            await out_file.write(image_bin)
    elif opt.format == "json":
        raise NotImplementedError
    elif opt.format == "parquet":
        raise NotImplementedError

def isWindows():
    return sys.platform.startswith("win")

def refresh_index():
    opt = options()
    os.startfile("activate_venv.bat")

    if opt.format not in ["filename", "mrwho", "joepenna", "txt", "text", "caption"]:
        raise ValueError("format must be 'filename', 'mrwho', 'txt', or 'caption'")
    if (isWindows()):
        print("Windows detected, using asyncio.WindowsSelectorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    else:
        print("Unix detected, using default asyncio event loop policy")
    if not os.path.exists("scripts/BLIP"):
        print("BLIP not found, cloning BLIP repo")
        subprocess.run(["git", "clone", "https://github.com/salesforce/BLIP", "scripts/BLIP"])
    blip_path = "scripts/BLIP"
    sys.path.append(blip_path)
    source_folder = os.path.join(os.getcwd(),"gic/static/new_images")
    destination_folder = os.path.join(os.getcwd(),"gic/static/images")
    source_video_folder = os.path.join(os.getcwd(),"gic/static/new_videos")
    destination_video_folder = os.path.join(os.getcwd(), "gic/static/videos")
    # Convert video to images
    for file_name in os.listdir(source_video_folder):
        # construct full file path
        video_file = os.path.join(source_video_folder , file_name)
        print("video_file name is ", video_file)
        cam = cv2.VideoCapture(video_file)
        currentframe = 0
        while (True):
            ret, frame = cam.read()
            if ret:
                # if video is still left continue creating images
                name = source_folder +'/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)
                cv2.imwrite(name, frame)
                cam.set(cv2.CAP_PROP_POS_MSEC, (currentframe * 1000))  # added this line
                currentframe += 10
            else:
                break
        # Release all space and windows once done
        cam.release()
        cv2.destroyAllWindows()
        # get the caption generated
        asyncio.run(main(opt,file_name))
        # Remove temp image files
        for file_name in os.listdir(source_folder):
            # construct full file path
            #source = source_folder + file_name
            #destination = destination_folder + file_name
            temp_image = os.path.join(source_folder , file_name)
            os.remove(temp_image)


    # copy video file
    for file_name in os.listdir(source_video_folder):
        # construct full file path
        #source = source_folder + file_name
        #destination = destination_folder + file_name
        source = os.path.join(source_video_folder , file_name)
        destination = os.path.join(destination_video_folder , file_name)
        # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            print('copied', file_name)
        os.remove(source)
