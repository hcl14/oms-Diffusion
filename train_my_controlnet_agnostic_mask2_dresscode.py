import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from utils.utils import is_torch2_available, prepare_image, prepare_mask
if is_torch2_available():
    from garment_adapter.attention_processor import REFAttnProcessor2_0 as REFAttnProcessor
    from garment_adapter.attention_processor import AttnProcessor2_0 as AttnProcessor
else:
    from garment_adapter.attention_processor import REFAttnProcessor, AttnProcessor

# from dresscode_dataset import get_opt, DressCodeDataset, DressCodeDataLoader
# from dresscode_aligned_dataset import AlignedDataset, TrainOptions
from dresscode_aligned_dataset_hdf5 import AlignedDatasetHDF5, TrainOptions

from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline, StableDiffusionControlNetPipeline

import bitsandbytes as bnb

import copy
from safetensors import safe_open

from torchvision.utils import save_image

# get list of keys:

orig_model_path='checkpoints/oms-diffusion/oms_diffusion_100000.safetensors'

garment_keys = []
with safe_open(orig_model_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        garment_keys.append(key)



def decode_latents(vae, latents):

    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def save_img(tensr, name):
    tensr = tensr * 0.5 + 0.5
    torchvision.utils.save_image(tensr[0, :, :, :],name)

# taken from DDIM scheduler
def scheduler_get_noise(
        scheduler,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        alphas_cumprod = scheduler.alphas_cumprod.to(device=original_samples.device)
        #alphas_cumprod = alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        #noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        #return noisy_samples
        return sqrt_alpha_prod, sqrt_one_minus_alpha_prod


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    #text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    category = torch.stack([example["category_id"] for example in data], dim=0)
    clip_images = torch.stack([example["color"] for example in data])
    mask_images = torch.stack([example["edge"] for example in data])
    drop_image_embeds = torch.stack([example["drop_image_embed"] for example in data])
    drop_prompt_embeds = torch.stack([example["drop_prompt_embed"] for example in data])
    pose_images = torch.stack([example["openpose"] for example in data])
    mask_agnostic = torch.stack([example["person_clothes_mask"] for example in data])

    return {
        "image": images,
        #"text_input_ids": text_input_ids,
        "category": category,
        "cloth": clip_images,
        "cloth_mask": mask_images,
        "drop_image_embed": drop_image_embeds,
        "drop_prompt_embed": drop_image_embeds,
        "pose": pose_images,
        "agnostic_mask": mask_agnostic
    }


#control_net_openpose = ControlNetModel.from_single_file("/workspace/ControlNet-v1-1/control_v11p_sd15_openpose.pth", torch_dtype=torch.float32)

control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", cache_dir="cache", torch_dtype=torch.float32)

class ClothAdapter(torch.nn.Module):
    """Cloth-Adapter"""
    def __init__(self, unet, ref_path=None, weight_dtype=torch.float32):
        super().__init__()
        self.unet = unet
        self.ref_path = ref_path
        #self.adapter_modules = adapter_modules
        self.device = 'cuda'

        self.set_adapter(self.unet, "write")

        ref_unet = copy.deepcopy(self.unet)  # copies with weights


        torch.save(self.unet.state_dict(), 'infer_unet.pt')

        self.controlnet = control_net_openpose

        # freeze controlnet
        # freeze unet
        for name, param in self.controlnet.named_parameters():
            #if param.requires_grad is True:
            param.requires_grad = False



        # load weights if they exist
        state_dict = {}
        if ref_path is not None and os.path.exists(self.ref_path):
            '''
            with safe_open(self.ref_path, framework="pt", device="cpu") as f:
                for key in garment_keys:
                    state_dict[key] = f.get_tensor(key)
            '''
            ref_unet.load_state_dict(torch.load(self.ref_path), strict=False)
            print("Cloth adapter unet loaded from checkpoint")


        self.ref_unet = ref_unet.to(self.device)


        self.set_adapter(self.ref_unet, "read")

        # if ckpt_path is not None:
        #     self.load_from_checkpoint(ckpt_path)

        # freeze unet
        for name, param in self.unet.named_parameters():
            #if param.requires_grad is True:
            param.requires_grad = False

        # freeze rest of ref_unet
        for name, param in self.ref_unet.named_parameters():
            if not name in garment_keys:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.attn_store = {}


        print('Initialized cloth adapter')


    def forward(self, noisy_latents, controlnet_image, timesteps, encoder_hidden_states, prompt_embeds_adapter, cloth_embeds, num_images_per_prompt):
        '''
        cloth = prepare_image(cloth_image, height, width)
        cloth_mask = prepare_mask(cloth_mask_image, height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=torch.float16)

        prompt_embeds_null = self.pipe.encode_prompt([""], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
        cloth_embeds = self.pipe.vae.encode(cloth).latent_dist.mode() * self.pipe.vae.config.scaling_factor
        '''
        down_block_res_samples, mid_block_res_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=1.0, #cond_scale,
                    guess_mode=False,
                    return_dict=False,

                )



        # compute cross_attention_kwargs
        self.ref_unet(cloth_embeds, 0, prompt_embeds_adapter, cross_attention_kwargs={"attn_store": self.attn_store})

        # Predict the noise residual
        noise_pred = self.unet(noisy_latents,
                               timesteps,
                               encoder_hidden_states,
                               cross_attention_kwargs={"attn_store": self.attn_store},
                               down_block_additional_residuals=down_block_res_samples,
                               mid_block_additional_residual=mid_block_res_sample,
                               ).sample
        return noise_pred

    def set_adapter(self, unet, type):
        attn_procs = {}
        for name in unet.attn_processors.keys():
            if "attn1" in name:
                attn_procs[name] = REFAttnProcessor(name=name, type=type)
            else:
                attn_procs[name] = AttnProcessor()
        unet.set_attn_processor(attn_procs)

    def save_checkpoint(self,name=None):

        d = self.ref_unet.state_dict()
        d_to_save = {k:v for k, v in d.items() if k in garment_keys}

        if name:
            torch.save(d_to_save, name)
            print(f'saved custom checkpoint to {name}')
        else:
            torch.save(d_to_save, self.ref_path)
            print(f'saved custom checkpoint to {self.ref_path}')


    def generate(
            self,
            pipe,
            weight_dtype,
            cloth_keyword = "",
            cloth_image='data/zalando/train/cloth/14114_00.jpg',
            cloth_mask_image='data/zalando/train/cloth-mask/14114_00.jpg',
            openpose_image='data/zalando/train/openpose_img/14114_00_rendered.png',
            prompt=None,
            a_prompt="best quality, high quality",
            num_images_per_prompt=1,
            negative_prompt=None,
            seed=1,
            guidance_scale=2.5,
            num_inference_steps=20,
            height=512,
            width=384,
            control_guidance_start=0.0,
            control_guidance_end=2.0,
            **kwargs,
    ):
        if cloth_mask_image is None:
            cloth_mask_image = generate_mask(cloth_image, net=self.seg_net, device=self.device)

        cloth = prepare_image(Image.open(cloth_image).convert('RGB'), height, width)

        cloth_mask = prepare_mask(Image.open(cloth_mask_image).convert('RGB'), height, width)
        cloth = (cloth * cloth_mask).to(self.device, dtype=weight_dtype) #torch.float16)

        if prompt is None:
            prompt = "a photography of a model"
        prompt = prompt + ", " + a_prompt
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = pipe.encode_prompt([cloth_keyword], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = pipe.vae.encode(cloth).latent_dist.mode() * pipe.vae.config.scaling_factor
            cloth_embeds = cloth_embeds.to(dtype=torch.float32)
            prompt_embeds_null = prompt_embeds_null.to(dtype=torch.float32)

            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

            for k, v in self.attn_store.items():
                v = v.to(dtype=torch.float32)

            # we have something wrong with original unet after training
            #unet = pipe.unet
            #pipe.unet = copy.deepcopy(pipe.unet)

            #pipe.unet.load_state_dict(torch.load('infer_unet.pt'), strict=False)

            #pipe.unet.to('cuda', dtype=torch.float32)
            #pipe.vae.to(dtype=torch.float32)

            self.set_adapter(pipe.unet, "write")

            img = Image.open(openpose_image)

            generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
            images = pipe(
                image=img,
                prompt_embeds=prompt_embeds.to(dtype=torch.float32),
                negative_prompt_embeds=negative_prompt_embeds.to(dtype=torch.float32),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                height=height,
                width=width,
                cross_attention_kwargs={"attn_store": self.attn_store, "do_classifier_free_guidance": guidance_scale > 1.0},
                **kwargs,
            ).images

            #del(pipe.unet)
            #pipe.vae.to(dtype=weight_dtype)
            #pipe.unet = unet


        return images







    # def load_from_checkpoint(self, ckpt_path: str):
    #     # Calculate original checksums
    #     orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

    #     state_dict = torch.load(ckpt_path, map_location="cpu")

    #     # Load state dict for adapter_modules
    #     self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

    #     # Calculate new checksums
    #     new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

    #     # Verify if the weights have changed
    #     assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

    #     print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    # parser.add_argument(
    #     "--data_json_file",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Training data",
    # )
    # parser.add_argument(
    #     "--data_root_path",
    #     type=str,
    #     default="",
    #     required=True,
    #     help="Training data root path",
    # )
    # parser.add_argument(
    #     "--image_encoder_path",
    #     type=str,
    #     default=None,
    #     required=True,
    #     help="Path to CLIP image encoder",
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=20,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",# None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.


    #pipe = StableDiffusionPipeline.from_single_file('realisticVisionV60B1_v60B1VAE.safetensors', torch_dtype=torch.float32)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir='cache')

    #pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", cache_dir='cache')
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", vae=vae, controlnet=control_net_openpose, cache_dir='cache')


    #pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to('cuda')

    #noise_scheduler = pipe.scheduler
    noise_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",cache_dir='cache')
    #noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler", cache_dir='cache')#pipe.scheduler

    tokenizer = pipe.tokenizer

    vae = pipe.vae

    unet = pipe.unet

    image_encoder = pipe.image_encoder

    text_encoder = pipe.text_encoder

    pipe.safety_checker = None
    '''
    noise_scheduler = DDPMScheduler.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="scheduler", cache_dir='cache')
    # tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="text_encoder", cache_dir='cache')
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir='cache')
    unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet", cache_dir='cache')
    # image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # freeze parameters of models to save more memory
    '''
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    # image_encoder.requires_grad_(False)
    
    # write - 
    attn_procs = {}
    for name in unet.attn_processors.keys():
        if "attn1" in name:
            # not sure about type param
            attn_procs[name] = REFAttnProcessor(name=name, type='write')
        else:
            attn_procs[name] = AttnProcessor()
    unet.set_attn_processor(attn_procs)
    #adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    # read, copy of the unet to process
    # TODO

    # Freeze everything except the f keys()
    
    # cloth_adapter = ClothAdapter(unet, 'weights_7000.pt')
    cloth_adapter = ClothAdapter(unet, ref_path='weights_0p.pt')

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    # image_encoder.to(accelerator.device, dtype=weight_dtype)


    
    # optimizer
    #print(cloth_adapter.adapter_modules.parameters())

    p_to_opt = []
    for name, param in cloth_adapter.ref_unet.named_parameters():
        if name in garment_keys:
            p_to_opt.append(param)

    params_to_opt = itertools.chain(p_to_opt)
    print(f'lr: {args.learning_rate}, weight_decay: {args.weight_decay}')
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    #optimizer = bnb.optim.Adam(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.995), optim_bits=8, percentile_clipping=5)
    #optimizer = bnb.optim.AdamW8bit(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    # train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )


    train_dataset = AlignedDatasetHDF5()
    # train_dataset = AlignedDataset()
    opt = TrainOptions().parse()
    train_dataset.initialize(opt, mode='train', stage='gen')
    # opt = get_opt()
    # train_dataset = DressCodeDataset(opt)
    # train_dataloader = DressCodeDataLoader(opt, dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    cloth_adapter, optimizer, train_dataloader = accelerator.prepare(cloth_adapter, optimizer, train_dataloader)

    num_images_per_prompt = args.train_batch_size
    with torch.no_grad():
        prompt_embeds_null = pipe.encode_prompt([""], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
        encoder_hidden_states_upper = pipe.encode_prompt(["upper body"], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
        encoder_hidden_states_lower = pipe.encode_prompt(["lower body"], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
        encoder_hidden_states_dress = pipe.encode_prompt(["dress"], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]


        unet_encoder_hidden_states = pipe.encode_prompt(["a photography of a model"], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]


    loss_sum = 0.

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            store_vis = False
            if store_vis:
                print(batch['image'].shape)
                save_image(batch['image'], 'controlnet_tests/image.png')
                save_image(batch['cloth'], 'controlnet_tests/cloth.png')
                save_image(batch['cloth_mask'], 'controlnet_tests/cloth_mask.png')
                save_image(batch['pose'], 'controlnet_tests/pose.png')
                save_image(batch['agnostic_mask'], 'controlnet_tests/agnostic_mask.png')

            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(cloth_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    #img = prepare_image(batch["image"]).to(accelerator.device, dtype=weight_dtype)

                    img = batch["image"].to(accelerator.device, dtype=weight_dtype)

                    latents = vae.encode(img).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    agnostic_mask = batch["agnostic_mask"].to(accelerator.device, dtype=weight_dtype)
                    agnostic_mask = F.interpolate(agnostic_mask, (latents.shape[-2], latents.shape[-1]))


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                sqrt_alpha_prod, sqrt_one_minus_alpha_prod = scheduler_get_noise(noise_scheduler, latents, noise, timesteps)

                #noisy_latents = latents*(1-agnostic_mask) + noise*agnostic_mask

                with torch.no_grad():

                    '''
                    if batch["category"].item() == 0:
                        encoder_hidden_states = encoder_hidden_states_upper
                    elif batch["category"].item() == 1:
                        encoder_hidden_states = encoder_hidden_states_lower
                    elif batch["category"].item() == 2:
                        encoder_hidden_states = encoder_hidden_states_dress
                    else:
                        raise ValueError("Invalid category")
                    '''

                    prompt_embeds_adapter = prompt_embeds_null.clone()
                    for idx, cat in enumerate(batch['category']):
                        if cat.item() == 0:
                            prompt_embeds_adapter[idx] = encoder_hidden_states_upper[idx]
                        elif cat.item() == 1:
                            prompt_embeds_adapter[idx] = encoder_hidden_states_lower[idx]
                        elif cat.item() == 2:
                            prompt_embeds_adapter[idx] = encoder_hidden_states_dress[idx]
                        else:
                            print("error getting category from dataloader")


                    encoder_hidden_states = unet_encoder_hidden_states.clone()

                    # classifier-free dropout
                    for idx, d in enumerate(batch['drop_image_embed']):
                        if d.item() == 1:
                            batch['cloth'][idx] = torch.zeros_like(batch['cloth'][idx])

                    # classifier-free dropout
                    for idx, d in enumerate(batch['drop_prompt_embed']):
                        if d.item() == 1:
                            encoder_hidden_states[idx] = prompt_embeds_null[idx]

                    cloth = batch['cloth'] #prepare_image(batch['cloth']['unpaired'], height, width)
                    cloth_mask = batch['cloth_mask'] #prepare_mask(batch['cloth_mask']['unpaired'], height, width)
                    cloth = (cloth * cloth_mask).to(accelerator.device, dtype=weight_dtype)
                    cloth_embeds = vae.encode(cloth).latent_dist.mode() * vae.config.scaling_factor


                noise_pred = cloth_adapter(noisy_latents, batch['pose'], timesteps, encoder_hidden_states, prompt_embeds_adapter, cloth_embeds, num_images_per_prompt)

                #loss = F.mse_loss(noise_pred.float()*agnostic_mask, noise.float()*agnostic_mask, reduction="mean")*torch.sum(torch.ones_like(agnostic_mask))/torch.sum(agnostic_mask)

                loss = 0.1*F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") + F.mse_loss(noise_pred.float()*agnostic_mask, noise.float()*agnostic_mask, reduction="mean")*1/(torch.sum(agnostic_mask) + 0.01)

                # gradient accumulation
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                # Backpropagate
                accelerator.backward(loss)

                loss_sum += loss.item()

                if step % 8 == 0:    # gradient accumulation batch size
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_sum = 0.

                if accelerator.is_main_process:
                    print("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1
            
            if global_step % args.save_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
            
            begin = time.perf_counter()

            if step % 100 == 0:
                with torch.no_grad():
                    images = cloth_adapter.generate(pipe, weight_dtype,
                                                    cloth_keyword='lower body',
                                                    cloth_image='val/013564_1.jpg',
                                                    cloth_mask_image='val/013564_1.png',
                                                    openpose_image='val/013564_5.jpg')
                    images[0].save(f"train_{step}.jpg")

                    images = cloth_adapter.generate(pipe, weight_dtype,
                                                    cloth_keyword='dress',
                                                    cloth_image='val/020714_1.jpg',
                                                    cloth_mask_image='val/020714_1.png',
                                                    openpose_image='val/020714_5.jpg')
                    images[0].save(f"train_2_{step}.jpg")

                    images = cloth_adapter.generate(pipe, weight_dtype,
                                                    cloth_keyword='upper body',
                                                    cloth_image='val/000000_1.jpg',
                                                    cloth_mask_image='val/000000_1.png',
                                                    openpose_image='val/000000_5.jpg')
                    images[0].save(f"train_3_{step}.jpg")

                    img_train = batch["image"]
                    img_train = (img_train / 2 + 0.5).clamp(0, 1)
                    img_train = (img_train.cpu().permute(0, 2, 3, 1).float().numpy()*255).astype(np.uint8)[0]

                    Image.fromarray(img_train).save('train.png')

                    # check output - restore amount of noise, added by scheduler
                    #latents = noisy_latents + noise_pred
                    #noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

                    latents = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred)/(sqrt_alpha_prod + 0.001)

                    image_pred = (decode_latents(vae, latents.to(dtype=weight_dtype))*255).astype(np.uint8)[0]
                    Image.fromarray(image_pred, mode='RGB').save('prediction.jpg')

        #if step % 5000 == 0:
        cloth_adapter.save_checkpoint(name=f"weights_epoch{epoch}.pt")
                
if __name__ == "__main__":
    main()
