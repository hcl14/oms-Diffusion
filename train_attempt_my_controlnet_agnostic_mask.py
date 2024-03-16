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

from viton_dataset import get_opt, VITONDataset, VITONDataLoader

from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline

import bitsandbytes as bnb

import copy
from safetensors import safe_open

# get list of keys:

orig_model_path='checkpoints/oms_diffusion_100000.safetensors'

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

def collate_fn(data):
    images = torch.stack([example["img"] for example in data])
    #text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.stack([example["cloth"]['unpaired'] for example in data])
    mask_images = torch.stack([example["cloth_mask"]['unpaired'] for example in data])
    drop_image_embeds = torch.stack([example["drop_image_embed"] for example in data])
    drop_prompt_embeds = torch.stack([example["drop_prompt_embed"] for example in data])
    pose_images = torch.stack([example["pose"] for example in data])
    mask_agnostic = torch.stack([example["agnostic_mask"]['unpaired'] for example in data])

    return {
        "img": images,
        #"text_input_ids": text_input_ids,
        "cloth": clip_images,
        "cloth_mask": mask_images,
        "drop_image_embed": drop_image_embeds,
        "drop_prompt_embed": drop_image_embeds,
        "pose": pose_images,
        "agnostic_mask": mask_agnostic
    }
    

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

        self.controlnet = ControlNetModel.from_single_file("/workspace/ControlNet-v1-1/control_v11p_sd15_openpose.pth",      torch_dtype=torch.float32)

        # freeze controlnet
        # freeze unet
        for name, param in self.controlnet.named_parameters():
            #if param.requires_grad is True:
            param.requires_grad = False



        # load weights if they exist
        state_dict = {}
        if os.path.exists(self.ref_path):
            with safe_open(self.ref_path, framework="pt", device="cpu") as f:
                for key in garment_keys:
                    state_dict[key] = f.get_tensor(key)
            ref_unet.load_state_dict(state_dict, strict=False)
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


    def forward(self, noisy_latents, controlnet_image, timesteps, encoder_hidden_states, prompt_embeds_null, cloth_embeds, num_images_per_prompt):
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
        self.ref_unet(cloth_embeds, 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

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
            cloth_image='data/zalando/train/cloth/14114_00.jpg',
            cloth_mask_image='data/zalando/train/cloth-mask/14114_00.jpg',
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

        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds_null = pipe.encode_prompt(["upper body"], device=self.device, num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]
            cloth_embeds = pipe.vae.encode(cloth).latent_dist.mode() * pipe.vae.config.scaling_factor
            cloth_embeds = cloth_embeds.to(dtype=torch.float32)
            prompt_embeds_null = prompt_embeds_null.to(dtype=torch.float32)

            self.ref_unet(torch.cat([cloth_embeds] * num_images_per_prompt), 0, prompt_embeds_null, cross_attention_kwargs={"attn_store": self.attn_store})

            for k, v in self.attn_store.items():
                v = v.to(dtype=torch.float32)

            # we have something wrong with original unet after training
            unet = pipe.unet
            pipe.unet = copy.deepcopy(pipe.unet)

            pipe.unet.load_state_dict(torch.load('infer_unet.pt'), strict=False)


            pipe.unet.to('cuda', dtype=torch.float32)
            pipe.vae.to(dtype=torch.float32)

            self.set_adapter(pipe.unet, "write")

            generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
            images = pipe(
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

            del(pipe.unet)
            pipe.vae.to(dtype=weight_dtype)
            pipe.unet = unet


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
        default=1e-5,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
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
        default="fp16",# None,
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

    pipe = StableDiffusionPipeline.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", cache_dir='cache')
    pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", cache_dir='cache')

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to('cuda')

    #noise_scheduler = pipe.scheduler
    noise_scheduler = DDPMScheduler.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="scheduler", cache_dir='cache')#pipe.scheduler

    tokenizer = pipe.tokenizer

    vae = pipe.vae

    unet = pipe.unet

    image_encoder = pipe.image_encoder

    text_encoder = pipe.text_encoder
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
    
    cloth_adapter = ClothAdapter(unet, 'weights.pt')

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
    #optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    #optimizer = bnb.optim.Adam(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.995), optim_bits=8, percentile_clipping=5)
    optimizer = bnb.optim.AdamW8bit(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    # train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution, image_root_path=args.data_root_path)
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    opt = get_opt()
    train_dataset = VITONDataset(opt)
    # train_dataloader = VITONDataLoader(opt, dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # ['13127_00.jpg'] - ['img_name']
    # ['03109_00.jpg'] - ['c_name']['unpaired']
    # torch.Size([1, 3, 1024, 768]) - ['img'].shape
    # torch.Size([1, 3, 1024, 768]) - ['img_agnostic'].shape
    # torch.Size([1, 13, 1024, 768]) - ['parse_agnostic'].shape
    # torch.Size([1, 3, 1024, 768]) - ['pose'].shape
    # torch.Size([1, 3, 1024, 768]) - ['cloth']['unpaired'].shape
    # torch.Size([1, 1, 1024, 768]) - ['cloth_mask']['unpaired'].shape

    # Prepare everything with our `accelerator`.
    cloth_adapter, optimizer, train_dataloader = accelerator.prepare(cloth_adapter, optimizer, train_dataloader)
    
    loss_sum = 0.

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(cloth_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    #img = prepare_image(batch["img"]).to(accelerator.device, dtype=weight_dtype)

                    img = batch["img"].to(accelerator.device, dtype=weight_dtype)

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

                noisy_latents = latents*(1-agnostic_mask) + noisy_latents*agnostic_mask


                with torch.no_grad():
                    # encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]
                    # num_images_per_prompt=4
                    num_images_per_prompt = len(img)

                    prompt_embeds_null = pipe.encode_prompt(["upper body"], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]

                    encoder_hidden_states = pipe.encode_prompt(["a photography of a model"], device='cuda', num_images_per_prompt=num_images_per_prompt, do_classifier_free_guidance=False)[0]

                    #print(batch['img'].shape, batch['cloth'].shape, batch['cloth_mask'].shape)

                    #save_img(batch['cloth'], 'cloth.png')
                    #save_img(batch['cloth_mask'], 'cloth_mask.png')
                    #save_img(batch['img'], 'img.png')

                    '''
                    # classifier-free dropout
                    for idx, d in enumerate(batch['drop_image_embed']):
                        if d.item() == 1:
                            batch['cloth'][idx] = torch.zeros_like(batch['cloth'][idx])

                    # classifier-free dropout
                    for idx, d in enumerate(batch['drop_prompt_embed']):
                        if d.item() == 1:
                            encoder_hidden_states[idx] = prompt_embeds_null[idx]
                    '''


                    cloth = batch['cloth'] #prepare_image(batch['cloth']['unpaired'], height, width)
                    cloth_mask = batch['cloth_mask'] #prepare_mask(batch['cloth_mask']['unpaired'], height, width)
                    cloth = (cloth * cloth_mask).to(accelerator.device, dtype=weight_dtype)
                    cloth_embeds = vae.encode(cloth).latent_dist.mode() * vae.config.scaling_factor


                noise_pred = cloth_adapter(noisy_latents, batch['pose'], timesteps, encoder_hidden_states, prompt_embeds_null, cloth_embeds, num_images_per_prompt)

        
                #loss = F.mse_loss(noise_pred.float()*agnostic_mask, noise.float()*agnostic_mask, reduction="mean")*torch.sum(torch.ones_like(agnostic_mask))/torch.sum(agnostic_mask)

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

            
                # gradient accumulation
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()



                # Backpropagate
                accelerator.backward(loss)

                loss_sum += loss.item()

                if step % 16 == 0:    # gradient accumulation batch size
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
                    images = cloth_adapter.generate(pipe, weight_dtype)
                    images[0].save(f"train_{step}.jpg")

                    img_train = batch["img"]
                    img_train = (img_train / 2 + 0.5).clamp(0, 1)
                    img_train = (img_train.cpu().permute(0, 2, 3, 1).float().numpy()*255).astype(np.uint8)[0]

                    Image.fromarray(img_train).save('train.png')

                    # check output
                    latents = noisy_latents + noise_pred
                    image_pred = (decode_latents(vae, latents.to(dtype=weight_dtype))*255).astype(np.uint8)[0]
                    Image.fromarray(image_pred, mode='RGB').save('prediction.jpg')


            if step % 1000 == 0:
                cloth_adapter.save_checkpoint(name=f"weights_{step}.pt")
                
if __name__ == "__main__":
    main()
