import numpy as np
import torch
from diffusers import StableDiffusionPipeline


class ImageGenerator:
    def __init__(self):
        # Load the pipeline
        self.device = torch.device('cuda')
        model_id = "stabilityai/stable-diffusion-2-1-base"
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, revision='fp16', torch_dtype=torch.float16).to(self.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        pass

    def generateImages(self, prompts, guidance_scale=8, num_inference_steps=30, seed=42):
        # Set up a generator for reproducibility
        generator = torch.Generator(device=self.device).manual_seed(seed)
        negative_prompt = "zoomed in, blurry, oversaturated, warped"
        # Encode the prompt
        text_embeddings = self.pipe._encode_prompt(prompts, self.device, 1, True,
                                              negative_prompt=[negative_prompt] * len(prompts))
        # Create our random starting point
        latents = torch.randn((len(prompts), 4, 64, 64), device=self.device, generator=generator)
        latents *= self.pipe.scheduler.init_noise_sigma

        # Prepare the scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        # Loop through the sampling timesteps
        for i, t in enumerate(self.pipe.scheduler.timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Apply any scaling required by the scheduler
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.half()
            # predict the noise residual with the unet
            with torch.no_grad():
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # guidance with text
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # p_sample
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            # print(i)
            pass
        # Decode back into image. From source code
        with torch.no_grad():
            latents = latents.half()
            latents = 1 / self.pipe.vae.config.scaling_factor * latents
            image = self.pipe.vae.decode(latents.detach()).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            pass
        # single_image = image[-1]
        # single_image = single_image*255
        # single_image = single_image.astype(np.uint8)
        images = image * 255
        images = images.astype(np.uint8)
        # image_file = Image.fromarray(single_image)
        # image_file.save("GeneratedImage.jpeg")
        return images