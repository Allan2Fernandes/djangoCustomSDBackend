
from rest_framework.response import Response
from rest_framework.views import APIView
import base64
from PIL import Image
import io
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# Load the pipeline
device = torch.device('cuda')
model_id = "stabilityai/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, revision='fp16', torch_dtype=torch.float16).to(device)
pipe.enable_xformers_memory_efficient_attention()

class ImageGeneratorView(APIView):
    progress_counter = 0

    def post(self, request):
        request = request.data
        # Parse the body
        prompts = request['prompts']
        num_inference_steps = int(request['inf_steps'])
        if num_inference_steps < 1:
            num_inference_steps = 1
        # Generate the images
        if request['with_seed']:
            seed = int(request['Seed'])
            images, seed = self.generateImages(prompts, seed=seed, num_inference_steps=num_inference_steps, with_seed=request['with_seed'])
        else:
            images, seed = self.generateImages(prompts, num_inference_steps=num_inference_steps, with_seed=request['with_seed'])

        # Create a list to append images to
        list_of_images = []
        num_generated_images = len(prompts)
        # #For every image in the tensor, create an image
        for i in range(num_generated_images):
            image_file = Image.fromarray(images[i])
            # Convert to byte format before you can encode to b64
            with io.BytesIO() as buffer:
                image_file.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            encoded_image = base64.b64encode(image_bytes)
            # Add every image to the list
            list_of_images.append(encoded_image)
            pass
        return Response({"ListImages": list_of_images, "Seed": seed})

    def get(self, request):
        return Response({"Progress": self.progress_counter})


    def generateImages(self, prompts, with_seed, guidance_scale=8, num_inference_steps=30, seed=42):
        # Set up a generator for reproducibility
        generator = torch.Generator(device=device).manual_seed(seed)
        negative_prompt = "zoomed in, blurry, oversaturated, warped"
        # Encode the prompt
        text_embeddings = pipe._encode_prompt(prompts, device, 1, True,
                                              negative_prompt=[negative_prompt] * len(prompts))
        # Create our random starting point
        if with_seed:
            latents = torch.randn((len(prompts), 4, 64, 64), device=device, generator=generator, requires_grad=False)
        else:
            latents = torch.randn((len(prompts), 4, 64, 64), device=device, requires_grad=False)
        latents *= pipe.scheduler.init_noise_sigma

        # Prepare the scheduler
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        # Loop through the sampling timesteps
        for i, t in enumerate(pipe.scheduler.timesteps):
            print(ImageGeneratorView.progress_counter)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)

            # Apply any scaling required by the scheduler
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            latent_model_input = latent_model_input.half()
            # predict the noise residual with the unet
            with torch.no_grad():
                noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # guidance with text
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # p_sample
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            ImageGeneratorView.progress_counter += 1
            pass
        ImageGeneratorView.progress_counter = 0
        # Decode back into image. From source code
        with torch.no_grad():
            latents = latents.half()
            latents = 1 / pipe.vae.config.scaling_factor * latents
            image = pipe.vae.decode(latents.detach()).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            pass
        images = image * 255
        images = images.astype(np.uint8)
        return images, seed
    pass

class ProgressView(APIView):
    def get(self, request):
        return Response({'progress': ImageGeneratorView.progress_counter})
