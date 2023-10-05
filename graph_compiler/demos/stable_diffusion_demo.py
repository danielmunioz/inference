import sys
sys.path.append('..')

import inspect
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler

import ivy
import graph_compiler as gc
import torch

# increase image size for a better result (at the risk of memory issues)
# (seems like it needs to be a multiple of 64)
IMG_SIZE = 256
NUM_STEPS = 50


# Load Hugging face model #
# ----------------------- #

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)


# The following two functions text_to_embeddings and embeddings_to_image are just attained by
# splitting the function: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L112#L317
# into 2 parts, since the graph compiler won't connect to str inputs so we instead only compile
# the embeddings_to_image which has tensor inputs and outputs (compatible with our graph compiler)


# get prompt text embeddings
@torch.no_grad()
def text_to_embeddings(
    prompt,
    height=IMG_SIZE,
    width=IMG_SIZE,
    guidance_scale=7.5,
    num_inference_steps=NUM_STEPS,
):
    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        raise ValueError(
            f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
        )
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(
            f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
        )

    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipe.text_encoder(text_input.input_ids)[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = pipe.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids)[0]

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings


@torch.no_grad()
def embeddings_to_image(
    text_embeddings,
    guidance_scale=7.5,
    num_inference_steps=NUM_STEPS,
    height=IMG_SIZE,
    width=IMG_SIZE,
    eta=0.0,
    generator=None,
    latents=None,
):
    batch_size = 1  # since prompt is a string

    # get the initial random noise unless the user supplied it

    # Unlike in other pipelines, latents need to be generated in the target device
    # for 1-to-1 results reproducibility with the CompVis implementation.
    # However this currently doesn't work in `mps`.
    latents_device = "cpu" if pipe.device.type == "mps" else pipe.device
    latents_shape = (batch_size, pipe.unet.in_channels, height // 8, width // 8)
    if latents is None:
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=latents_device,
        )
    else:
        if latents.shape != latents_shape:
            raise ValueError(
                f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}"
            )
    latents = latents.to(pipe.device)

    pipe.scheduler.set_timesteps(num_inference_steps)

    # if we use LMSDiscreteScheduler, let's make sure latents are multiplied by sigmas
    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        latents = latents * pipe.scheduler.sigmas[0]

    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]
    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    for i, t in enumerate(pipe.progress_bar(pipe.scheduler.timesteps)):
        do_classifier_free_guidance = guidance_scale > 1.0
        # expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        )
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            sigma = pipe.scheduler.sigmas[i]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # compute the previous noisy sample x_t -> x_t-1
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            latents = pipe.scheduler.step(
                noise_pred, i, latents, **extra_step_kwargs
            ).prev_sample
        else:
            latents = pipe.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1)  # .numpy()

    # run safety checker
    # safety_checker_input = pipe.feature_extractor(pipe.numpy_to_pil(image), return_tensors="pt").to(pipe.device)
    # image, _ = pipe.safety_checker(
    #     images=image, clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype)
    # )
    return image


ivy.set_backend("torch")
prompt = "border collie playing fetch"

embeddings = text_to_embeddings(prompt)
latents_shape = (1, pipe.unet.in_channels, IMG_SIZE // 8, IMG_SIZE // 8)
latents = torch.randn(latents_shape)
compiled_fn = gc.compile(embeddings_to_image, embeddings, latents=latents)
no_grad_compiled = torch.no_grad()(compiled_fn)

image = embeddings_to_image(embeddings, latents=latents)
image = image.numpy()
image = pipe.numpy_to_pil(image)[0]
image.save("doggy.png")

image = no_grad_compiled(embeddings, latents=latents)
image = image.numpy()
image = pipe.numpy_to_pil(image)[0]
image.save("doggy_compiled.png")
