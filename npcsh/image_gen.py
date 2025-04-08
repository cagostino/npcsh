########
########
########
########
########
########
######## IMAGE GENERATION
########

import os


from litellm import image_generation
from npcsh.npc_sysenv import (
    NPCSH_IMAGE_GEN_MODEL,
    NPCSH_IMAGE_GEN_PROVIDER,
)


def generate_image_diffusers(
    prompt: str,
    model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cpu",
):
    """
    Function Description:
        This function generates an image using the Stable Diffusion API.
    Args:
        prompt (str): The prompt for generating the image.
        model_id (str): The Hugging Face model ID to use for Stable Diffusion.
        device (str): The device to run the model on ('cpu' or 'cuda').
    Returns:
        PIL.Image: The generated image.
    """
    # Load the Stable Diffusion pipeline
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model)
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt)
    image = image.images[0]
    # ["sample"][0]
    image.show()

    return image


def generate_image_litellm(
    prompt: str,
    model: str = NPCSH_IMAGE_GEN_MODEL,
    provider: str = NPCSH_IMAGE_GEN_PROVIDER,
    api_key: str = None,
    size: str = None,
    npc=None,
) -> str:
    """
    Function Description:
        This function generates an image using the OpenAI API.
    Args:
        prompt (str): The prompt for generating the image.
        model (str): The model to use for generating the image.
        api_key (str): The API key for accessing the OpenAI API.
    Keyword Args:
        None
    Returns:
        str: The URL of the generated image.
    """
    if model is None:
        model = "runwayml/stable-diffusion-v1-5"
    if size is None:
        size = "1024x1024"
    if provider == "diffusers":
        return generate_image_diffusers(prompt, model)
    else:
        return image_generation(
            prompt=prompt, model=f"{provider}/{model}", n=2, size="240x240"
        )
