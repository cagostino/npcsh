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
    response = image_generation(
        prompt=prompt, model=f"{provider}/{model}", n=2, size="240x240"
    )
    return response
