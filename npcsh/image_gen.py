########
########
########
########
########
########
######## IMAGE GENERATION
########

import os

from openai import OpenAI
from diffusers import StableDiffusionPipeline


def generate_image_openai(
    prompt: str,
    model: str,
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
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if model is None:
        model = "dall-e-2"
    client = OpenAI(api_key=api_key)
    if size is None:
        size = "1024x1024"
    if model not in ["dall-e-3", "dall-e-2"]:
        # raise ValueError(f"Invalid model: {model}")
        print(f"Invalid model: {model}")
        print("Switching to dall-e-3")
        model = "dall-e-3"
    image = client.images.generate(model=model, prompt=prompt, n=1, size=size)
    if image is not None:
        # print(image)
        return image


def generate_image_hf_diffusion(
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
    pipe = StableDiffusionPipeline.from_pretrained(model)
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt)
    image = image.images[0]
    # ["sample"][0]
    image.show()

    return image
