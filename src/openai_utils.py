# %%
import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

load_dotenv()

# %%
client = OpenAI()


# Function to encode the image
def encode_image(image_path):
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height
    new_width = 300
    new_height = int(new_width / aspect_ratio)
    img = img.resize((new_width, new_height))

    buffered = BytesIO()
    img.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return img_str
    # Open the downsampled image and convert to base64


# %%
# Describe image
#
def gpt_vision(image_path: str, prompt: str):
    base64_image = encode_image(image_path)

    # Call GPT4V using OpenAI API
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    return response


def generate_image_description(image_path: str):
    """Describes the piece of clothing on the image"""
    prompt = """
    Describe the piece of clothing in the image of the following category: Womens Sweatshirts & Hoodies
    Do include the color, style, material and other important attributes of the item.
    """
    return gpt_vision(image_path, prompt)


# %%
# Generate image
#
from openai import OpenAI

client = OpenAI()


def generate_image(prompt):
    """Generate images based on the prompt description"""
    # Call Dalle3 using OpenAI API
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return response


if __name__ == "__main__":
    response_img = generate_image("outfit with a blue top and a black jeans")
