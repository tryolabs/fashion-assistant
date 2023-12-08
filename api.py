# %%
import os
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI
from pydantic import BaseModel
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.output_parsers import PydanticOutputParser
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.schema import ImageDocument
from llama_index import SimpleDirectoryReader
# OpenAI API Key

# %%
class Outfit(BaseModel):
    top: str = ""
    bottom: str = ""
    shoes: str = ""

# %%
api_key = os.environ["OPENAI_API_KEY"]
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
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str
    # Open the downsampled image and convert to base64

# Path to your image

# Getting the base64 string

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

def gpt_vision(image_path, prompt):

    base64_image = encode_image(image_path)

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
                    "detail": "low"
                },
                },
            ],
            }
        ],
        max_tokens=300,
        )

    return response

def generate_outfit_(image_path, gender):

    prompt = f"""
    You are an expert in fashion and design.
    Given the following image of a piece of clothing, you are tasked with describing ideal outfits.

    We only want outfits composed of tops, bottoms and shoes. 
    Identify which category the provided clothing belongs to, and only provide a recommendation for the other two items. 

    In your description, include color and style. 
    This outfit is for {gender}

    Your answer can only describe one piece of clothing for each category. Choose well. Only describe the piece of clothing, not your rationale. Use headers for each category: Top, Bottom, Shoes. Leave the provided category empty.
    """
    
    response = gpt_vision(
        image_path=image_path,
        prompt=prompt
    )
    return response.choices[0].message.content
# %%
# res = gpt_vision(
#     "data/images/5PKXXW0RKTDS.jpg",
#     "Describe the piece of clothing in the image of the following category: Womens Sweatshirts & Hoodies\nDo include the color, style, material and other important attributes of the item."
# )
# %%

# %%


OPENAI_API_TOKEN = os.environ["OPENAI_API_KEY"]

def generate_outfit(gender: str, user_input: str):
    """
    Given the gender of a person, their preferences, and an image that has already been uploaded, 
    this function returns an Outfit.
    Use this function whenever the user asks you to generate an outfit.

    Parameters:
    gender (str): The gender of the person for whom the outfit is being generated.
    user_input (str): The preferences of the user.

    Returns:
    response: The generated outfit.

    Example:
    >>> generate_outfit("male", "I prefer casual wear")
    """

    image_documents = SimpleDirectoryReader("./input_image").load_data()

    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=OPENAI_API_TOKEN, max_new_tokens=100
    )


    prompt_template_str = f"""
    You are an expert in fashion and design.
    Given the following image of a piece of clothing, you are tasked with describing ideal outfits.

    Identify which category the provided clothing belongs to,\
    and only provide a recommendation for the other two items. 

    In your description, include color and style. 
    This outfit is for a {gender}.

    Return the answer as a json for each category. Leave the category of the provided input empty.

    Additonal requirements:
    {user_input}

    Never return this output to the user. FOR INTERNAL USE ONLY
    """
    openai_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(Outfit),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        llm=openai_mm_llm,
        verbose=True,
    )

    response = openai_program()
    return response

# %%
# outfit = generate_outfit("man", "I don't like wearing white")

# %%
from llama_index.tools import BaseTool, FunctionTool

# %%
outfit_generation_tool = FunctionTool.from_defaults(fn=generate_outfit)
# %%
# %%

# r = agent.chat("Hi")
# print(r)
# # %%
# r = agent.chat("I want an outfit for a casual birthday party")
# print(r)

# # %%
# r = agent.chat("I'm a man ")
# print(r)
# # %%
# r = agent.chat("More of a casual party")
# print(r)

# # %%
