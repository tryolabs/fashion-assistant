# %%
# Set up
#
import logging
import os
import sys

from dotenv import load_dotenv

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=LOG_FORMAT)
logging.basicConfig(filename="debug.log", level=logging.DEBUG, format=LOG_FORMAT)

# Load environment variables from .env
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPEN_WEATHER_MAP_KEY = os.environ["OPEN_WEATHER_MAP_KEY"]
ACTIVELOOP_DATASET_TEXT = os.getenv("ACTIVELOOP_DATASET_TEXT")
ACTIVELOOP_DATASET_IMG = os.getenv("ACTIVELOOP_DATASET_IMG")

INPUT_IMAGE_DIR = "./input_image"
os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)

# %%
# Imports
#
from typing import List

from llama_hub.tools.weather import OpenWeatherMapToolSpec
from llama_index import (
    Document,
    ServiceContext,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.multi_modal_llms import OpenAIMultiModal
from llama_index.output_parsers import PydanticOutputParser
from llama_index.program import MultiModalLLMCompletionProgram
from llama_index.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.vector_stores import DeepLakeVectorStore
from pydantic import BaseModel

from src.openai_utils import generate_image_description
from src.utils import show_product_in_notebook


# %%
# Output models
#
class Clothing(BaseModel):
    """Data moel for clothing items"""

    name: str
    product_id: str
    price: float


class ClothingList(BaseModel):
    """A list of clothing items for the model to use"""

    cloths: List[Clothing]


class Outfit(BaseModel):
    top: str = ""
    bottom: str = ""
    shoes: str = ""


# %%
# VectorStore connected to our DeepLake dataset
#
vector_store = DeepLakeVectorStore(
    dataset_path=ACTIVELOOP_DATASET_TEXT, overwrite=False, read_only=True
)


def clean_input_image():
    if len(os.listdir(INPUT_IMAGE_DIR)) > 0:
        for file in os.listdir(INPUT_IMAGE_DIR):
            os.remove(os.path.join(INPUT_IMAGE_DIR, file))

def has_user_input_image():
    """
Check if the INPUT_IMAGE_DIR directory contains exactly one image.
Useful for checking if there is an image before generating an outfit.

Returns:
    bool: True if INPUT_IMAGE_DIR contains exactly one image, False otherwise.
    """
    return len(os.listdir(INPUT_IMAGE_DIR)) == 1

    
check_input_image_tool = FunctionTool.from_defaults(fn=has_user_input_image)
       

# %%
# LLM
#
llm = OpenAI(model="gpt-4", temperature=0.7)

# %%
# Inventory query engine tool
#
service_context = ServiceContext.from_defaults(llm=llm)
inventory_index = VectorStoreIndex.from_vector_store(
    vector_store, service_context=service_context
)
inventory_query_engine = inventory_index.as_query_engine(output_cls=ClothingList)

# for debugging purposes
if __name__ == "__main__":
    r = inventory_query_engine.query("a red bennie hat")
    product_id = r.response.cloths[0].product_id
    show_product_in_notebook(product_id)
    print(r)


inventory_query_engine_tool = QueryEngineTool(
    query_engine=inventory_query_engine,
    metadata=ToolMetadata(
        name="inventory_query_engine_tool",
        description=(
            "Useful for finding clothing items in our inventory"
            "Usage: input: 'Give me the product_id of a product matching `product description`'"
            "Always ask the product_id of the product when using this tool"
        ),
    ),
)

# for debugging purposes
if __name__ == "__main__":
    r = inventory_query_engine_tool("a red bennie hat")
    product_id = r.raw_output.cloths[0].product_id
    show_product_in_notebook(product_id)
    print(r)


# %%
# Outfit recommender tool
#
# TODO: add input_image as a parameter to this function, pass image path to the uploaded image.
def generate_outfit_description(gender: str, user_input: str):
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

    # Load input image
    image_documents = SimpleDirectoryReader(INPUT_IMAGE_DIR).load_data()

    # Define multi-modal llm
    openai_mm_llm = OpenAIMultiModal(model="gpt-4-vision-preview", max_new_tokens=100)

    # Define multi-modal completion program to recommend complementary products
    prompt_template_str = f"""
You are an expert in fashion and design.
Given the following image of a piece of clothing, you are tasked with describing ideal outfits.

Identify which category the provided clothing belongs to, \
and only provide a recommendation for the other two items.

In your description, include color and style.
This outfit is for a {gender}.

Return the answer as a json for each category. Leave the category of the provided input empty.

Additional requirements:
{user_input}

Never return this output to the user. FOR INTERNAL USE ONLY
    """
    recommender_completion_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(Outfit),
        image_documents=image_documents,
        prompt_template_str=prompt_template_str,
        llm=openai_mm_llm,
        verbose=True,
    )

    # Run recommender program
    response = recommender_completion_program()

    return response


# for debugging purposes
if __name__ == "__main__":
    outfit = generate_outfit_description("man", "I don't like wearing white")


outfit_description_tool = FunctionTool.from_defaults(fn=generate_outfit_description)

# %%
# Tool to get current date
#
from datetime import date


def get_current_date():
    """
    A function to return todays date.

    Call this before any other functions if you are unaware of the current date.
    """
    return date.today()


get_current_date_tool = FunctionTool.from_defaults(fn=get_current_date)


# %%
# Tool to describe product image
#
generate_image_description_tool = FunctionTool.from_defaults(
    fn=generate_image_description
)


# %%
# Tool to get weather conditions
#
class CustomOpenWeatherMapToolSpec(OpenWeatherMapToolSpec):
    spec_functions = ["weather_at_location", "forecast_at_location"]

    def __init__(self, key: str, temp_units: str = "celsius") -> None:
        super().__init__(key, temp_units)

    def forecast_at_location(self, location: str, date: str) -> List[Document]:
        """
        Finds the weather forecast for a given date at a location.

        The forecast goes from today until 5 days ahead.

        Args:
            location (str):
                The location to find the weather at.
                Should be a city name and country.
            date (str):
                The desired date to get the weather for.
        """
        from pyowm.commons.exceptions import NotFoundError
        from pyowm.utils import timestamps

        try:
            forecast = self._mgr.forecast_at_place(location, "3h")
        except NotFoundError:
            return [Document(text=f"Unable to find weather at {location}.")]

        w = forecast.get_weather_at(date)

        temperature = w.temperature(self.temp_units)
        temp_unit = "°C" if self.temp_units == "celsius" else "°F"

        # TODO: this isn't working.. Error: 'max' key.
        try:
            temp_str = self._format_forecast_temp(temperature, temp_unit)
        except:
            logging.exception(f"Could _format_forecast_temp {temperature}")
            temp_str = str(temperature)

        try:
            weather_text = self._format_weather(location, temp_str, w)
        except:
            logging.exception(f"Could _format_weather {w}")
            weather_text = str(w) + " " + str(temp_str)

        return [
            Document(
                text=weather_text,
                metadata={
                    "weather from": location,
                    "forecast for": date,
                },
            )
        ]


weather_tool_spec = CustomOpenWeatherMapToolSpec(key=OPEN_WEATHER_MAP_KEY)

# %%
# Agent
#
agent = OpenAIAgent.from_tools(
    system_prompt="""
You are a specialized shopping assistant.

You are tasked to recommend an outfit for an upcoming event based on the
user's gender, style preferences, occasion type, weather conditions on event's date and location, etc.

Don't ask all the questions at once, gather the required information step by step.

Always check if the user has uploaded an image. If it has not, wait until they do. Never proceed without an image.

Once you have the required information, your answer needs to be the outfit composed by the
product_id with the best matching products in our inventory.

Include the the total price of the recommended outfit.
    """,
    tools=[
        get_current_date_tool,
        *weather_tool_spec.to_tool_list(),
        inventory_query_engine_tool,
        outfit_description_tool,
        check_input_image_tool,
    ],
    llm=llm,
    verbose=True,
)

# %%
# Sample messages to test the Agent
#
if __name__ == "__main__":
    # %%
    r = agent.chat("Hi")
    print(r)
    # %%
    r = agent.chat("What are your tools?")
    print(r)
    # %%
    r = agent.chat("I want an outfit for a casual birthday party")
    print(r)
    # %%
    r = agent.chat("I'm a man")
    print(r)
    # %%
    r = agent.chat("My budget is only 30, can you recommend an alternative?")
    print(r)
# %%
