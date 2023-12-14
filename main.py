# %%
#
#
import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# %%
from typing import List

import nest_asyncio
from dotenv import load_dotenv
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
from llama_index.schema import ImageDocument
from llama_index.storage.storage_context import StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores import DeepLakeVectorStore
from pydantic import BaseModel

from api import generate_image, generate_image_description
from utils import get_product_image_path_for_gradio, show_product_in_notebook

# nest_asyncio.apply()

load_dotenv()  # take environment variables from .env
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPEN_WEATHER_MAP_KEY = os.environ["OPEN_WEATHER_MAP_KEY"]


# %%
class Clothing(BaseModel):
    """Data moel for clothing items"""

    name: str
    product_id: str
    price: float


class ClothingList(BaseModel):
    """A list of clothing items for the model to use"""

    cloths: List[Clothing]


# %%

dataset_path = "hub://kiedanski/walmart_clothing4"
vector_store = DeepLakeVectorStore(
    dataset_path=dataset_path, overwrite=False, read_only=True
)
# note: it takes some time to load

# %%
llm = OpenAI(model="gpt-4", temperature=0.7)
service_context = ServiceContext.from_defaults(llm=llm)
inventory_index = VectorStoreIndex.from_vector_store(
    vector_store, service_context=service_context
)

# %%
# Inventory query engine tool
#
inventory_query_engine = inventory_index.as_query_engine(output_cls=ClothingList)

# debug
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

# debug
if __name__ == "__main__":
    r = inventory_query_engine_tool("a red bennie hat")
    product_id = r.raw_output.cloths[0].product_id
    show_product_in_notebook(product_id)
    print(r)


# %%
# Query Rewriting Retriever Pack
#
from llama_index.llama_pack import download_llama_pack

# # download and install dependencies
# QueryRewritingRetrieverPack = download_llama_pack(
#     "QueryRewritingRetrieverPack", "./query_rewriting_pack"
# )

# # create the pack
# query_rewriting_pack_tool = QueryRewritingRetrieverPack(
#     index,
#     chunk_size=256,
#     vector_similarity_top_k=2,
# )

# # debug
# if __name__ == "__main__":
#     r = query_rewriting_pack_tool("a red bennie hat")
#     print(r)

# %%
# Outfit recommender tool
#


class Outfit(BaseModel):
    top: str = ""
    bottom: str = ""
    shoes: str = ""


# TODO: add input_image as a parameter to this function..
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
    image_documents = SimpleDirectoryReader("./input_image").load_data()

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


# debug
if __name__ == "__main__":
    outfit = generate_outfit_description("man", "I don't like wearing white")

from llama_index.tools import BaseTool, FunctionTool

outfit_description_tool = FunctionTool.from_defaults(fn=generate_outfit_description)

# %%
# Today's date
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
# Tool receive product image from the user
#
generate_image_description_tool = FunctionTool.from_defaults(
    fn=generate_image_description
)


# %%
# Tool to show product on the UI
#

load_product_image_tool = FunctionTool.from_defaults(
    fn=get_product_image_path_for_gradio
)


# %%
# Weather tool
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

# from llama_index import download_loader

# WeatherReader = download_loader("WeatherReader")

# loader = WeatherReader(token=OPEN_WEATHER_MAP_KEY)
# documents = loader.load_data(places=["Montevideo"])

# %%
# Tool to show image in notebook
#
# show_product_in_notebook_tool = FunctionTool.from_defaults(fn=show_product_in_notebook)

# %%
# Image generation tool
#
from llama_hub.tools.openai_image_generation import OpenAIImageGenerationToolSpec

# Note: tried it out and it doesn't work, multiple errors (e.g access to internal modules, missing metadata)
# image_generation_tool = OpenAIImageGenerationToolSpec(api_key=OPENAI_API_KEY)
image_generation_tool = FunctionTool.from_defaults(fn=generate_image)


# %%
# Agent
#
agent = OpenAIAgent.from_tools(
    system_prompt="""
    You are a specialized shopping assistant.

    You are tasked to recommend an outfit for an upcoming event based on the
    user's gender, style preferences, occasion type, weather conditions on event's date and location, etc.

    Don't ask all the questions at once, gather the required information step by step.

    Once you have the required information, your answer needs to be the outfit composed by the
    product_id with the best matching products in our inventory.

    Include the the total price of the recommended outfit.
    """,
    # Customers will provide you with a piece of clothing and occasion, and you will generate a matching outfit.
    # If it's an especial event, you can ask for the city to get the weather conditions.
    tools=[
        get_current_date_tool,
        *weather_tool_spec.to_tool_list(),
        inventory_query_engine_tool,
        outfit_description_tool,
        generate_image_description_tool,
    ],
    llm=llm,
    verbose=True,
)

# %%
#
#

if __name__ == "__main__":
    # %%
    r = agent.chat("What are your tools?")
    print(r)
    # %%
    r = agent.chat("What's the weather like in 3 days?")
    print(r)
    # %%
    r = agent.chat("I am in Montevideo, Uruguay")
    print(r)
    # %%
    r = agent.chat("Hi")
    print(r)
    # %%
    # r = agent.chat("I want an outfit for a casual birthday party")
    r = agent.chat("I want an outfit for a formal birthday party")
    print(r)

    # %%
    r = agent.chat("I'm a man")
    print(r)

    # %%
    # r = agent.chat("My budget is only 30, can you recommend an alternative?")
    # print(r)

    # %%
    r = agent.chat("Show me the product images in the notebook")
    print(r)
    # %%
    r = agent.chat(
        "Can you provide me that information in JSON format so I can parse it"
    )
    print(r)

# %%
