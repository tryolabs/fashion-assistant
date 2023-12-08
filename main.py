# %%
import os
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

from api import outfit_generation_tool
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document, ServiceContext
from pydantic import BaseModel
from typing import List

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

os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMjAzMzM3OCwiZXhwIjoxNzMzNjU1NzczfQ.eyJpZCI6ImtpZWRhbnNraSJ9.QBhyvCxFaL-m5nB4wnp5Us2mMpoTPvMWbSknAk7R6myUvBCM6l1yLb0i5T6RMc46cDhBr_qfyOMzHkExsnzyPQ"
dataset_path = "hub://kiedanski/walmart_clothing4"
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=False, read_only=True)

# %%
llm = OpenAI(model="gpt-4", temperature=0)
service_context = ServiceContext.from_defaults(llm=llm)
index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
query_engine = index.as_query_engine(output_cls=ClothingList)
# %%
r = query_engine.query("a red bennie hat")



# %%

inventory_item = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=(
                    "Useful for finding clothing items in our inventory"
                    "Usage: input: 'Give me the product_id of a product matching `product description`'"
                    "Alwasys ask the product_id of the product when using this tool"
                ),
            ),
        )

# %%
llm = OpenAI(model="gpt-4")
agent = OpenAIAgent.from_tools(
    [outfit_generation_tool, inventory_item], llm=llm, verbose=True,
    system_prompt="""
    You are a specialized shopping assistant.
    Customers will provide you with a piece of clothing, and you will generate a matching outfit.
    Your final answer needs to be the product_id associated with the best matching product in our inventory.
    For each product of the outfit, search the inventory.
    Incldue the total price of the recommended outfit.
    """
)
# %%
r = agent.chat("Hi")
print(r)
# %%
r = agent.chat("I want an outfit for a casual birthday party")
print(r)

# %%
r = agent.chat("I'm a woman ")
print(r)

# %%
# r = agent.chat("My budget is only 30, can you recommend an alternative?")
# print(r)
