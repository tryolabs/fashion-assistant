# %%

import os
import urllib.request
import math
import json
import pandas as pd
from api import gpt_vision
from tqdm import tqdm

from llama_index import Document
from llama_index import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.vector_stores import DeepLakeVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import KeywordNodePostprocessor
# %%
df_raw = pd.read_json("data/walmart.json")
# %%
df_raw.head()
# %%

df = pd.DataFrame({
    'brand': df_raw["brand"],
    'category': df_raw["category"].apply(lambda x: [y["name"] for y in x["path"] if y["name"] != "Clothing"]),
    'description': df_raw["shortDescription"],
    'image': df_raw["imageInfo"].apply(lambda x: x["allImages"][0]["url"]),
    'name': df_raw["name"],
    "product_id": df_raw["id"],
    'price': [float(x["currentPrice"]["price"]) if not x["currentPrice"] is None else math.inf for x in df_raw["priceInfo"]]
})
df = df[df["category"].transform(lambda x: len(x)) >= 2]

gender_map = {
    "Womens Clothing": "women",
    "Mens Clothing": "men",
    "Shoes": "either"
}
df["gender"] = df["category"].apply(lambda x: gender_map.get(x[0], "either"))
# %%
df.head()

# %%
with open("data/descriptions.json", "r") as fh:
    descriptions = json.load(fh)

# %%

# Create images directory if it doesn't exist
if not os.path.exists('data/images'):
    os.makedirs('data/images')

# Download images
for index, row in tqdm(df.iterrows()):
    image_url = row['image']
    product_id = row["product_id"]
    category = row["category"][1]

    image_name = os.path.join('data/images', product_id + '.jpg')
    if not os.path.exists(image_name):
        urllib.request.urlretrieve(image_url, image_name)

    if product_id not in descriptions:
        prompt = f"""
        Describe the piece of clothing in the image of the following category: {category}
        Do include the color, style, material and other important attributes of the item.
        """
        image_path = f"data/images/{product_id}.jpg"

        try:
            result = gpt_vision(image_path, prompt)
            message = result.choices[0].message.content
        except Exception as e:
            print(e)
            message = None
        
        descriptions[row["product_id"]] = message


# %%
# with open("data/descriptions.json", "w") as fh:
#     json.dump(descriptions, fh)

    
# %%
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTcwMjAzMzM3OCwiZXhwIjoxNzMzNjU1NzczfQ.eyJpZCI6ImtpZWRhbnNraSJ9.QBhyvCxFaL-m5nB4wnp5Us2mMpoTPvMWbSknAk7R6myUvBCM6l1yLb0i5T6RMc46cDhBr_qfyOMzHkExsnzyPQ"


# %%

# %%
dataset_path = "hub://kiedanski/walmart_clothing4"
vector_store = DeepLakeVectorStore(dataset_path=dataset_path, overwrite=True)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# %%

# %%
documents = []
for i, row in df.iterrows():
    product_id = row["product_id"] 
    description = descriptions[product_id]
    name = row["name"]
    gender = row["gender"]
    price = row["price"]

    desc = f"""
    # Description
    {description}

    # Name
    {name}

    # Product ID
    {product_id}

    # Price
    {price}
    
    """
    if all([product_id, description, name, gender]):
        doc = Document(
            text = desc,
            metadata = {
                "name": name,
                "product_id": product_id,
                "gender": gender
            }
        )
        documents.append(doc)
    else:
        print(row)
# %%
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
# %%

# query_engine = index.as_query_engine()
# response = query_engine.query(
#     "Hat"
# )
# # %%
# retriever = VectorIndexRetriever(
#     index=index,
#     similarity_top_k=5,
# )
# # %%
# node_postprocessors = [
#     KeywordNodePostprocessor(
#         exclude_keywords=["either"]
#     )
# ]
# query_engine = RetrieverQueryEngine.from_args(
#     retriever, node_postprocessors=node_postprocessors
# )
# # %%
# response = query_engine.query("Hat")

# # %%
# from llama_index.vector_stores.types import MetadataFilters, ExactMatchFilter

# filters = MetadataFilters(filters=[ExactMatchFilter(key="gender", value="men")])

# query_engine = index.as_query_engine(filters=filters)
# # %%
# response = query_engine.query("Hat")

# # %%
