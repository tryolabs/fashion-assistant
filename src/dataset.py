# %%

import json
import math
import os
import urllib.request
import deeplake

import pandas as pd
from llama_index import Document, SimpleDirectoryReader, VectorStoreIndex
from llama_index.indices.postprocessor import KeywordNodePostprocessor
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import DeepLakeVectorStore
from tqdm import tqdm

from dotenv import load_dotenv

from src.openai_utils import gpt_vision

load_dotenv()

IMAGE_DIR = os.path.abspath("data/images")
ACTIVELOOP_DATASET_TEXT = os.getenv("ACTIVELOOP_DATASET_TEXT")
ACTIVELOOP_DATASET_IMG = os.getenv("ACTIVELOOP_DATASET_IMG")

# %%
# Run dataset creation

if __name__ == "__main__":
    # %%
    # TODO: merge items into a single json file
    dfs = []
    for i in range(1, 4):
        df_ = pd.read_json(f"data/walmart{i}.json")
        dfs.append(df_)

    df_raw = pd.concat(dfs).drop_duplicates(subset="id")
    df_raw.head()

    # %%
    df = pd.DataFrame(
        {
            "brand": df_raw["brand"],
            "category": df_raw["category"].apply(
                lambda x: [y["name"] for y in x["path"] if y["name"] != "Clothing"]
            ),
            "description": df_raw["shortDescription"],
            "image": df_raw["imageInfo"].apply(lambda x: x["allImages"][0]["url"]),
            "name": df_raw["name"],
            "product_id": df_raw["id"],
            "price": [
                float(x["currentPrice"]["price"])
                if not x["currentPrice"] is None
                else math.inf
                for x in df_raw["priceInfo"]
            ],
        }
    )
    df = df[df["category"].transform(lambda x: len(x)) >= 2]

    gender_map = {"Womens Clothing": "women", "Mens Clothing": "men", "Shoes": "either"}
    df["gender"] = df["category"].apply(lambda x: gender_map.get(x[0], "either"))
    df.head()

    # %%
    with open("data/descriptions.json", "r") as fh:
        descriptions = json.load(fh)

    # %%
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Download images
    for index, row in tqdm(df.iterrows()):
        image_url = row["image"]
        product_id = row["product_id"]
        category = row["category"][1]

        image_name = os.path.join(IMAGE_DIR, product_id + ".jpg")
        if not os.path.exists(image_name):
            urllib.request.urlretrieve(image_url, image_name)

        if product_id not in descriptions:
            prompt = f"""
            Describe the piece of clothing in the image of the following category: {category}
            Do include the color, style, material and other important attributes of the item.
            """
            image_path = f"{IMAGE_DIR}/{product_id}.jpg"

            try:
                result = gpt_vision(image_path, prompt)
                message = result.choices[0].message.content
            except Exception as e:
                print(e)
                message = None

            descriptions[row["product_id"]] = message

    # %%
    with open("data/descriptions.json", "w") as fh:
        json.dump(descriptions, fh)

    # %%
    vector_store = DeepLakeVectorStore(
        dataset_path=ACTIVELOOP_DATASET_TEXT, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

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
                text=desc,
                metadata={"name": name, "product_id": product_id, "gender": gender},
            )
            documents.append(doc)
        else:
            print(row)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    # %%

    ds = deeplake.empty(ACTIVELOOP_DATASET_IMG)

    with ds:
        ds.create_tensor("images", htype="image", sample_compression="jpeg")
        ds.create_tensor("ids", htype="tag")

    # %%
    with ds:
        # Iterate through the files and append to Deep Lake dataset
        for index, row in tqdm(df.iterrows()):
            product_id = row["product_id"]

            image_name = os.path.join(IMAGE_DIR, product_id + ".jpg")
            if os.path.exists(image_name):
                # Append data to the tensors
                ds.append({"images": deeplake.read(image_name), "ids": product_id})
# %%
