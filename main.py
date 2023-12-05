# %%

import os
import urllib.request
import math
import pandas as pd

# %%
df_raw = pd.read_json("data/walmart.json")
# %%
df_raw.head()
print("\n".join([c for c in df_raw.columns]))
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

# %%
df.head()

# %%

# Create images directory if it doesn't exist
if not os.path.exists('data/images'):
    os.makedirs('data/images')

# Download images
for index, row in df.iterrows():
    image_url = row['image']
    image_name = os.path.join('data/images', row['product_id'] + '.jpg')
    if not os.path.exists(image_name):
        urllib.request.urlretrieve(image_url, image_name)
# %%
