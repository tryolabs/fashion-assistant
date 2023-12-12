"""
Utils functions
"""
from IPython.display import Image, display


def show_product_in_notebook(product_id, image_path=None, width=250):
    """Show image in the notebook given the product_id"""
    IMAGE_PATH = image_path or f"data/images/{product_id}.jpg"
    display(Image(filename=IMAGE_PATH, width=width))
