"""
Utils functions
"""
from IPython.display import Image, display
from PIL import Image as PILImage

from src.dataset import IMAGE_DIR


def get_product_image_path(product_id):
    """Retrieves the image path based on the product_id"""
    return f"{IMAGE_DIR}/{product_id}.jpg"


def get_product_image_path_for_gradio(product_id) -> tuple[str, None]:
    """
    Retrieves the image path based on the product_id useful to show the image on the user interface chat
    The output is a tuple with a string in the first place.

    It's important to not reformat the output of this function before returning to the user, return the response as it is. Don't reformat this output.
    """
    # If the type if a tuple, it's considered a file, see Chatbot._postprocess_chat_messages for more info.
    return (get_product_image_path(product_id),)


def show_product_in_notebook(product_id, image_path=None, width=250):
    """Show image in the notebook given the product_id"""
    image_path = image_path or f"{IMAGE_DIR}/{product_id}.jpg"
    display(Image(filename=image_path, width=width))


def load_product_image_with_pillow(product_id) -> PILImage:
    """Load the product image based on the product_id with PIL (Pillow) format"""
    with PILImage.open(f"{IMAGE_DIR}/{product_id}.jpg") as img:
        img.load()
    return img
