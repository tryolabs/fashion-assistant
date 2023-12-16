import os
import shutil
import logging
import re
import urllib.parse
from typing import List, Tuple

import gradio as gr

from main import agent, clean_input_image, INPUT_IMAGE_DIR

reg = re.compile(r'[0-9A-Z]{12}')


#
# Event handlers
#
def handle_user_message(user_message, history):
    """Handle the user submitted message. Clear message box, and append
    to the history."""
    return "", history + [(user_message, "")]


def handle_image(image, history):
    """Handle uploaded image. Add it to the chat history"""

    path = os.path.join(INPUT_IMAGE_DIR, os.path.basename(image.name))
    shutil.copyfile(image.name, path)    
    message = "I just uploaded the image"

    history = history + [(message, " ")]
    return history


def generate_response(chat_history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Generate the response from agent"""

    iframe_html = '<iframe src={url} width="300px" height="600px"></iframe>'
    iframe_url = "https://app.activeloop.ai/visualizer/iframe?url=hub://genai360/walmart-images&query=" 

    response = agent.stream_chat(chat_history[-1][0])

    for token in response.response_gen:
        chat_history[-1][1] += token

        product_ids = reg.findall(chat_history[-1][1])        
        if len(product_ids) >= 2:
            query = "select * where " + " or ".join([f"contains(ids, '{x}')" for x in product_ids])          
            url = iframe_url + urllib.parse.quote(query)
        else:
            url = "about:blank"
        
        html = iframe_html.format(url=url)

        yield chat_history, html

    print("Mua ha ha ha")


def reset_chat(self) -> Tuple[str, str]:
    """Reset the agent's chat history. And clear all dialogue boxes."""
    # Clear agent history
    agent.reset()
    clean_input_image()

    # Reset chat history
    return "", ""


def print_like_dislike(x: gr.LikeData):
    logging.info(x.index, x.value, x.liked)


#
# Gradio application
#
with gr.Blocks(
    title="Outfit Recommender ✨",
    css="#box { height: 420px; overflow-y: scroll !important} #logo { align-self: right }",
    theme='gradio/soft'
) as demo:
    #
    # Add components
    #

    with gr.Row():
        gr.Markdown(
            """
            # Chat with your Outfit Recommender ✨
            """,
            elem_classes="center",
        )
    with gr.Row():
        chat_history = gr.Chatbot(
            label="Chat",
            avatar_images=("assets/user.png", "assets/smith.png"),
            scale = 2,
            show_copy_button=True,
        )
        outfit = gr.HTML(
            """
            <iframe src="about:blank" width="300px" height="600px"></iframe>
            """
        )
    with gr.Row():
        user_message = gr.Textbox(
            placeholder="Enter text and press enter, or upload an image",
            scale=4,
            container=False,
        )
    with gr.Row():
        btn_upload_img = gr.UploadButton(
            "Upload image 🖼️ ", size="sm", file_types=["image"]
        )
        btn_reset = gr.ClearButton(value="Reset", size="sm")

    gr.Markdown(
        """
        Made with ❤️ by [Tryolabs](https://tryolabs.com/)
        """,
        elem_classes="center",
    )


    #
    # Configure Event Listeners
    #

    # Add like/dislike event to the chat
    chat_history.like(print_like_dislike, None, None)

    # Handle new user message
    new_msg_event = user_message.submit(
        fn=handle_user_message,
        inputs=[user_message, chat_history],
        outputs=[user_message, chat_history],
        show_progress=True,
    )
    new_msg_event.then(
        fn=generate_response,
        inputs=chat_history,
        outputs=[chat_history, outfit],
        show_progress=True,
    )

    # Handle upload file
    new_file_event = btn_upload_img.upload(
        fn=handle_image,
        inputs=[btn_upload_img, chat_history],
        outputs=[chat_history],
        show_progress=True,
    )
    new_file_event.then(
        fn=generate_response,
        inputs=chat_history,
        outputs=[chat_history, outfit],
        show_progress=True,
    )

    # Handle click on reset button
    btn_reset.click(reset_chat, None, [user_message, chat_history])

# Run `gradio app.py` on the terminal
if __name__ == "__main__":
    clean_input_image()
    demo.launch(server_name="0.0.0.0", server_port=8080)
