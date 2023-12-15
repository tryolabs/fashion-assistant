import logging
from typing import List, Tuple

import gradio as gr

from main import agent


#
# Event handlers
#
def handle_user_message(user_message, history):
    """Handle the user submitted message. Clear message box, and append
    to the history."""
    return "", history + [(user_message, "")]


def handle_image(image, history):
    """Handle uploaded image. Add it to the chat history"""
    history = history + [((image.name,), None)]
    return history


def generate_response(chat_history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Generate the response from agent"""

    response = agent.stream_chat(chat_history[-1][0])

    for token in response.response_gen:
        chat_history[-1][1] += token
        yield chat_history


def reset_chat(self) -> Tuple[str, str]:
    """Reset the agent's chat history. And clear all dialogue boxes."""
    # Clear agent history
    agent.reset()

    # Reset chat history
    return "", ""


def print_like_dislike(x: gr.LikeData):
    logging.info(x.index, x.value, x.liked)


#
# Gradio application
#
with gr.Blocks(
    title="Outfit Recommender ✨",
    css=".center { display: flex; justify-content: center; }",
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
            avatar_images=(None, "activeloop_avatar.png"),
            height=800,
            show_copy_button=True,
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
        outputs=[chat_history],
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
        outputs=chat_history,
        show_progress=True,
    )

    # Handle click on reset button
    btn_reset.click(reset_chat, None, [user_message, chat_history])

# Run `gradio app.py` on the terminal
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
