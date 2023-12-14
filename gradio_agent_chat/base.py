import ast
import os
import re
import sys
from io import StringIO
from typing import Any, Dict, List, Tuple

import gradio as gr
from gradio.themes import ThemeClass as Theme
from gradio.themes.utils import colors, fonts, sizes
from llama_index.agent.types import BaseAgent
from llama_index.llama_pack.base import BaseLlamaPack


class GradioAgentChatPack(BaseLlamaPack):
    """Gradio chatbot to chat with your own Agent."""

    def __init__(
        self,
        agent: BaseAgent,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        try:
            from ansi2html import Ansi2HTMLConverter
        except ImportError:
            raise ImportError("Please install ansi2html via `pip install ansi2html`")

        self.agent = agent
        self.thoughts = ""
        self.conv = Ansi2HTMLConverter()
        self.theme = self._get_theme()
        self.demo = gr.Blocks(
            theme=self.theme,
            css=(
                "#box { height: 420px; overflow-y: scroll !important} "
                "#logo { display: flex; justify-content: flex-end; }"
            ),
        )

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"agent": self.agent}

    def _handle_user_message(self, user_message, history):
        """Handle the user submitted message. Clear message box, and append
        to the history."""
        return "", history + [(user_message, "")]

    def _generate_response(
        self, chat_history: List[Tuple[str, str]]
    ) -> List[Tuple[str, str]]:
        """Generate the response from agent"""

        response = self.agent.stream_chat(chat_history[-1][0])

        for token in response.response_gen:
            chat_history[-1][1] += token
            yield chat_history

        # def parse_file(msg: str):
        #     try:
        #         return ast.literal_eval(msg.splitlines()[-1])
        #     except:
        #         try:
        #             path = re.findall(r"\[Product Image\]\((.*?)\)", msg)[0]
        #             print(f"found path {path}")
        #             return (path,)
        #         except:
        #             return msg

        # chat_history[-1][1] = parse_file(chat_history[-1][1])
        # return chat_history

    def _reset_chat(self) -> Tuple[str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        # clear agent history
        self.agent.reset()
        return "", "", ""  # clear textboxes

    def _get_theme(self) -> Theme:
        llama_theme = gr.themes.Soft(
            primary_hue=colors.purple,
            secondary_hue=colors.pink,
            neutral_hue=colors.gray,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_md,
            text_size=sizes.text_lg,
            font=(
                fonts.GoogleFont("Quicksand"),
                "ui-sans-serif",
                "sans-serif",
            ),
            font_mono=(
                fonts.GoogleFont("IBM Plex Mono"),
                "ui-monospace",
                "monospace",
            ),
        )
        llama_theme.set(
            body_background_fill="#FFFFFF",
            body_background_fill_dark="#000000",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )

        return llama_theme

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Run the pipeline."""

        with self.demo:
            with gr.Row():
                gr.Markdown(
                    """
                    # Chat with your Outfit Recommender Assistant
                    powered by DeepLake, Gradio, LlamaIndex and LlamaHub 🦙\n
                    """
                )
                gr.Markdown(
                    "[![LlamaIndex](https://d3ddy8balm3goa.cloudfront.net/other/llama-index-light-transparent-sm-font.svg)](https://llamaindex.ai)",
                    elem_id="logo",
                )
            with gr.Row():
                chat_history = gr.Chatbot(
                    label="Chat",
                    avatar_images=(None, "activeloop_avatar.png"),
                    height=800,
                    show_copy_button=True,
                )
                # with gr.Accordion("Console log"):
                #     console_log = gr.HTML(elem_id="box")
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
                clear = gr.ClearButton(value="Reset", size="sm")

            # Add like/dislike event to the chat
            chat_history.like(self.print_like_dislike, None, None)

            # Handle new user message
            new_msg_event = user_message.submit(
                fn=self._handle_user_message,
                inputs=[user_message, chat_history],
                outputs=[user_message, chat_history],
                show_progress=True,
                # queue=False,
            )
            new_msg_event.then(
                fn=self._generate_response,
                inputs=chat_history,
                outputs=[chat_history],
                show_progress=True,
            )

            # Handle upload fine
            new_file_event = btn_upload_img.upload(
                fn=self._handle_image,
                inputs=[btn_upload_img, chat_history],
                outputs=[chat_history],
                show_progress=True,
                # queue=False,
            )
            new_file_event.then(
                fn=self._generate_response,
                inputs=chat_history,
                outputs=chat_history,
                show_progress=True,
            )

            # Handle click on clear button
            clear.click(self._reset_chat, None, [user_message, chat_history])

        self.demo.launch(server_name="0.0.0.0", server_port=8080)

    def _handle_image(self, image, history):
        history = history + [((image.name,), None)]
        return history

    def print_like_dislike(self, x: gr.LikeData):
        print(x.index, x.value, x.liked)
