import sys
from io import StringIO
from typing import Any, Dict, List, Tuple

import gradio as gr
from gradio.themes import ThemeClass as Theme
from llama_index.agent.types import BaseAgent
from llama_index.llama_pack.base import BaseLlamaPack


class Capturing(list):
    """To capture the stdout from `BaseAgent.stream_chat` with `verbose=True`. Taken from
    https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
    """

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


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

    def get_modules(self) -> Dict[str, Any]:
        """Get modules."""
        return {"agent": self.agent}

    def _handle_user_message(self, user_message, history):
        """Handle the user submitted message. Clear message box, and append
        to the history."""
        return "", history + [(user_message, "")]

    def _generate_response(
        self, chat_history: List[Tuple[str, str]]
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Generate the response from agent, and capture the stdout of the
        ReActAgent's thoughts.
        """

        with Capturing() as output:
            response = self.agent.stream_chat(chat_history[-1][0])
        ansi = "\n========\n".join(output)
        html_output = self.conv.convert(ansi)

        for token in response.response_gen:
            chat_history[-1][1] += token
            yield chat_history, str(html_output)

    def _reset_chat(self) -> Tuple[str, str]:
        """Reset the agent's chat history. And clear all dialogue boxes."""
        # clear agent history
        self.agent.reset()
        return "", "", ""  # clear textboxes

    def _get_theme(self) -> Theme:
        from gradio.themes.utils import colors, fonts, sizes

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
        import gradio as gr

        demo = gr.Blocks(
            theme=self.theme,
            css="#box { height: 420px; overflow-y: scroll !important} #logo { align-self: center }",
        )
        with demo:
            with gr.Row():
                gr.Markdown(
                    """
                    # Chat With Your Agent
                    Powered by DeepLake, Gradio, LlamaIndex and LlamaHub 🦙\n
                    """
                )
                gr.Markdown(
                    "[![LlamaIndex](https://d3ddy8balm3goa.cloudfront.net/other/llama-index-light-transparent-sm-font.svg)](https://llamaindex.ai)",
                    elem_id="logo",
                )
            with gr.Row():
                chat_history = gr.Chatbot(
                    label="Message History",
                    scale=3,
                )
                with gr.Accordion("label"):
                    console_log = gr.HTML(elem_id="box")
            with gr.Row():
                user_message = gr.Textbox(
                    label="Enter text and press enter, or upload an image", scale=4
                )
                btn = gr.UploadButton("📁", file_types=["image"])
                clear = gr.ClearButton()

            # Handle new user message
            user_message.submit(
                fn=self._handle_user_message,
                inputs=[user_message, chat_history],
                outputs=[user_message, chat_history],
                queue=False,
            ).then(
                fn=self._generate_response,
                inputs=chat_history,
                outputs=[chat_history, console_log],
            )

            # Handle upload fine
            file_msg = btn.upload(
                fn=self._handle_image,
                inputs=[btn, chat_history],
                outputs=[chat_history],
                queue=False,
            ).then(
                fn=self._generate_response, inputs=chat_history, outputs=chat_history
            )

            # Handle click on clear button
            clear.click(
                self._reset_chat, None, [user_message, chat_history, console_log]
            )

        demo.launch(server_name="0.0.0.0", server_port=8080)

    def _handle_image(self, image, history):
        history = history + [((image.name,), None)]
        return history
