# %%
#
#
import os
import time

import gradio as gr

from main import agent


# %%
#
#
def generate_outfit_test(weather, occasion):
    if weather.lower() == "hot":
        if occasion.lower() == "casual":
            return "A light t-shirt and shorts."
        elif occasion.lower() == "formal":
            return "A light suit with a white shirt."
    elif weather.lower() == "cold":
        if occasion.lower() == "casual":
            return "A warm sweater and jeans."
        elif occasion.lower() == "formal":
            return "A heavy suit with a warm coat."
    else:
        return "Sorry, I could not find an outfit for your input."


# 1 try
# demo = gr.Interface(fn=greet, inputs="text", outputs="text")

# 2 try
# demo = gr.Interface(
#     generate_outfit_test,
#     inputs=[
#         gr.inputs.Dropdown(choices=["hot", "cold"], label="Weather"),
#         gr.inputs.Dropdown(choices=["casual", "formal"], label="Occasion"),
#     ],
#     outputs="text",
# )

# 3 try
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button("Clear")

#     def user(user_message, history):
#         return "", history + [[user_message, None]]

#     def bot(history):
#         # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
#         bot_message = generate_outfit_test()
#         history[-1][1] = ""
#         for character in bot_message:
#             history[-1][1] += character
#             time.sleep(0.05)
#             yield history

#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
#         bot, chatbot, chatbot
#     )
#     clear.click(lambda: None, None, chatbot, queue=False)

# demo.queue()
# demo.launch()


# 4 try
def print_like_dislike(x: gr.LikeData):
    print(x.index, x.value, x.liked)


def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)


def add_file(history, file):
    history = history + [((file.name,), None)]
    return history


def bot(history):
    # response = generate_outfit_test("hot", "casual")
    response = agent.chat("a black top")
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        time.sleep(0.05)
        yield history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            None,
            (os.path.join(os.path.dirname(__file__), "activeloop_avatar.png")),
        ),
    )

    with gr.Row():
        txt = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter, or upload an image",
            container=False,
        )
        btn = gr.UploadButton("📁", file_types=["image", "video", "audio"])

    txt_msg = txt.submit(
        fn=add_text, inputs=[chatbot, txt], outputs=[chatbot, txt], queue=False
    ).then(bot, chatbot, chatbot, api_name="bot_response")
    txt_msg.then(lambda: gr.Textbox(interactive=True), None, [txt], queue=False)
    file_msg = btn.upload(add_file, [chatbot, btn], [chatbot], queue=False).then(
        bot, chatbot, chatbot
    )

    chatbot.like(print_like_dislike, None, None)


if __name__ == "__main__":
    demo.queue()
    demo.launch(show_api=False, show_error=True, debug=True)

# %%
