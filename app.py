from gradio_agent_chat.base import GradioAgentChatPack
from main import agent

# Run `gradio app.py`
if __name__ == "__main__":
    GradioAgentChatPack(agent).run()
