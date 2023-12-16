# %%
from llama_index.agent import OpenAIAgent
from main import inventory_query_engine_tool, outfit_description_tool
from llama_index.llms import OpenAI

# %%
llm = OpenAI(model="gpt-4", temperature=0.2)

agent = OpenAIAgent.from_tools(
    system_prompt="""
    You are a specialized shopping assistant.

    Customers will provide you with a piece of clothing, and you will generate a matching outfit.

    Always remember to ask for the user gender.

    Your final answer needs to be the product_id associated with the best matching product in our inventory.

    For each product of the outfit, search the inventory.

    Incldue the total price of the recommended outfit.
    """,
    tools=[
        inventory_query_engine_tool,
        outfit_description_tool,
    ],
    llm=llm,
    verbose=True,
)

# %%
#
# %%
r = agent.chat("Hi")
print(r)
# %%
r = agent.chat("What are your tools?")
print(r)
# %%
r = agent.chat("I want an outfit for a casual birthday party")
print(r)
# %%
r = agent.chat("I'm a woman")
print(r)
# %%
r = agent.chat("My budget is only 20, can you recommend an alternative?")
print(r)
# %%
