# Smart Fashion Assistant

We are building an Outfit Recommender Agent that uses multiple tools.


## Getting started

For this project we use two external services: OpenAI and Open Weather Map. Both services require an API KEY.

1. [OpenAI](https://openai.com/blog/openai-api)
2. [Open Weather Map](https://openweathermap.org/api)

## Running the project

There are two main notebooks in this project. `simple_agent` and `main`.

The notebook `simple_agent` ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/shopping-assistant/blob/main/simple_agent.ipynb)) showcases an earlier version of the agent without UI and the weather integration. We recommend getting started there.

The notebook `main` ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tryolabs/shopping-assistant/blob/main/main.ipynb)) puts everything together and can be used to run the complete demo.

## Structure

If you want to take a closer look at the implementation behind the demo, most of the code can be located inside `src/`. In particular, `dataset.py` and `main.py` contain most of this project logic.

```bash
├── README.md
├── __init__.py
├── assets
│   ├── smith.png
│   └── user.png
├── blogpost
│   ├── assets
│   └── blog.md
├── data
│   ├── descriptions.json
│   ├── walmart1.json
│   ├── walmart2.json
│   └── walmart3.json
├── main.ipynb
├── requirements.txt
├── simple_agent.ipynb
└── src
    ├── dataset.py
    ├── main.py
    ├── openai_utils.py
    └── utils.py
```