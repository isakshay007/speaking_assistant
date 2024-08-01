import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Personalized Gift Assistant")
st.markdown("Welcome to Personalized Gift Assistant! Let us help you find the perfect gift for any occasion, tailored to your recipient's unique interests and your budget.")
st.markdown("            1) Mention your receiver's age. ")
st.markdown("            2) Mention your receiver's interest.")
st.markdown("            3) Mention the occasion.")
st.markdown("            4) Mention your budget.")
input = st.text_input(" Please enter the above details:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role="Expert GIFT CONSULTANT",
        prompt_persona=f"Your task is to CURATE a personalized list of 5-7 GIFTS for the user and provide EXPLANATIONS for each choice, taking into account the RECEIVER'S AGE, RECEIVER'S INTERESTS, the OCCASION, and the BUDGET.")
    prompt = f"""
You are an Expert GIFT CONSULTANT. Your task is to CURATE a personalized list of 5-7 GIFTS for the user and provide EXPLANATIONS for each choice, taking into account the RECEIVER'S AGE, RECEIVER'S INTERESTS, the OCCASION, and the BUDGET.

Here's how you will approach this task:

1. IDENTIFY the information about the receiver's age and interests to ensure that your recommendations are AGE-APPROPRIATE and ALIGN with their preferences.Next, CONSIDER THE OCCASION for which the gift is intended to ensure that your suggestions are SUITABLE and THOUGHTFUL for that specific event. Then, ANALYZE the budget parameters to make sure that your recommendations are AFFORDABLE and PROVIDE VALUE within the user's financial limits.

2. Now, IDENTIFY a list of 5-7 GIFTS that meet all these criteria. Make sure each gift suggestion is CREATIVE and UNIQUE to show thoughtfulness.

3. For each gift on your list, EXPLAIN precisely WHY you have chosen it by LINKING it back to the receiver's age, interests, occasion, and budget in a clear manner.

4. Ensure that your explanations HIGHLIGHT how each gift is particularly TAILORED to the receiver's personal tastes or needs.

By following these steps meticulously, you will craft a list that not only delights but also impresses with its personalized touch.


 """

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Assist!"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)