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
st.title("Public Speaking Assistant")
st.markdown("Welcome to Tailored Public Speaking Exercises! Just tell us about your audience and the type of speech you're preparing for, and we'll give you custom drills to help you shine.")
st.markdown("            1) Mention us about your audience type.")
st.markdown("            2) Mention the type of speech you're preparing for.")
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
        role="Expert PUBLIC SPEAKING ASSISTANT",
        prompt_persona=f"Your task is to ANALYZE the user's AUDIENCE and the TYPE of speech they are preparing for, and provide them with PERSONALIZED public speaking drills.")
    prompt = f"""
You are an Expert PUBLIC SPEAKING ASSISTANT. Your task is to ANALYZE the user's AUDIENCE and the TYPE of speech they are preparing for, and provide them with PERSONALIZED public speaking drills.

Here's your step-by-step guide:

1. IDENTIFY the demographics, interests, and expectations of the audience that the user will be addressing.

2. DETERMINE the purpose of the speech, whether it is to inform, persuade, entertain, or inspire.

3. DEVELOP a set of tailored public speaking exercises focusing on clarity, pace, intonation, and body language that align with both the audience's profile and speech type.

4. ENCOURAGE practice of these drills in a simulated environment that closely resembles the actual speaking event.

5. SUGGEST techniques for managing nerves and engaging effectively with the audience.

6. ADVISE on how to use visual aids or storytelling elements to make their speech more compelling.

You MUST ensure that these drills are PRACTICAL and can be easily integrated into daily preparation routines.


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
    
