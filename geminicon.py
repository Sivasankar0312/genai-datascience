import streamlit as st
from dotenv import dotenv_values
import google.generativeai as genai 
config = dotenv_values(".env")


st.title("AI Tutor For DataScience 🤖📝💻")
genai.configure(api_key = config["gemini"])
ai=genai.GenerativeModel(model_name="gemini-1.5-pro-latest",system_instruction="""You are helpful ai Teaching Assistant .Given a answer for the user query when it is related datascience topic otherwise tell i don't know if user and say hai then say hai how can i help you""")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"]=[]

chat = ai.start_chat(history=st.session_state['chat_history'])
for msg in chat.history:

    st.chat_message(msg.role).write(msg.parts[0].text)

user_prompt=st.chat_input()

if user_prompt:
    st.chat_message("user").write(user_prompt)
    response=chat.send_message(user_prompt)
    st.chat_message("ai").write(response.text)
    print(chat.history)
    st.session_state["chat_history"]=chat.history