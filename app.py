import streamlit as st

from openai import OpenAI

from dotenv import dotenv_values

config = dotenv_values(".env")

client = OpenAI(api_key = config["OPENAI_API_KEY"])
st.title("AI Code Reviewer💬🤖")
st.subheader("Write your program📝")
prompt=st.text_area("Enter your text")
if st.button("Find Bugs🐞")==True:
    response=client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role":"system","content":"""You are a helpful AI Assistant and Code reviewer
            Find the Bugs and error in the program."""},
            {"role":"user","content":prompt}
        ]

    )
    st.write(response.choices[0].message.content)
