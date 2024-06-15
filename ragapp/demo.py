from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
import streamlit as st
# from collections.abc import Mapping
# from IPython.display import Markdown as md
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import SystemMessagePromptTemplate
# chat_template = ChatPromptTemplate.from_messages([SystemMessage(content="""You are a Helpful AI Bot. You take the context and question from user. Your answer should be based on the specific context."""),
#     HumanMessagePromptTemplate.from_template("""Aswer the question based on the given context.Context:{context}Question: {question}Answer: """)
# ])
chat_template = ChatPromptTemplate.from_messages([
    ("ai", "You are a Helpful AI Bot. You take the context and question from user. Your answer should be based on the specific context."),
    ("human", """Aswer the question based on the given context.Context:{context}Question: {question}Answer: """)
])

st.title("Q&A for Leave No Context Behind” Paper using RAG 🤖📝")
# st.subheading("Enter the query")
text=st.text_input("Leave your query based on LLM")
if st.button("Search"):
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyAbjdm8WqCNMjNfARjwI1ODNQD_0mfTzPE",model="models/embedding-001")
    chat_model = ChatGoogleGenerativeAI(google_api_key="AIzaSyAbjdm8WqCNMjNfARjwI1ODNQD_0mfTzPE",model="gemini-1.5-pro-latest")
    db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)
    output_parser = StrOutputParser()
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    retriever = db_connection.as_retriever(search_kwargs={"k": 5})
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}| chat_template| chat_model| output_parser)
    response = rag_chain.invoke(text)
    st.write(response)
