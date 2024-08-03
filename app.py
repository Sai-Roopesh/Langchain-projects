import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()



# prompt

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Please respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)


def generate_response(question, openai_api_key, model, temperature=0.2, max_tokens=1024):
    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, api_key=openai_api_key)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke(
        {
            "question": question
        },
    )

    return answer


## Title

st.title("Q&A Chatbot")

## Sidebar

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter OpenAI API Key: ", type="password")

## drop down to select model

model = st.sidebar.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4o-mini","gpt-4o"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)

max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)


## Main Page


user_input = st.text_input("Enter your question here:")

if st.button("Ask") and user_input is not None:
    response = generate_response(user_input, api_key, model, temperature, max_tokens)
    st.write(response)
else:
    if user_input is None:
        st.write("Please enter a question to get a response.")






