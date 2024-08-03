import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="GPT Model Tester", page_icon="🤖", layout="wide")

# Prompt template
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant. Please respond to the user's questions."),
    ("user", "Question: {question}")
])

# Function to generate response
def generate_response(question, openai_api_key, model, temperature=0.2, max_tokens=1024):
    try:
        llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens, api_key=openai_api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({"question": question})
        return answer
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Main app
def main():
    st.title("🤖 GPT Model Tester")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        model = st.selectbox("Select OpenAI Model", [
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo-0125"
        ])
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=50, max_value=2000, value=150, step=50)

    # Main page
    st.write("Enter your question below and select the GPT model to test.")
    user_input = st.text_area("Your question:", height=100)

    if st.button("Generate Response", type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not user_input:
            st.warning("Please enter a question to get a response.")
        else:
            with st.spinner("Generating response..."):
                response = generate_response(user_input, api_key, model, temperature, max_tokens)
            
            st.subheader("Response:")
            st.write(response)

            # Display model settings
            st.subheader("Model Settings:")
            st.write(f"Model: {model}")
            st.write(f"Temperature: {temperature}")
            st.write(f"Max Tokens: {max_tokens}")

    # Add information about the app
    st.markdown("---")
    st.markdown("""
    ### About this app
    This app allows you to test different GPT models using the OpenAI API. 
    Enter your API key, select a model, and adjust the settings to see how 
    the responses change.

    Please note that using this app will consume API credits from your OpenAI account.
    """)

if __name__ == "__main__":
    main()
