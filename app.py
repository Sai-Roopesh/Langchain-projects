import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
import os
from dotenv import load_dotenv
import time
import tiktoken
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(page_title="GPT Model Comparator",
                   page_icon="🤖", layout="wide")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant. Please respond to the user's questions."),
    HumanMessagePromptTemplate.from_template("Question: {question}")
])

# Function to generate response


def generate_response(question, openai_api_key, model, temperature, max_tokens):
    try:
        start_time = time.time()

        # Tokenize input
        encoding = tiktoken.encoding_for_model(model)
        input_tokens = len(encoding.encode(question))

        llm = ChatOpenAI(model=model, temperature=temperature,
                         max_tokens=max_tokens, api_key=openai_api_key)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser

        # Measure API call time
        api_call_start = time.time()
        answer = chain.invoke({"question": question})
        api_call_end = time.time()

        end_time = time.time()

        # Calculate metrics
        output_tokens = len(encoding.encode(answer))
        total_tokens = input_tokens + output_tokens
        inference_time = end_time - start_time
        api_call_time = api_call_end - api_call_start
        tokens_per_second = output_tokens / inference_time if inference_time > 0 else 0

        # Estimate cost (you'd need to update this based on current pricing)
        cost_per_1k_tokens = 0.002  # Example price, update as needed
        estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens

        metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "inference_time": inference_time,
            "api_call_time": api_call_time,
            "tokens_per_second": tokens_per_second,
            "estimated_cost": estimated_cost,
        }

        return answer, metadata
    except Exception as e:
        return f"An error occurred: {str(e)}", {}

# Function to generate PDF


def generate_pdf(user_input, results):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("GPT Model Comparison Results", styles['Title']))
    elements.append(Paragraph(f"Prompt: {user_input}", styles['Heading2']))

    for model, data in results.items():
        elements.append(Paragraph(f"Model: {model}", styles['Heading3']))
        elements.append(
            Paragraph(f"Response: {data['response']}", styles['BodyText']))

        # Create a table for metadata
        metadata = data['metadata']
        table_data = [
            ["Setting/Metric", "Value"],
            ["Temperature", f"{data['temperature']}"],
            ["Max Tokens", f"{data['max_tokens']}"],
            ["Input Tokens", f"{metadata['input_tokens']}"],
            ["Output Tokens", f"{metadata['output_tokens']}"],
            ["Total Tokens", f"{metadata['total_tokens']}"],
            ["Inference Time", f"{metadata['inference_time']:.2f} seconds"],
            ["API Call Time", f"{metadata['api_call_time']:.2f} seconds"],
            ["Tokens per Second", f"{metadata['tokens_per_second']:.2f}"],
            ["Estimated Cost", f"${metadata['estimated_cost']:.6f}"],
        ]

        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Paragraph("<br/><br/>", styles['Normal']))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Main app


def main():
    st.title("🤖 GPT Model Comparator")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        models = st.multiselect("Select OpenAI Models", [
            "gpt-3.5-turbo",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo-0125"
        ])

    # Main page
    st.write("Enter your question below and select the GPT models to compare.")
    user_input = st.text_area("Your question:", height=100)

    if not models:
        st.warning("Please select at least one model from the sidebar.")
    else:
        # Create columns for model settings
        cols = st.columns(len(models))
        model_settings = {}

        for i, model in enumerate(models):
            with cols[i]:
                st.subheader(f"Settings for {model}")
                temperature = st.slider(
                    f"Temperature for {model}", min_value=0.0, max_value=1.0, value=0.2, step=0.1, key=f"temp_{model}")
                max_tokens = st.slider(
                    f"Max Tokens for {model}", min_value=50, max_value=2000, value=150, step=50, key=f"tokens_{model}")
                model_settings[model] = {
                    "temperature": temperature, "max_tokens": max_tokens}

        if st.button("Generate Responses", type="primary"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            elif not user_input:
                st.warning("Please enter a question to get responses.")
            else:
                results = {}
                for model in models:
                    with st.spinner(f"Generating response for {model}..."):
                        response, metadata = generate_response(
                            user_input,
                            api_key,
                            model,
                            model_settings[model]["temperature"],
                            model_settings[model]["max_tokens"]
                        )

                    st.subheader(f"Response from {model}:")
                    st.write(response)

                    # Display model settings and metadata
                    st.subheader("Model Settings and Metadata:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(
                            f"Temperature: {model_settings[model]['temperature']}")
                        st.write(
                            f"Max Tokens: {model_settings[model]['max_tokens']}")
                        st.write(f"Input Tokens: {metadata['input_tokens']}")
                        st.write(f"Output Tokens: {metadata['output_tokens']}")
                    with col2:
                        st.write(f"Total Tokens: {metadata['total_tokens']}")
                        st.write(
                            f"Inference Time: {metadata['inference_time']:.2f} seconds")
                        st.write(
                            f"API Call Time: {metadata['api_call_time']:.2f} seconds")
                        st.write(
                            f"Tokens per Second: {metadata['tokens_per_second']:.2f}")
                        st.write(
                            f"Estimated Cost: ${metadata['estimated_cost']:.6f}")
                    st.markdown("---")

                    # Store results for PDF generation
                    results[model] = {
                        "response": response,
                        "metadata": metadata,
                        "temperature": model_settings[model]["temperature"],
                        "max_tokens": model_settings[model]["max_tokens"]
                    }

                # Add PDF download button
                pdf_buffer = generate_pdf(user_input, results)
                st.download_button(
                    label="Download Results as PDF",
                    data=pdf_buffer,
                    file_name="gpt_model_comparison_results.pdf",
                    mime="application/pdf"
                )

    # Add information about the app
    st.markdown("---")
    st.markdown("""
    ### About this app
    This app allows you to compare responses from different GPT models using the OpenAI API. 
    Enter your API key, select multiple models, and adjust the settings for each model to see how 
    the responses differ across models.

    Please note that using this app will consume API credits from your OpenAI account.
    """)


if __name__ == "__main__":
    main()
