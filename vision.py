import streamlit as st
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import time

@st.cache_resource
def load_model():
    # Load the model and tokenizer
    model_id = "vikhyatk/moondream2"
    revision = "2024-03-05"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# Streamlit app title
st.title("🌝 Moondream2 Vision Model")
st.write("An enhanced vision model that outperforms its predecessor.")
st.markdown("Model created by [@vikhyatk](https://twitter.com/vikhyatk). App by [@skirano](https://twitter.com/skirano)")

# Initialize session state for uploaded image and prompt
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = ""

# File uploader for the image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    st.session_state['uploaded_image'] = uploaded_image

# Display the uploaded image
if st.session_state['uploaded_image']:
    image = Image.open(st.session_state['uploaded_image'])
    st.image(image, caption='Uploaded Image.', use_column_width=True)

# Text input for the prompt
prompt = st.text_input("Question", value=st.session_state['prompt'])
st.session_state['prompt'] = prompt

# Function to generate text from the image and prompt
def generate_text(image, prompt):
    # Placeholder for the output text
    text_placeholder = st.empty()

    # Encode the image
    enc_image = model.encode_image(image)

    # Generate text
    generated_text = model.answer_question(enc_image, prompt, tokenizer)

    # Display the generated text
    text_placeholder.markdown(generated_text)

# Button to trigger text generation
if st.button("Generate"):
    if st.session_state['uploaded_image'] is not None and st.session_state['prompt']:
        # Open the uploaded image
        image = Image.open(st.session_state['uploaded_image'])

        # Call the generate_text function
        generate_text(image, st.session_state['prompt'])
    else:
        st.warning("Please upload an image and enter a prompt.")