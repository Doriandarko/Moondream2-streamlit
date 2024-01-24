import streamlit as st
from moondream import VisionEncoder, TextModel
from PIL import Image
from huggingface_hub import snapshot_download
from threading import Thread
from transformers import TextIteratorStreamer
import re
import time

@st.cache_resource
def load_model():
    # Download the model
    model_path = snapshot_download("vikhyatk/moondream1")
    # Initialize the vision encoder and text model
    vision_encoder = VisionEncoder(model_path)
    text_model = TextModel(model_path)
    return vision_encoder, text_model

# Load the model
vision_encoder, text_model = load_model()



# Streamlit app title
st.title("🌝 Moondream1 Vision Model")
st.write("A small but powerful vision model that outperforms models twice its size.")
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
    image = Image.open(st.session_state['uploaded_image'])
    st.image(image, caption='Uploaded Image.', use_column_width=True)

# Text input for the prompt
prompt = st.text_input("Question", value=st.session_state['prompt'])
st.session_state['prompt'] = prompt

# Function to generate text from the image and prompt
def generate_text(image_embeds, prompt):
    # Placeholder for the output text
    text_placeholder = st.empty()
    # Initialize the streamer
    streamer = TextIteratorStreamer(text_model.tokenizer, skip_special_tokens=True)
    # Arguments for the text generation
    generation_kwargs = dict(
        image_embeds=image_embeds, question=prompt, streamer=streamer
    )
    # Start the thread for text generation
    thread = Thread(target=text_model.answer_question, kwargs=generation_kwargs)
    thread.start()
    # Buffer to accumulate text
    buffer = ""
    # Loop to update the placeholder with the generated text
    while True:
        if thread.is_alive():
            time.sleep(0.1)  # Use time.sleep() instead of st.sleep()
        try:
            new_text = next(streamer)
            buffer += new_text
            if not new_text.endswith("<") and not new_text.endswith("END"):
                text_placeholder.markdown(buffer)
        except StopIteration:
            break  # No more items in the streamer

    # Wait for the thread to finish if it's still running
    thread.join()

    # Clean up the final buffer and display it
    final_text = re.sub("<$", "", re.sub("END$", "", buffer))
    text_placeholder.markdown(final_text)

# Button to trigger text generation
if st.button("Generate"):
    if st.session_state['uploaded_image'] is not None and st.session_state['prompt']:
        # Open the uploaded image
        image = Image.open(st.session_state['uploaded_image'])
        # Get image embeddings
        image_embeds = vision_encoder(image)
        # Call the generate_text function
        generate_text(image_embeds, st.session_state['prompt'])
    else:
        st.warning("Please upload an image and enter a prompt.")