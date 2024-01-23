import streamlit as st
from moondream import VisionEncoder, TextModel
from PIL import Image
from huggingface_hub import snapshot_download
from threading import Thread
from transformers import TextIteratorStreamer
import re
import time

# Download the model
model_path = snapshot_download("vikhyatk/moondream1")

# Initialize the vision encoder and text model
vision_encoder = VisionEncoder(model_path)
text_model = TextModel(model_path)

# Streamlit app title
st.title("🌝 Moondream1 Vision Model")
st.write("A small but powerful vision model that outperforms models twice its size.")
st.markdown("Created by [@vikhyatk](https://twitter.com/vikhyatk)")
# File uploader for the image
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
# Text input for the prompt
prompt = st.text_input("Question")

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
    if uploaded_image is not None and prompt:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        # Get image embeddings
        image_embeds = vision_encoder(image)
        # Call the generate_text function
        generate_text(image_embeds, prompt)
    else:
        st.warning("Please upload an image and enter a prompt.")