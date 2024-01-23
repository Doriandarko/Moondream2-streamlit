# Moondream1 Vision Model Streamlit App

This is a Streamlit app that uses the Moondream1 Vision Model to generate text based on an uploaded image and a user-provided prompt.

## Features

- Upload an image in PNG or JPEG format.
- Enter a prompt to guide the text generation.
- Generate text based on the uploaded image and prompt.

## How to Run

1. Install the required Python packages:

\ bash
pip install streamlit moondream PIL huggingface_hub transformers
\


2. Run the Streamlit app:

\ bash
streamlit run vision.py
\

3. Open the app in your web browser at `http://localhost:8501`.

## Usage

1. Upload an image using the file uploader.
2. Enter a prompt in the text input field.
3. Click the "Generate" button to generate text based on the image and prompt.

## About the Model

The Moondream1 Vision Model is a small but powerful vision model that outperforms models twice its size. It was created by [@vikhyatk](https://twitter.com/vikhyatk).

## License

This project is open source under the MIT license.
