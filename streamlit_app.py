import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Gantilah dengan nama model LLaMA yang sesuai dari Hugging Face
model_name = "meta-llama/LLaMA-7B"  # Contoh model

# Load tokenizer dan model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Fungsi untuk menghasilkan respons dari model
def generate_response(prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit UI
st.title("Asisten AI berbasis LLaMA")
st.write("Masukkan pertanyaan atau perintah di bawah ini:")

# Input pengguna
user_input = st.text_input("Anda:", "")

# Ketika tombol ditekan
if user_input:
    response = generate_response(user_input)
    st.write("Asisten AI: " + response)
