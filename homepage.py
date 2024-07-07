import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from gtts import gTTS
import os

model_name = 'neuralmind/bert-base-portuguese-cased'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

# Função para gerar o áudio e atualizar o status
def generate_audio(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_text = tokenizer.decode(torch.argmax(logits, dim=-1)[0])
    tts = gTTS(predicted_text, lang='pt-br')
    tts.save("output.mp3")

# Configuração da página Streamlit
def main():
    st.set_page_config(page_title="Contador de Histórias", page_icon="📚", layout="wide")
    st.title("Contador de Histórias")
    st.subheader("Feito por EagleAI")

    st.markdown("---")

    if st.button("Clique para Gerar Audio Aleatorio" , use_container_width=True):
        st.audio('', format='audio/mp3', data=None, start_time=0)
        st.text("Pronto para começar!")

    text_input = st.text_area("Insira o texto para ser falado", height=150)
    
    if st.button("Gerar Áudio" , use_container_width=True):
        if text_input.strip():
            generate_audio(text_input)
            st.success("Áudio gerado com sucesso!")
        else:
            st.warning("Por favor, insira algum texto para gerar o áudio.")

    st.markdown("---")

    if os.path.exists('output.mp3'):
        audio_file = open('output.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", start_time=0)
        audio_file.close()

        st.download_button("Baixar Áudio", data=audio_bytes, file_name="output.mp3" , use_container_width=True)

if __name__ == '__main__':
    main()
