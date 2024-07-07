import openai
from dotenv import load_dotenv
import os
import streamlit as st


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = api_key

def createHistory(mensagem_usuario):
    contexto = """
    Voce é Um contador de historias profissional e voce é muito criativo em narrativas, entretanto elas sao tao boas
    que prendem a atencao do ouvinte , entao sua tarefa e criar um historia completamente inovadora sobre qualquer tema aleatorio e que prenda a atencao do ouvinte
    """
    prompt = f"{contexto}\n\nCliente: {mensagem_usuario}\nAssistente:"
    resposta = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200
    )
    return resposta.choices[0].text.strip()

def sendHistory():
    mensagem = "Crie uma historia, e me mande somente o texto da historia ate 300 carateres"
    resposta = createHistory(mensagem)
    return resposta




