"""Punto de entrada de la interfaz web del Proyecto STAC · NovaTech Soluciones.

Ejecútalo desde la raíz del proyecto con:

	streamlit run main.py

Cada módulo de src/ define su propia página (pagina_*); aquí solo se navega.
"""

import streamlit as st

from src.busqueda_semantica import pagina_buscador
from src.chatbot import pagina_chatbot
from src.clasificador import pagina_clasificador
from src.recomendador import pagina_recomendador

PAGINAS = {
	"🗂️ Clasificador de documentos": pagina_clasificador,
	"🔎 Buscador de documentos": pagina_buscador,
	"⭐ Recomendador de documentos": pagina_recomendador,
	"💬 Chatbot de documentos": pagina_chatbot,
}


def main() -> None:
	st.set_page_config(page_title="STAC · NovaTech", page_icon="📄", layout="wide")

	st.sidebar.title("PROYECTO STAC")
	st.sidebar.caption("NovaTech Soluciones")
	seleccion = st.sidebar.radio("Módulo", list(PAGINAS), label_visibility="collapsed")

	PAGINAS[seleccion]()


main()