import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocesamiento import cargar_textos, fragmentar_texto, limpiar_texto


# ---------------------------------------------------------------------------
# Lógica del chatbot
# ---------------------------------------------------------------------------

def crear_indice(tamanio_chunk: int = 500, solapamiento: int = 50):
    """Carga los documentos y crea una matriz TF-IDF reutilizable."""
    if tamanio_chunk <= 0 or solapamiento < 0 or solapamiento >= tamanio_chunk:
        raise ValueError(
            "El tamaño del fragmento debe ser positivo y el solapamiento "
            "debe ser menor que el tamaño."
        )

    documentos = cargar_textos()
    fragmentos = []

    for _, documento in documentos.iterrows():
        for texto in fragmentar_texto(
            documento["texto"], tamanio_chunk, solapamiento
        ):
            fragmentos.append({
                "texto": texto,
                "titulo": documento["titulo"],
                "categoria": documento["categoria"],
                "archivo": documento["archivo"],
            })

    if not fragmentos:
        raise ValueError("No se encontraron fragmentos en data/documentos/.")

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    textos_limpios = [limpiar_texto(fragmento["texto"]) for fragmento in fragmentos]
    matriz = vectorizador.fit_transform(textos_limpios)

    return fragmentos, vectorizador, matriz


def buscar_respuesta(pregunta: str, indice) -> str:
    """Devuelve el fragmento más relevante y la fuente del documento."""
    fragmentos, vectorizador, matriz = indice
    pregunta_limpia = limpiar_texto(pregunta)
    if not pregunta_limpia:
        return "No encontré términos útiles en la pregunta."

    vector_pregunta = vectorizador.transform([pregunta_limpia])
    if vector_pregunta.nnz == 0:
        return "No se encontró una respuesta relevante en los documentos."

    similitudes = cosine_similarity(vector_pregunta, matriz).flatten()
    indice_mejor = int(similitudes.argmax())
    similitud = float(similitudes[indice_mejor])
    if similitud <= 0:
        return "No se encontró una respuesta relevante en los documentos."

    fragmento = fragmentos[indice_mejor]
    return (
        f"Fragmento relevante:\n{fragmento['texto'].strip()}\n\n"
        f"Fuente: {fragmento['titulo']} — {fragmento['archivo']}"
    )


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Creando índice del chatbot...")
def indice_cacheado():
    """El índice se crea una sola vez, no en cada mensaje."""
    return crear_indice()


def _mostrar_respuesta(texto: str) -> None:
    """Separa el fragmento de la fuente para mostrarlos con mejor formato."""
    if "\n\nFuente: " not in texto:
        st.write(texto)
        return
    fragmento, fuente = texto.split("\n\nFuente: ", 1)
    st.markdown("**Fragmento relevante:**")
    st.info(fragmento.removeprefix("Fragmento relevante:\n"))
    st.caption(f"📄 Fuente: {fuente}")


def pagina_chatbot() -> None:
    st.header("💬 Chatbot de documentos")
    try:
        indice = indice_cacheado()
    except ValueError as error:
        st.error(str(error))
        return

    st.caption(
        f"Fragmentos indexados: {len(indice[0])}. "
        "Responde con el fragmento más relevante de los documentos."
    )

    if "chat_mensajes" not in st.session_state:
        st.session_state["chat_mensajes"] = []

    if st.session_state["chat_mensajes"] and st.button("🧹 Limpiar conversación"):
        st.session_state["chat_mensajes"] = []
        st.rerun()

    for mensaje in st.session_state["chat_mensajes"]:
        with st.chat_message(mensaje["rol"]):
            if mensaje["rol"] == "user":
                st.write(mensaje["texto"])
            else:
                _mostrar_respuesta(mensaje["texto"])

    pregunta = st.chat_input("Escribe tu pregunta...")
    if pregunta and pregunta.strip():
        st.session_state["chat_mensajes"] += [
            {"rol": "user", "texto": pregunta.strip()},
            {"rol": "assistant", "texto": buscar_respuesta(pregunta.strip(), indice)},
        ]
        st.rerun()