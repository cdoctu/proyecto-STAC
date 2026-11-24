from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocesamiento import cargar_textos, fragmentar_texto, limpiar_texto


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


def iniciar_chatbot() -> None:
    indice = crear_indice()
    print(f"Chatbot listo. Fragmentos indexados: {len(indice[0])}")
    print("Escribe 'salir' para terminar.")

    while True:
        try:
            pregunta = input("\nPregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if pregunta.lower() in {"salir", "0"}:
            break
        if pregunta:
            print(f"\n{buscar_respuesta(pregunta, indice)}")
