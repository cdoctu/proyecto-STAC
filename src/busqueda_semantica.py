from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocesamiento import limpiar_texto

BASE = Path(__file__).resolve().parent.parent
CARPETA_DOCS = BASE / "data" / "documentos"
CARPETA_RESULTADOS = BASE / "resultados"

# Consultas de prueba
PRUEBAS = [
    ("¿Cuántos días de vacaciones me tocan y cómo los pido?", "rh_01_politica_vacaciones"),
    ("¿Cómo publico una nueva versión en producción?", "dev_04_manual_despliegue_cicd"),
    ("¿Cómo reporto un correo sospechoso?", "seg_06_guia_phishing"),
    ("¿Cómo se atiende un ticket de un cliente?", "sop_01_procedimiento_tickets"),
    ("¿Cómo se hace una retrospectiva de sprint?", "pro_06_guia_retrospectiva"),
    ("El cliente no puede entrar a su cuenta", "sop_09_kb_no_puedo_entrar"),
    ("¿Qué hago si pierdo mi computadora de trabajo?", "seg_07_politica_dispositivos"),
]

# Nombres para mostrar
NOMBRES_AREAS = {
    "recursos_humanos": "Recursos Humanos",
    "desarrollo_software": "Desarrollo de Software",
    "seguridad_informacion": "Seguridad de la Información",
    "soporte_clientes": "Soporte a Clientes",
    "gestion_proyectos": "Gestión de Proyectos",
}

# Preguntas de ejemplo
EJEMPLOS_POR_AREA = {
    "recursos_humanos": ["¿Cómo pido mis vacaciones?", "¿Cuántos días de home office tengo?"],
    "desarrollo_software": ["¿Cómo se publica una nueva versión?", "¿Cómo se revisa el código de un compañero?"],
    "seguridad_informacion": ["¿Cómo reporto un correo sospechoso?", "¿Cada cuánto se cambian las contraseñas?"],
    "soporte_clientes": ["¿Cómo se atiende un ticket?", "¿Cuáles son los tiempos del SLA?"],
    "gestion_proyectos": ["¿Cómo se hace una retrospectiva?", "¿Qué hago si el cliente pide un cambio?"],
}


# ---------------------------------------------------------------------------
# Lógica del buscador
# ---------------------------------------------------------------------------

def nombre_area(carpeta):
    return NOMBRES_AREAS.get(carpeta, carpeta.replace("_", " ").title())


def resumir(texto, largo=110):
    """Primeras palabras del documento (sin el título), para mostrar en los resultados."""
    cuerpo = " ".join(texto.splitlines()[1:]).strip()
    if len(cuerpo) <= largo:
        return cuerpo
    return cuerpo[:largo].rsplit(" ", 1)[0] + "..."


def texto_sin_titulo(texto):
    """Documento sin la primera línea (el título), listo para mostrar en Markdown."""
    return "\n".join(texto.splitlines()[1:]).strip().replace("\n", "  \n")


# Limpieza

def reducir_a_raiz(palabra):
    """Quita el plural y la última vocal: correo / correos -> corr."""
    if len(palabra) > 4 and palabra.endswith("es"):
        palabra = palabra[:-2]
    elif len(palabra) > 3 and palabra.endswith("s"):
        palabra = palabra[:-1]
    if len(palabra) > 3 and palabra[-1] in "aeiou":
        palabra = palabra[:-1]
    return palabra


def limpiar(texto, usar_raiz=False):
    texto_limpio = limpiar_texto(texto)
    if usar_raiz:
        texto_limpio = " ".join(reducir_a_raiz(p) for p in texto_limpio.split())
    return texto_limpio


# 1. Indexar: limpiar todos los documentos y aplicar fit_transform
def cargar_documentos():
    filas = []
    for archivo in sorted(CARPETA_DOCS.rglob("*.txt")):
        texto = archivo.read_text(encoding="utf-8")
        filas.append({
            "id": archivo.stem,
            "titulo": texto.splitlines()[0],
            "categoria": archivo.parent.name,
            "resumen": resumir(texto),
            "texto": texto,
        })
    return pd.DataFrame(filas)


def crear_indice(usar_raiz=False):
    df = cargar_documentos()
    df["texto_limpio"] = df["texto"].apply(lambda t: limpiar(t, usar_raiz))

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matriz = vectorizador.fit_transform(df["texto_limpio"])  # matriz documentos x términos
    return df, vectorizador, matriz


# Vectorizar la consulta, comparar y ordenar
def buscar(consulta, indice, k=3, usar_raiz=False, area=None):
    df, vectorizador, matriz = indice

    consulta_limpia = limpiar(consulta, usar_raiz)
    vector_consulta = vectorizador.transform([consulta_limpia])

    similitudes = cosine_similarity(vector_consulta, matriz).flatten()
    if area:
        similitudes = np.where(df["categoria"].values == area, similitudes, 0)
    orden = np.argsort(similitudes)[::-1][:k]

    resultado = pd.DataFrame({
        "posicion": range(1, len(orden) + 1),
        "id": df["id"].iloc[orden].values,
        "titulo": df["titulo"].iloc[orden].values,
        "categoria": df["categoria"].iloc[orden].values,
        "similitud": similitudes[orden].round(4),
    })
    return resultado[resultado["similitud"] > 0].reset_index(drop=True)


# Prueba con varias consultas
def probar(k=3):
    """Compara la búsqueda con y sin raíz. Devuelve la tabla y la guarda en resultados/."""
    indice_normal = indice_cacheado(False)
    indice_raiz = indice_cacheado(True)

    filas = []
    for consulta, esperado in PRUEBAS:
        normal = buscar(consulta, indice_normal, k, usar_raiz=False)
        con_raiz = buscar(consulta, indice_raiz, k, usar_raiz=True)

        primero_n = normal["id"].iloc[0] if not normal.empty else "(ninguno)"
        primero_r = con_raiz["id"].iloc[0] if not con_raiz.empty else "(ninguno)"
        filas.append({
            "consulta": consulta,
            "esperado": esperado,
            "primero_sin_raiz": primero_n,
            "similitud_sin_raiz": normal["similitud"].iloc[0] if not normal.empty else 0,
            "acierto_sin_raiz": primero_n == esperado,
            "primero_con_raiz": primero_r,
            "similitud_con_raiz": con_raiz["similitud"].iloc[0] if not con_raiz.empty else 0,
            "acierto_con_raiz": primero_r == esperado,
        })

    tabla = pd.DataFrame(filas)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    tabla.to_csv(CARPETA_RESULTADOS / "consultas_prueba.csv", index=False, encoding="utf-8")
    return tabla


# Detalles técnicos: por qué salió cada resultado
def aportes_por_termino(consulta, indice, id_documento, usar_raiz=False, max_terminos=5):
    """Qué palabras coincidieron y cuánto aportó cada una a la similitud."""
    df, vectorizador, matriz = indice
    terminos = vectorizador.get_feature_names_out()

    vector_consulta = vectorizador.transform([limpiar(consulta, usar_raiz)])
    numero_doc = df.index[df["id"] == id_documento][0]
    aportes = vector_consulta.multiply(matriz[numero_doc]).toarray().flatten()
    mejores = np.argsort(aportes)[::-1][:max_terminos]
    return pd.Series(
        {terminos[i]: aportes[i] for i in mejores if aportes[i] > 0},
        name="Aporte",
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Indexando documentos...")
def indice_cacheado(usar_raiz=False):
    """El índice se crea una sola vez por configuración, no en cada interacción."""
    return crear_indice(usar_raiz)


def _mostrar_resultados(consulta, resultado, indice, usar_raiz, area):
    donde = nombre_area(area) if area else "todas las áreas"
    st.markdown(f"Resultados para **“{consulta}”** en {donde}:")
    if resultado.empty:
        st.warning(
            "No encontré documentos con esas palabras. El buscador compara palabras: "
            "prueba con otras o con una sola palabra clave."
        )
        return

    st.caption("La similitud va de 0 (nada en común) a 1 (idéntico).")
    documentos = indice[0].set_index("id")

    for _, fila in resultado.iterrows():
        doc = documentos.loc[fila["id"]]
        with st.container(border=True):
            st.markdown(f"**{fila['posicion']}. {fila['titulo']}**")
            st.caption(f"Área: {nombre_area(fila['categoria'])}  ·  Id: {fila['id']}")
            st.progress(min(float(fila["similitud"]), 1.0), text=f"Similitud: {fila['similitud']}")
            st.write(doc["resumen"])

            col_leer, col_explicar = st.columns(2)
            with col_leer.expander("📖 Leer documento"):
                st.markdown(texto_sin_titulo(doc["texto"]))
            with col_explicar.expander("🔍 ¿Por qué salió este resultado?"):
                aportes = aportes_por_termino(consulta, indice, fila["id"], usar_raiz)
                if aportes.empty:
                    st.write("Sin palabras en común.")
                else:
                    st.bar_chart(aportes, horizontal=True)
                    st.caption(f"Suma de las palabras mostradas: {aportes.sum():.4f}")


def _modo_buscar(indice, areas, k, usar_raiz):
    area = st.selectbox(
        "Área",
        [None] + areas,
        key="bus_area",
        format_func=lambda a: "Todas las áreas" if a is None else nombre_area(a),
    )
    if area:
        ejemplos = EJEMPLOS_POR_AREA.get(area, [])
        if ejemplos:
            st.caption("Ejemplos: " + "  ·  ".join(f"“{e}”" for e in ejemplos))

    consulta = st.text_input("Escribe tu pregunta con tus propias palabras", key="bus_consulta").strip()
    if consulta:
        resultado = buscar(consulta, indice, k, usar_raiz, area)
        _mostrar_resultados(consulta, resultado, indice, usar_raiz, area)


def _modo_explorar(indice, areas):
    area = st.selectbox("Área", areas, format_func=nombre_area, key="bus_explorar_area")
    grupo = indice[0][indice[0]["categoria"] == area]
    st.write(f"**{len(grupo)} documentos en {nombre_area(area)}**")
    for _, fila in grupo.iterrows():
        with st.expander(fila["titulo"]):
            st.markdown(texto_sin_titulo(fila["texto"]))


def _modo_pruebas(k):
    st.write("Compara la búsqueda con y sin raíz de palabras usando las consultas de prueba.")
    if not st.button("▶️ Correr consultas de prueba"):
        return

    with st.spinner("Probando..."):
        tabla = probar(k)

    existentes = set(indice_cacheado(False)[0]["id"])
    for _, esperado in PRUEBAS:
        if esperado not in existentes:
            st.warning(
                f"El documento esperado '{esperado}' no existe en data/documentos/. "
                "Cambia la lista PRUEBAS en src/busqueda_semantica.py."
            )

    col1, col2 = st.columns(2)
    col1.metric("Aciertos sin raíz", f"{tabla['acierto_sin_raiz'].sum()}/{len(tabla)}")
    col2.metric("Aciertos con raíz", f"{tabla['acierto_con_raiz'].sum()}/{len(tabla)}")
    st.dataframe(tabla, hide_index=True)
    st.caption(f"Tabla guardada en: {CARPETA_RESULTADOS / 'consultas_prueba.csv'}")


def pagina_buscador():
    st.header("🔎 Buscador de documentos")
    st.caption("Búsqueda por similitud TF-IDF (n-gramas 1–2) entre tu pregunta y los documentos.")

    with st.expander("⚙️ Opciones avanzadas"):
        col1, col2 = st.columns(2)
        k = col1.slider("Resultados a mostrar", 1, 10, 3)
        usar_raiz = col2.checkbox("Reducir palabras a su raíz", value=False)

    indice = indice_cacheado(usar_raiz)
    areas = sorted(indice[0]["categoria"].unique())
    modo = st.radio("Modo", ["Buscar", "Explorar por área", "Consultas de prueba"], horizontal=True)

    if modo == "Buscar":
        _modo_buscar(indice, areas, k, usar_raiz)
    elif modo == "Explorar por área":
        _modo_explorar(indice, areas)
    else:
        _modo_pruebas(k)