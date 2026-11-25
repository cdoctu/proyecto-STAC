"""Interfaz Streamlit del Proyecto STAC · NovaTech Soluciones.

Colócalo en la raíz del proyecto (junto a main.py) y ejecútalo con:

    streamlit run app.py

No modifica nada de src/: reutiliza sus funciones y clases.
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from src import busqueda_semantica as bs
from src import chatbot as cb
from src import recomendador as rec
from src.clasificador import (
    RUTA_MODELO,
    cargar_modelo,
    clasificar_nuevos,
    crear_pipeline,
    dividir_datos,
    guardar_modelo,
)
from src.preprocesamiento import cargar_nuevos, cargar_textos, limpiar_texto

st.set_page_config(page_title="STAC · NovaTech", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Recursos en caché (se crean una sola vez, no en cada interacción)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Indexando documentos...")
def indice_busqueda(usar_raiz: bool):
    return bs.crear_indice(usar_raiz)


@st.cache_resource(show_spinner="Preparando el recomendador...")
def get_recomendador():
    return rec.crear_recomendador()


@st.cache_resource(show_spinner="Creando índice del chatbot...")
def indice_chatbot():
    return cb.crear_indice()


def cuerpo(texto: str) -> str:
    """Documento sin la primera línea (el título), listo para mostrar."""
    return "\n".join(texto.splitlines()[1:]).strip().replace("\n", "  \n")


# ---------------------------------------------------------------------------
# 1. Clasificador
# ---------------------------------------------------------------------------

def figura_matriz(matriz, etiquetas, nombre):
    fig, ax = plt.subplots(figsize=(max(6, len(etiquetas) * 1.1), max(4.5, len(etiquetas) * 0.9)))
    imagen = ax.imshow(matriz, interpolation="nearest", cmap="Blues")
    fig.colorbar(imagen, ax=ax)
    ax.set(xlabel="Categoría predicha", ylabel="Categoría real",
           title=f"Matriz de confusión: {nombre}")
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas)
    umbral = matriz.max() / 2
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > umbral else "black")
    fig.tight_layout()
    return fig


def entrenar_y_evaluar(test_size: float, cv: int, guardar: bool):
    df = cargar_textos()
    if df.empty:
        st.error("No se encontraron documentos en data/documentos/.")
        return
    df["texto_limpio"] = df["texto"].apply(limpiar_texto)

    df_train, df_test = dividir_datos(df, test_size=test_size)
    puntuaciones = cross_val_score(
        crear_pipeline(), df_train["texto_limpio"], df_train["categoria"],
        cv=cv, scoring="accuracy",
    )

    modelo = crear_pipeline().fit(df_train["texto_limpio"], df_train["categoria"])
    y_pred = modelo.predict(df_test["texto_limpio"])
    y_real = df_test["categoria"]
    etiquetas = sorted(y_real.unique())

    reporte = pd.DataFrame(classification_report(
        y_real, y_pred, labels=etiquetas, target_names=etiquetas,
        zero_division=0, output_dict=True,
    )).T
    matriz = confusion_matrix(y_real, y_pred, labels=etiquetas)

    # Modelo final: se reentrena con todos los documentos (igual que main.py)
    modelo_final = crear_pipeline().fit(df["texto_limpio"], df["categoria"])
    st.session_state["modelo"] = modelo_final
    if guardar:
        guardar_modelo(modelo_final)

    st.session_state["clf_resultados"] = {
        "total": len(df), "n_train": len(df_train), "n_test": len(df_test),
        "cv": cv, "cv_scores": puntuaciones,
        "exactitud": accuracy_score(y_real, y_pred),
        "reporte": reporte, "matriz": matriz, "etiquetas": etiquetas,
        "guardado": guardar,
    }


def obtener_modelo():
    if "modelo" in st.session_state:
        return st.session_state["modelo"]
    if RUTA_MODELO.exists():
        st.session_state["modelo"] = cargar_modelo()
        return st.session_state["modelo"]
    return None


def pagina_clasificador():
    st.header("🗂️ Clasificador de documentos")
    st.caption("TF-IDF + Naive Bayes sobre los documentos de data/documentos/.")

    with st.expander("⚙️ Parámetros de entrenamiento", expanded=True):
        c1, c2, c3 = st.columns(3)
        test_size = c1.slider("Proporción de prueba", 0.1, 0.5, 0.3, 0.05)
        cv = c2.slider("Folds de validación cruzada", 2, 5, 3)
        guardar = c3.checkbox("Guardar modelo en models/", value=True)
        if st.button("🚀 Entrenar y evaluar", type="primary"):
            with st.spinner("Entrenando..."):
                try:
                    entrenar_y_evaluar(test_size, cv, guardar)
                except ValueError as error:
                    st.error(f"No se pudo entrenar: {error}")

    r = st.session_state.get("clf_resultados")
    if r:
        st.subheader("Resultados")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Documentos", r["total"])
        m2.metric("Entrenamiento / Prueba", f"{r['n_train']} / {r['n_test']}")
        m3.metric(f"Validación cruzada ({r['cv']} folds)", f"{r['cv_scores'].mean():.1%}",
                  f"± {r['cv_scores'].std():.1%}", delta_color="off")
        m4.metric("Exactitud en prueba", f"{r['exactitud']:.1%}")
        if r["guardado"]:
            st.success(f"Modelo guardado en: {RUTA_MODELO}")

        izq, der = st.columns([1, 1])
        with izq:
            st.markdown("**Reporte de clasificación**")
            st.dataframe(r["reporte"].style.format("{:.2f}"), use_container_width=True)
        with der:
            fig = figura_matriz(r["matriz"], r["etiquetas"], "Naive Bayes")
            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.subheader("Clasificar documentos nuevos")
    modelo = obtener_modelo()
    if modelo is None:
        st.info("Entrena el modelo primero (o coloca uno en models/clasificador.joblib).")
        return

    tab_carpeta, tab_texto = st.tabs(["📁 Carpeta data/nuevos/", "✍️ Pegar un texto"])
    with tab_carpeta:
        if st.button("Clasificar archivos de data/nuevos/"):
            df_nuevos = cargar_nuevos()
            if df_nuevos.empty:
                st.warning("No hay archivos .txt nuevos para clasificar.")
            else:
                df_nuevos["texto_limpio"] = df_nuevos["texto"].apply(limpiar_texto)
                resultado = clasificar_nuevos(modelo, df_nuevos)
                resultado["confianza"] = resultado["confianza"].map(lambda c: f"{c:.2%}")
                st.dataframe(resultado, hide_index=True, use_container_width=True)
    with tab_texto:
        texto = st.text_area("Texto a clasificar", height=160, key="clf_texto")
        if st.button("Clasificar texto") and texto.strip():
            limpio = limpiar_texto(texto)
            if not limpio:
                st.warning("El texto no tiene términos útiles.")
            else:
                proba = modelo.predict_proba([limpio])[0]
                mejor = modelo.classes_[int(np.argmax(proba))]
                st.success(f"Categoría predicha: **{bs.nombre_area(mejor)}** "
                           f"({proba.max():.1%} de confianza)")
                st.bar_chart(pd.Series(proba, index=modelo.classes_, name="Probabilidad"))


# ---------------------------------------------------------------------------
# 2. Buscador semántico
# ---------------------------------------------------------------------------

def aportes_por_termino(consulta, indice, usar_raiz, id_doc, maximo=5):
    """Qué palabras coincidieron y cuánto aportó cada una a la similitud."""
    df, vectorizador, matriz = indice
    vector = vectorizador.transform([bs.limpiar(consulta, usar_raiz)])
    fila = df.index[df["id"] == id_doc][0]
    aportes = vector.multiply(matriz[fila]).toarray().flatten()
    terminos = vectorizador.get_feature_names_out()
    orden = np.argsort(aportes)[::-1][:maximo]
    return pd.Series({terminos[i]: aportes[i] for i in orden if aportes[i] > 0}, name="Aporte")


def mostrar_resultados_busqueda(consulta, resultado, indice, usar_raiz, area):
    df = indice[0]
    donde = bs.nombre_area(area) if area else "todas las áreas"
    st.markdown(f"Resultados para **“{consulta}”** en {donde}:")
    if resultado.empty:
        st.warning("No encontré documentos con esas palabras. El buscador compara palabras: "
                   "prueba con otras o con una sola palabra clave.")
        return
    st.caption("La similitud va de 0 (nada en común) a 1 (idéntico).")
    docs = df.set_index("id")

    for _, fila in resultado.iterrows():
        doc = docs.loc[fila["id"]]
        with st.container(border=True):
            st.markdown(f"**{fila['posicion']}. {fila['titulo']}**")
            st.caption(f"Área: {bs.nombre_area(fila['categoria'])}  ·  Id: {fila['id']}")
            st.progress(min(float(fila["similitud"]), 1.0), text=f"Similitud: {fila['similitud']}")
            st.write(doc["resumen"])
            c1, c2 = st.columns(2)
            with c1.expander("📖 Leer documento"):
                st.markdown(cuerpo(doc["texto"]))
            with c2.expander("🔍 ¿Por qué salió este resultado?"):
                aportes = aportes_por_termino(consulta, indice, usar_raiz, fila["id"])
                if aportes.empty:
                    st.write("Sin palabras en común.")
                else:
                    st.bar_chart(aportes, horizontal=True)
                    st.caption(f"Suma de las palabras mostradas: {aportes.sum():.4f}")


def correr_pruebas(k: int):
    normal, con_raiz = indice_busqueda(False), indice_busqueda(True)
    existentes = set(normal[0]["id"])
    filas = []
    for consulta, esperado in bs.PRUEBAS:
        if esperado not in existentes:
            st.warning(f"El documento esperado '{esperado}' no existe en data/documentos/.")
        a = bs.buscar(consulta, normal, k, usar_raiz=False)
        b = bs.buscar(consulta, con_raiz, k, usar_raiz=True)
        pa = a["id"].iloc[0] if not a.empty else "(ninguno)"
        pb = b["id"].iloc[0] if not b.empty else "(ninguno)"
        filas.append({
            "Consulta": consulta, "Esperado": esperado,
            "Sin raíz": pa, "Sim. sin raíz": a["similitud"].iloc[0] if not a.empty else 0,
            "✔ sin raíz": pa == esperado,
            "Con raíz": pb, "Sim. con raíz": b["similitud"].iloc[0] if not b.empty else 0,
            "✔ con raíz": pb == esperado,
        })
    tabla = pd.DataFrame(filas)
    c1, c2 = st.columns(2)
    c1.metric("Aciertos sin raíz", f"{tabla['✔ sin raíz'].sum()}/{len(tabla)}")
    c2.metric("Aciertos con raíz", f"{tabla['✔ con raíz'].sum()}/{len(tabla)}")
    st.dataframe(tabla, hide_index=True, use_container_width=True)


def pagina_buscador():
    st.header("🔎 Buscador de documentos")
    st.caption("Búsqueda por similitud TF-IDF (n-gramas 1–2) entre tu pregunta y los documentos.")

    with st.expander("⚙️ Opciones avanzadas"):
        c1, c2 = st.columns(2)
        k = c1.slider("Resultados a mostrar", 1, 10, 3)
        usar_raiz = c2.checkbox("Reducir palabras a su raíz", value=False)

    indice = indice_busqueda(usar_raiz)
    areas = sorted(indice[0]["categoria"].unique())
    modo = st.radio("Modo", ["Buscar", "Explorar por área", "Consultas de prueba"], horizontal=True)

    if modo == "Buscar":
        area = st.selectbox(
            "Área", [None] + areas, key="bus_area",
            format_func=lambda a: "Todas las áreas" if a is None else bs.nombre_area(a),
        )
        if area:
            ejemplos = bs.EJEMPLOS_POR_AREA.get(area, [])
            if ejemplos:
                st.caption("Ejemplos: " + "  ·  ".join(f"“{e}”" for e in ejemplos))
        consulta = st.text_input("Escribe tu pregunta con tus propias palabras", key="bus_consulta")
        if consulta.strip():
            resultado = bs.buscar(consulta.strip(), indice, k, usar_raiz, area)
            mostrar_resultados_busqueda(consulta.strip(), resultado, indice, usar_raiz, area)

    elif modo == "Explorar por área":
        area = st.selectbox("Área", areas, format_func=bs.nombre_area, key="exp_area")
        grupo = indice[0][indice[0]["categoria"] == area]
        st.write(f"**{len(grupo)} documentos en {bs.nombre_area(area)}**")
        for _, fila in grupo.iterrows():
            with st.expander(fila["titulo"]):
                st.markdown(cuerpo(fila["texto"]))

    else:
        st.write("Compara la búsqueda con y sin raíz de palabras usando las consultas de prueba.")
        if st.button("▶️ Correr consultas de prueba"):
            with st.spinner("Probando..."):
                correr_pruebas(k)


# ---------------------------------------------------------------------------
# 3. Recomendador
# ---------------------------------------------------------------------------

def usuario_actual() -> str:
    return st.session_state.get("usuario", "").strip() or "invitado"


def abrir_documento(id_documento: str):
    """Callback: abre un documento en el lector y lo anota en el historial."""
    get_recomendador().registrar_lectura(usuario_actual(), id_documento)
    st.session_state["leyendo"] = id_documento


def cerrar_documento():
    st.session_state["leyendo"] = None


def render_lista(tabla, prefijo, etiqueta="Puntaje"):
    if tabla.empty:
        st.info("No hay documentos para mostrar.")
        return
    for _, fila in tabla.iterrows():
        with st.container(border=True):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"**{fila['titulo']}**")
            c1.caption(f"Área: {bs.nombre_area(fila['categoria'])}  ·  {etiqueta}: {fila['puntaje']}")
            c1.write(fila["resumen"])
            if "motivo" in tabla.columns and fila["motivo"]:
                c1.caption(f"💡 Motivo: {fila['motivo']}")
            c2.button("📖 Leer", key=f"{prefijo}_{fila['id']}",
                      on_click=abrir_documento, args=(fila["id"],))


def lector():
    id_doc = st.session_state.get("leyendo")
    if not id_doc:
        return
    r = get_recomendador()
    fila = r.documentos[r.documentos["id"] == id_doc].iloc[0]
    with st.container(border=True):
        st.subheader(fila["titulo"])
        st.caption(f"Área: {bs.nombre_area(fila['categoria'])}  ·  Id: {fila['id']}")
        st.markdown(cuerpo(fila["texto"]))
        st.markdown("**Documentos parecidos a este**")
        render_lista(r.similares(id_doc, 3), "sim", "Similitud")
        st.button("✖ Cerrar documento", on_click=cerrar_documento)
    st.divider()


def pagina_recomendador():
    st.header("⭐ Recomendador de documentos")
    r = get_recomendador()

    c1, c2 = st.columns([2, 3])
    c1.text_input("Tu nombre (para guardar tu historial)", key="usuario",
                  placeholder="invitado")
    usuario = usuario_actual()
    total = len(r.historial.get(usuario, []))
    c2.markdown(f"<br>Método: **{r.vectorizador.nombre}** · Persona: **{usuario}** · "
                f"Historial: **{total}** eventos", unsafe_allow_html=True)

    lector()

    modo = st.radio(
        "Qué quieres hacer",
        ["Buscar y recibir recomendaciones", "Según mi historial", "Explorar documentos",
         "Mi historial", "Probar el recomendador"],
        horizontal=True,
    )

    if modo == "Buscar y recibir recomendaciones":
        with st.form("form_rec_busqueda"):
            consulta = st.text_input("Escribe tu pregunta con tus propias palabras")
            enviado = st.form_submit_button("Buscar", type="primary")
        if enviado and consulta.strip():
            consulta = consulta.strip()
            tiene_historial = bool(r.historial.get(usuario))
            resultados = r.buscar(consulta, 3)
            recomendados = r.recomendar(usuario, consulta, 3, excluir=resultados["id"])
            r.registrar_consulta(usuario, consulta)
            st.session_state["rec_busqueda"] = (consulta, resultados, recomendados, tiene_historial)

        guardado = st.session_state.get("rec_busqueda")
        if guardado:
            consulta, resultados, recomendados, tiene_historial = guardado
            st.markdown(f"#### Resultados para: {consulta}")
            render_lista(resultados, "res", "Similitud")
            if tiene_historial:
                st.markdown("#### También te pueden servir según tu historial")
                render_lista(recomendados, "rec")
            else:
                st.info("Cuando tengas historial, aquí verás recomendaciones pensadas para ti.")

    elif modo == "Según mi historial":
        tabla = r.recomendar(usuario, k=3)
        if tabla.empty:
            st.info("Todavía no hay historial para recomendar. Haz una búsqueda primero.")
        else:
            render_lista(tabla, "his")

    elif modo == "Explorar documentos":
        for categoria, grupo in r.documentos.groupby("categoria"):
            with st.expander(bs.nombre_area(categoria)):
                for _, fila in grupo.iterrows():
                    a, b = st.columns([6, 1])
                    a.write(fila["titulo"])
                    b.button("📖 Leer", key=f"exp_{fila['id']}",
                             on_click=abrir_documento, args=(fila["id"],))

    elif modo == "Mi historial":
        eventos = r.historial.get(usuario, [])
        if not eventos:
            st.info("Tu historial está vacío.")
        else:
            filas = [{
                "Fecha": e["fecha"],
                "Tipo": "Lectura" if e["tipo"] == "lectura" else "Consulta",
                "Detalle": r.titulo(e["valor"]) if e["tipo"] == "lectura" else e["valor"],
            } for e in reversed(eventos)]
            st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)

            st.divider()
            confirmar = st.checkbox("Confirmo que quiero borrar mi historial")
            if st.button("🗑️ Borrar mi historial", disabled=not confirmar):
                r.borrar_historial(usuario)
                st.session_state.pop("rec_busqueda", None)
                st.rerun()

    else:
        st.write("Mide qué tan bien recomienda el sistema (no toca tu historial guardado).")
        if st.button("▶️ Ejecutar evaluación"):
            with st.spinner("Evaluando..."):
                similares = r.evaluar_similares(3)
                historial = r.evaluar_historial(3)

            st.markdown("**Documentos parecidos** — % de los 3 recomendados de la misma área que el origen")
            similares = similares.assign(categoria=similares["categoria"].map(bs.nombre_area))
            st.dataframe(similares.rename(columns={"categoria": "Área", "precision": "Precisión"})
                         .style.format({"Precisión": "{:.0%}"}),
                         hide_index=True, use_container_width=True)
            st.metric("Promedio", f"{similares['precision'].mean():.0%}")

            st.markdown("**Recomendaciones según el historial** — % de los 3 recomendados del área esperada")
            historial = historial.assign(area_esperada=historial["area_esperada"].map(bs.nombre_area))
            st.dataframe(historial.rename(columns={
                "area_esperada": "Área esperada", "consultas": "Consultas",
                "recomendados": "Recomendados", "precision": "Precisión",
            }).style.format({"Precisión": "{:.0%}"}),
                hide_index=True, use_container_width=True)
            st.metric("Promedio", f"{historial['precision'].mean():.0%}")


# ---------------------------------------------------------------------------
# 4. Chatbot
# ---------------------------------------------------------------------------

def render_respuesta(texto: str):
    if "\n\nFuente: " in texto:
        fragmento, fuente = texto.split("\n\nFuente: ", 1)
        fragmento = fragmento.removeprefix("Fragmento relevante:\n")
        st.markdown("**Fragmento relevante:**")
        st.info(fragmento)
        st.caption(f"📄 Fuente: {fuente}")
    else:
        st.write(texto)


def pagina_chatbot():
    st.header("💬 Chatbot de documentos")
    try:
        indice = indice_chatbot()
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"Fragmentos indexados: {len(indice[0])}. "
               "Responde con el fragmento más relevante de los documentos.")

    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if st.session_state["chat"] and st.button("🧹 Limpiar conversación"):
        st.session_state["chat"] = []
        st.rerun()

    for mensaje in st.session_state["chat"]:
        with st.chat_message(mensaje["rol"]):
            if mensaje["rol"] == "user":
                st.write(mensaje["texto"])
            else:
                render_respuesta(mensaje["texto"])

    pregunta = st.chat_input("Escribe tu pregunta...")
    if pregunta and pregunta.strip():
        respuesta = cb.buscar_respuesta(pregunta.strip(), indice)
        st.session_state["chat"] += [
            {"rol": "user", "texto": pregunta.strip()},
            {"rol": "assistant", "texto": respuesta},
        ]
        st.rerun()


# ---------------------------------------------------------------------------
# Navegación
# ---------------------------------------------------------------------------

PAGINAS = {
    "🗂️ Clasificador": pagina_clasificador,
    "🔎 Buscador": pagina_buscador,
    "⭐ Recomendador": pagina_recomendador,
    "💬 Chatbot": pagina_chatbot,
}

st.sidebar.title("PROYECTO STAC")
st.sidebar.caption("NovaTech Soluciones")
seleccion = st.sidebar.radio("Módulo", list(PAGINAS), label_visibility="collapsed")
PAGINAS[seleccion]()