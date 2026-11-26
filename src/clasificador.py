from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocesamiento import cargar_nuevos, cargar_textos, limpiar_texto

RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "clasificador.joblib"


# ---------------------------------------------------------------------------
# Lógica del clasificador
# ---------------------------------------------------------------------------

def dividir_datos(
	df: pd.DataFrame,
	test_size: float = 0.3,
	random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	return train_test_split(
		df,
		test_size=test_size,
		random_state=random_state,
		stratify=df["categoria"],
	)


def crear_pipeline() -> Pipeline:
	return Pipeline([
		("tfidf", TfidfVectorizer()),
		("classifier", MultinomialNB()),
	])


def guardar_modelo(modelo: Pipeline, ruta: Path = RUTA_MODELO) -> Path:
	ruta.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(modelo, ruta)
	return ruta


def cargar_modelo(ruta: Path = RUTA_MODELO) -> Pipeline:
	return joblib.load(ruta)


def evaluar_validacion_cruzada(
	df_train: pd.DataFrame,
	cv: int = 3,
) -> np.ndarray:
	"""Devuelve la exactitud de cada fold."""
	return cross_val_score(
		crear_pipeline(),
		df_train["texto_limpio"],
		df_train["categoria"],
		cv=cv,
		scoring="accuracy",
	)


def figura_matriz(matriz, etiquetas, nombre: str):
	"""Dibuja la matriz de confusión y devuelve la figura (sin plt.show)."""
	fig, ax = plt.subplots(
		figsize=(max(6, len(etiquetas) * 1.1), max(4.5, len(etiquetas) * 0.9))
	)
	imagen = ax.imshow(matriz, interpolation="nearest", cmap="Blues")
	fig.colorbar(imagen, ax=ax)
	ax.set(
		xlabel="Categoría predicha",
		ylabel="Categoría real",
		title=f"Matriz de confusión: {nombre}",
	)
	ax.set_xticks(range(len(etiquetas)))
	ax.set_xticklabels(etiquetas, rotation=45, ha="right")
	ax.set_yticks(range(len(etiquetas)))
	ax.set_yticklabels(etiquetas)

	umbral = matriz.max() / 2
	for fila in range(matriz.shape[0]):
		for columna in range(matriz.shape[1]):
			ax.text(
				columna,
				fila,
				str(matriz[fila, columna]),
				ha="center",
				va="center",
				color="white" if matriz[fila, columna] > umbral else "black",
			)

	fig.tight_layout()
	return fig


def entrenar_y_evaluar(
	df: pd.DataFrame,
	test_size: float = 0.3,
	cv: int = 3,
) -> dict:
	"""
	Divide los datos, hace validación cruzada, evalúa en prueba y devuelve
	las métricas junto con el modelo final, reentrenado con todos los documentos.
	"""
	df_train, df_test = dividir_datos(df, test_size)
	puntuaciones = evaluar_validacion_cruzada(df_train, cv)

	modelo = crear_pipeline()
	modelo.fit(df_train["texto_limpio"], df_train["categoria"])
	y_pred = modelo.predict(df_test["texto_limpio"])

	y_real = df_test["categoria"]
	etiquetas = sorted(y_real.unique())
	reporte = pd.DataFrame(classification_report(
		y_real,
		y_pred,
		labels=etiquetas,
		target_names=etiquetas,
		zero_division=0,
		output_dict=True,
	)).T

	modelo_final = crear_pipeline()
	modelo_final.fit(df["texto_limpio"], df["categoria"])

	return {
		"total": len(df),
		"n_train": len(df_train),
		"n_test": len(df_test),
		"cv": cv,
		"cv_scores": puntuaciones,
		"exactitud": accuracy_score(y_real, y_pred),
		"reporte": reporte,
		"matriz": confusion_matrix(y_real, y_pred, labels=etiquetas),
		"etiquetas": etiquetas,
		"modelo": modelo_final,
	}


def clasificar_nuevos(
	modelo: Pipeline,
	df_nuevos: pd.DataFrame,
) -> pd.DataFrame:
	if df_nuevos.empty:
		return pd.DataFrame(
			columns=["archivo", "categoria_predicha", "confianza"]
		)

	textos = df_nuevos["texto_limpio"]
	return pd.DataFrame({
		"archivo": df_nuevos["archivo"].to_numpy(),
		"categoria_predicha": modelo.predict(textos),
		"confianza": modelo.predict_proba(textos).max(axis=1),
	})


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

def _obtener_modelo():
	"""Modelo de la sesión o, si existe, el guardado en models/."""
	if "clf_modelo" not in st.session_state and RUTA_MODELO.exists():
		st.session_state["clf_modelo"] = cargar_modelo()
	return st.session_state.get("clf_modelo")


def _seccion_entrenamiento() -> None:
	with st.expander("⚙️ Parámetros de entrenamiento", expanded=True):
		col1, col2, col3 = st.columns(3)
		test_size = col1.slider("Proporción de prueba", 0.1, 0.5, 0.3, 0.05)
		cv = col2.slider("Folds de validación cruzada", 2, 5, 3)
		guardar = col3.checkbox("Guardar modelo en models/", value=True)

		if st.button("🚀 Entrenar y evaluar", type="primary"):
			df = cargar_textos()
			if df.empty:
				st.error("No se encontraron documentos en data/documentos/.")
				return
			df["texto_limpio"] = df["texto"].apply(limpiar_texto)

			with st.spinner("Entrenando..."):
				try:
					resultados = entrenar_y_evaluar(df, test_size, cv)
				except ValueError as error:
					st.error(f"No se pudo entrenar: {error}")
					return

			st.session_state["clf_modelo"] = resultados.pop("modelo")
			resultados["ruta"] = (
				guardar_modelo(st.session_state["clf_modelo"]) if guardar else None
			)
			st.session_state["clf_resultados"] = resultados


def _seccion_resultados() -> None:
	r = st.session_state.get("clf_resultados")
	if not r:
		return

	st.subheader("Resultados")
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Documentos", r["total"])
	c2.metric("Entrenamiento / Prueba", f"{r['n_train']} / {r['n_test']}")
	c3.metric(
		f"Validación cruzada ({r['cv']} folds)",
		f"{r['cv_scores'].mean():.1%}",
		f"± {r['cv_scores'].std():.1%}",
		delta_color="off",
	)
	c4.metric("Exactitud en prueba", f"{r['exactitud']:.1%}")
	if r["ruta"]:
		st.success(f"Modelo guardado en: {r['ruta']}")

	izquierda, derecha = st.columns(2)
	with izquierda:
		st.markdown("**Reporte de clasificación**")
		st.dataframe(r["reporte"].round(2))
	with derecha:
		figura = figura_matriz(r["matriz"], r["etiquetas"], "Naive Bayes")
		st.pyplot(figura)
		plt.close(figura)


def _seccion_clasificar_nuevos() -> None:
	st.subheader("Clasificar documentos nuevos")
	modelo = _obtener_modelo()
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
				st.dataframe(resultado, hide_index=True)

	with tab_texto:
		texto = st.text_area("Texto a clasificar", height=160, key="clf_texto")
		if st.button("Clasificar texto") and texto.strip():
			limpio = limpiar_texto(texto)
			if not limpio:
				st.warning("El texto no tiene términos útiles.")
				return
			probabilidades = modelo.predict_proba([limpio])[0]
			mejor = modelo.classes_[int(np.argmax(probabilidades))]
			st.success(
				f"Categoría predicha: **{mejor}** "
				f"({probabilidades.max():.1%} de confianza)"
			)
			st.bar_chart(pd.Series(probabilidades, index=modelo.classes_, name="Probabilidad"))


def pagina_clasificador() -> None:
	st.header("🗂️ Clasificador de documentos")
	st.caption("TF-IDF + Naive Bayes sobre los documentos de data/documentos/.")
	_seccion_entrenamiento()
	_seccion_resultados()
	st.divider()
	_seccion_clasificar_nuevos()