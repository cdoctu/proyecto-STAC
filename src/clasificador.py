from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

RUTA_MODELO = Path(__file__).resolve().parent.parent / "models" / "clasificador.joblib"


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
	print(f"\nModelo guardado en: {ruta}")
	return ruta


def cargar_modelo(ruta: Path = RUTA_MODELO) -> Pipeline:
	return joblib.load(ruta)


def evaluar_validacion_cruzada(
	df_train: pd.DataFrame,
	cv: int = 3,
) -> None:
	pipeline = crear_pipeline()
	puntuaciones = cross_val_score(
		pipeline,
		df_train["texto_limpio"],
		df_train["categoria"],
		cv=cv,
		scoring="accuracy",
	)

	print(f"\nValidación cruzada ({cv} folds):")
	print("Exactitud por fold:", [f"{puntuacion:.2%}" for puntuacion in puntuaciones])
	print(f"Promedio: {puntuaciones.mean():.2%} (+/- {puntuaciones.std():.2%})")


def mostrar_resultados(nombre: str, df_test: pd.DataFrame, y_pred) -> None:
	y_real = df_test["categoria"]
	etiquetas = sorted(y_real.unique())

	print(f"\nResultados: {nombre}")
	print(f"Exactitud: {accuracy_score(y_real, y_pred):.2%}")
	print("Reporte de clasificación:")
	print(classification_report(
		y_real,
		y_pred,
		labels=etiquetas,
		target_names=etiquetas,
		zero_division=0,
	))

	matriz = confusion_matrix(y_real, y_pred, labels=etiquetas)
	print("Matriz de confusión:")
	print(pd.DataFrame(matriz, index=etiquetas, columns=etiquetas))

	fig, ax = plt.subplots(
		figsize=(max(8, len(etiquetas) * 1.3), max(6, len(etiquetas)))
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
	plt.show()


def vectorizar_y_entrenar(
	df_train: pd.DataFrame,
	df_test: pd.DataFrame,
	nombre: str = "Naive Bayes",
) -> Pipeline:
	pipeline = crear_pipeline()

	pipeline.fit(df_train["texto_limpio"], df_train["categoria"])
	y_pred = pipeline.predict(df_test["texto_limpio"])
	mostrar_resultados(nombre, df_test, y_pred)
	return pipeline


def clasificar_nuevos(
	modelo: Pipeline,
	df_nuevos: pd.DataFrame,
) -> pd.DataFrame:
	if df_nuevos.empty:
		print("\nNo hay archivos nuevos para clasificar.")
		return pd.DataFrame(
			columns=["archivo", "categoria_predicha", "confianza"]
		)

	textos = df_nuevos["texto_limpio"]
	resultado = pd.DataFrame({
		"archivo": df_nuevos["archivo"].to_numpy(),
		"categoria_predicha": modelo.predict(textos),
		"confianza": modelo.predict_proba(textos).max(axis=1),
	})

	resultado_impreso = resultado.copy()
	resultado_impreso["confianza"] = resultado_impreso["confianza"].map(
		lambda confianza: f"{confianza:.2%}"
	)
	print("\nClasificación de documentos nuevos:")
	print(resultado_impreso.to_string(index=False))
	return resultado
