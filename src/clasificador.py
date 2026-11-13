import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

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
	pipeline = Pipeline([
		("tfidf", TfidfVectorizer()),
		("classifier", MultinomialNB()),
	])

	pipeline.fit(df_train["texto_limpio"], df_train["categoria"])
	y_pred = pipeline.predict(df_test["texto_limpio"])
	mostrar_resultados(nombre, df_test, y_pred)
	return pipeline
