from src.clasificador import dividir_datos, vectorizar_y_entrenar
from src.preprocesamiento import cargar_textos, limpiar_texto


def main() -> None:
	df = cargar_textos()
	df["texto_limpio"] = df["texto"].apply(limpiar_texto)

	print(f"Documentos cargados: {len(df)}")
	print("\nMuestra de textos limpios:")
	print(df[["texto_limpio", "categoria"]].head(10))

	df_train, df_test = dividir_datos(df)
	print(f"\nDocumentos de entrenamiento: {len(df_train)}")
	print(f"Documentos de prueba: {len(df_test)}")
	vectorizar_y_entrenar(df_train, df_test)


if __name__ == "__main__":
	main()