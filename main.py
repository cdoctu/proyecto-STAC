from src.busqueda_semantica import menu as menu_busqueda
from src.clasificador import (
	dividir_datos,
	evaluar_validacion_cruzada,
	clasificar_nuevos,
	guardar_modelo,
	vectorizar_y_entrenar,
)
from src.preprocesamiento import cargar_nuevos, cargar_textos, limpiar_texto


def ejecutar_clasificador() -> None:
	df = cargar_textos()
	df["texto_limpio"] = df["texto"].apply(limpiar_texto)

	print(f"Documentos cargados: {len(df)}")
	print("\nMuestra de textos limpios:")
	print(df[["texto_limpio", "categoria"]].head(10))

	df_train, df_test = dividir_datos(df)
	print(f"\nDocumentos de entrenamiento: {len(df_train)}")
	print(f"Documentos de prueba: {len(df_test)}")
	evaluar_validacion_cruzada(df_train)
	modelo = vectorizar_y_entrenar(df_train, df_test)
	modelo.fit(df["texto_limpio"], df["categoria"])
	guardar_modelo(modelo)

	df_nuevos = cargar_nuevos()
	df_nuevos["texto_limpio"] = df_nuevos["texto"].apply(limpiar_texto)
	clasificar_nuevos(modelo, df_nuevos)


def ejecutar_buscador() -> None:
	menu_busqueda(desde_main=True)


def main() -> None:
	try:
		while True:
			print("\n" + "=" * 58)
			print("  PROYECTO STAC · NovaTech Soluciones")
			print("=" * 58)
			print("  1. Clasificador de documentos")
			print("  2. Buscador de documentos (búsqueda semántica)")
			print("  3. Ejecutar todo (clasificador y luego buscador)")
			print("  0. Salir")
			opcion = input("\nElige una opción: ").strip()

			if opcion == "1":
				ejecutar_clasificador()
			elif opcion == "2":
				ejecutar_buscador()
			elif opcion == "3":
				ejecutar_clasificador()
				ejecutar_buscador()
			elif opcion == "0":
				break
			else:
				print("Opción no válida.")
	except (EOFError, KeyboardInterrupt):
		pass
	print("\nHasta luego.")


if __name__ == "__main__":
	main()