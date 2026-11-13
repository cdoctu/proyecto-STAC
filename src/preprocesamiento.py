import pandas as pd
from pathlib import Path


def cargar_textos() -> pd.DataFrame:
	directorio_data = Path(__file__).resolve().parent.parent / "data"
	registros = [
		{
			"texto": archivo.read_text(encoding="utf-8"),
			"categoria": archivo.parent.name,
		}
		for archivo in sorted(directorio_data.rglob("*.txt"))
	]

	return pd.DataFrame(registros, columns=["texto", "categoria"])

df = cargar_textos()
print(df.head(10))

