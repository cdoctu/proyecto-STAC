import pandas as pd
from pathlib import Path
import re
import unicodedata
from sklearn.model_selection import train_test_split

#Limpieza de texto y stopwords en español
STOPWORDS_ES = {
	"a", "al", "algo", "algunas", "algunos", "ante", "antes", "como",
	"con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
	"durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era",
	"erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso",
	"esos", "esta", "estaba", "estaban", "estado", "estais", "estamos",
	"estan", "estar", "estas", "este", "esto", "estos", "fue", "fueron",
	"ha", "habeis", "han", "has", "hasta", "hay", "la", "las", "le", "les",
	"lo", "los", "mas", "me", "mi", "mis", "mucho", "muy", "nada", "ni",
	"no", "nos", "nosotros", "o", "os", "otra", "otro", "para", "pero",
	"poco", "por", "porque", "que", "quien", "se", "sea", "ser", "si", "sin",
	"sobre", "son", "su", "sus", "te", "tiene", "todo", "tu", "tus", "un",
	"una", "uno", "unos", "vosotros", "y", "ya",
}


def limpiar_texto(texto: str) -> str:
	"""Normaliza un texto y elimina signos, numeros y stopwords en espanol."""
	texto_protegido = texto.lower().replace("ñ", "\uE000")
	texto_normalizado = unicodedata.normalize("NFD", texto_protegido)
	texto_sin_acentos = "".join(
		caracter
		for caracter in texto_normalizado
		if unicodedata.category(caracter) != "Mn"
	).replace("\uE000", "ñ")
	tokens = re.sub(r"[^a-zñ\s]", " ", texto_sin_acentos).split()
	return " ".join(token for token in tokens if token not in STOPWORDS_ES)

#Carga de Documentos
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

#División de datos para entrenamiento y prueba
def dividir_datos(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df["categoria"])

#Flujo principal
df = cargar_textos()
print("Carga de Documentos:")
print(df.head(10))
df["texto_limpio"] = df["texto"].apply(limpiar_texto)
print("\nTextos Limpios:")
print(df[["texto_limpio", "categoria"]].head(10))
df_train, df_test = dividir_datos(df)
print("\nDatos de Entrenamiento:")
print(df_train.head(10))   
print("\nDatos de Prueba:")
print(df_test.head(10))   

