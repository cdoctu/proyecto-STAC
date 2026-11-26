import json
from datetime import datetime
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
RUTA_HISTORIAL = BASE / "data" / "historial.json"

# Si es True se usan embeddings de un modelo preentrenado en lugar de TF-IDF.
# Hay que instalar sentence-transformers y tener internet la primera vez, porque descarga el modelo.
USAR_EMBEDDINGS = False
MODELO_EMBEDDINGS = "paraphrase-multilingual-MiniLM-L12-v2"

# Si es True, cada palabra se reduce a su raíz quitando el plural y la última vocal.
# Así reporto y reporta, o contraseña y contraseñas, cuentan como la misma palabra.
# Solo se usa con TF-IDF.
USAR_RAIZ = True

# Cuánto pesa cada tipo de evento del historial. Leer un documento dice más
# sobre los intereses de una persona que solo buscarlo.
PESO_CONSULTA = 1.0
PESO_LECTURA = 1.5

# Cada evento pesa esta fracción de lo que pesa el evento que le sigue.
# Con 0.8, algo que se hizo hace tres pasos pesa cerca de la mitad que lo último.
OLVIDO = 0.8

# Peso de la consulta actual frente al historial. Con 0.6, la consulta actual
# aporta el 60 % del puntaje y el historial el 40 %.
PESO_CONSULTA_ACTUAL = 0.6

# Cuántos eventos recientes del historial se toman en cuenta.
MAX_EVENTOS = 20

NOMBRES_AREAS = {
	"recursos_humanos": "Recursos Humanos",
	"desarrollo_software": "Desarrollo de Software",
	"seguridad_informacion": "Seguridad de la Información",
	"soporte_clientes": "Soporte a Clientes",
	"gestion_proyectos": "Gestión de Proyectos",
}

# Historiales de prueba: el área que debería recomendarse y las consultas
# que hizo esa persona.
ESCENARIOS = [
	("recursos_humanos", ["¿Cuántos días de vacaciones me tocan?", "¿Cómo funciona el home office?"]),
	("desarrollo_software", ["¿Cómo se publica una nueva versión?", "¿Cómo se revisa el código?"]),
	("seguridad_informacion", ["¿Cómo reporto un correo sospechoso?", "¿Cada cuánto cambio mi contraseña?"]),
	("soporte_clientes", ["¿Cómo se atiende un ticket?", "¿Cuál es el tiempo de respuesta del SLA?"]),
	("gestion_proyectos", ["¿Cómo se hace una retrospectiva?", "¿Cómo se estiman los puntos de historia?"]),
]


def nombre_area(carpeta):
	return NOMBRES_AREAS.get(carpeta, carpeta.replace("_", " ").title())


def resumir(texto, largo=110):
	"""Devuelve el inicio del documento sin el título."""
	cuerpo = " ".join(texto.splitlines()[1:]).strip()
	if len(cuerpo) <= largo:
		return cuerpo
	return cuerpo[:largo].rsplit(" ", 1)[0] + "..."


def reducir_a_raiz(palabra):
	"""Quita el plural y la última vocal de una palabra. Por ejemplo, correos queda como corr."""
	if len(palabra) > 4 and palabra.endswith("es"):
		palabra = palabra[:-2]
	elif len(palabra) > 3 and palabra.endswith("s"):
		palabra = palabra[:-1]
	if len(palabra) > 3 and palabra[-1] in "aeiou":
		palabra = palabra[:-1]
	return palabra


def cargar_documentos():
	"""Lee los .txt de data/documentos. La carpeta de cada archivo es su categoría."""
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


class VectorizadorTfidf:
	"""Convierte textos en vectores TF-IDF usando la limpieza del proyecto."""

	def __init__(self, usar_raiz=USAR_RAIZ):
		self.usar_raiz = usar_raiz
		self.nombre = "TF-IDF con raíz" if usar_raiz else "TF-IDF"
		self.vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

	def _limpiar(self, textos):
		"""Aplica la limpieza del proyecto y, si se pidió, reduce cada palabra a su raíz."""
		limpios = []
		for texto in textos:
			palabras = limpiar_texto(texto).split()
			if self.usar_raiz:
				palabras = [reducir_a_raiz(p) for p in palabras]
			limpios.append(" ".join(palabras))
		return limpios

	def ajustar(self, textos):
		"""Aprende el vocabulario con los documentos y devuelve su matriz."""
		return self.vectorizador.fit_transform(self._limpiar(textos)).toarray()

	def transformar(self, textos):
		"""Convierte textos nuevos usando el vocabulario ya aprendido."""
		return self.vectorizador.transform(self._limpiar(textos)).toarray()


class VectorizadorEmbeddings:
	"""
	Convierte textos en embeddings con un modelo preentrenado.
	A diferencia de TF-IDF, reconoce sinónimos, por ejemplo computadora y laptop.
	"""

	nombre = "Embeddings"

	def __init__(self, nombre_modelo=MODELO_EMBEDDINGS):
		from sentence_transformers import SentenceTransformer
		self.modelo = SentenceTransformer(nombre_modelo)

	def ajustar(self, textos):
		return self.transformar(textos)

	def transformar(self, textos):
		# El modelo recibe el texto original, sin limpiar, porque entiende
		# mejor las frases completas.
		return np.asarray(self.modelo.encode(list(textos), normalize_embeddings=True))


class Recomendador:
	"""Recomienda documentos según su contenido y el historial de cada persona."""

	def __init__(self, vectorizador=None, ruta_historial=RUTA_HISTORIAL):
		self.documentos = cargar_documentos()
		self.vectorizador = vectorizador or VectorizadorTfidf()
		self.matriz = self.vectorizador.ajustar(self.documentos["texto"])
		self.ruta_historial = ruta_historial
		self.historial = self._leer_historial()

	# Historial

	def _leer_historial(self):
		if self.ruta_historial is None or not Path(self.ruta_historial).exists():
			return {}
		try:
			return json.loads(Path(self.ruta_historial).read_text(encoding="utf-8"))
		except json.JSONDecodeError:
			return {}

	def _guardar_historial(self):
		if self.ruta_historial is None:
			return
		ruta = Path(self.ruta_historial)
		ruta.parent.mkdir(parents=True, exist_ok=True)
		ruta.write_text(json.dumps(self.historial, ensure_ascii=False, indent=2), encoding="utf-8")

	def _agregar(self, usuario, tipo, valor):
		evento = {"tipo": tipo, "valor": valor, "fecha": datetime.now().strftime("%Y-%m-%d %H:%M")}
		self.historial.setdefault(usuario, []).append(evento)
		self._guardar_historial()

	def registrar_consulta(self, usuario, consulta):
		if consulta.strip():
			self._agregar(usuario, "consulta", consulta.strip())

	def registrar_lectura(self, usuario, id_documento):
		self._agregar(usuario, "lectura", id_documento)

	def borrar_historial(self, usuario):
		self.historial.pop(usuario, None)
		self._guardar_historial()

	def titulo(self, id_documento):
		fila = self.documentos[self.documentos["id"] == id_documento]
		return fila["titulo"].iloc[0] if len(fila) else id_documento

	# Vectores y puntajes

	def _similitud(self, vector):
		"""Similitud coseno entre un vector y todos los documentos."""
		return cosine_similarity(vector.reshape(1, -1), self.matriz).flatten()

	def _aplicar_area(self, puntajes, area):
		"""Si se pide un área, los documentos de las otras áreas quedan descartados."""
		if area:
			return np.where(self.documentos["categoria"].values == area, puntajes, -1)
		return puntajes

	def _tabla(self, puntajes, k):
		"""Arma la tabla con los k documentos de mayor puntaje. Los puntajes de 0 o menos se descartan."""
		orden = [i for i in np.argsort(puntajes)[::-1][:k] if puntajes[i] > 0]
		tabla = self.documentos.iloc[orden][["id", "titulo", "categoria", "resumen"]].copy()
		tabla.insert(0, "posicion", range(1, len(tabla) + 1))
		tabla["puntaje"] = puntajes[orden].round(4)
		return tabla.reset_index(drop=True)

	def _eventos_con_vector(self, usuario):
		"""
		Devuelve los eventos recientes del historial con su vector y su peso.
		Los eventos más nuevos pesan más y las lecturas pesan más que las consultas.
		"""
		eventos = self.historial.get(usuario, [])[-MAX_EVENTOS:]
		resultado = []
		for posicion, evento in enumerate(eventos):
			if evento["tipo"] == "lectura":
				fila = self.documentos.index[self.documentos["id"] == evento["valor"]]
				if len(fila) == 0:
					continue
				vector = self.matriz[fila[0]]
				descripcion = f"se parece al documento que leíste: {self.titulo(evento['valor'])}"
				peso = PESO_LECTURA
			else:
				vector = self.vectorizador.transformar([evento["valor"]])[0]
				descripcion = f"se parece a tu consulta anterior: {evento['valor']}"
				peso = PESO_CONSULTA
			if not vector.any():
				continue
			antiguedad = len(eventos) - 1 - posicion
			resultado.append({"descripcion": descripcion, "vector": vector, "peso": peso * OLVIDO ** antiguedad})
		return resultado

	def _promediar(self, eventos):
		"""Resume los eventos en un solo vector, el perfil de la persona."""
		if not eventos:
			return None
		vectores = np.vstack([e["vector"] for e in eventos])
		promedio = np.average(vectores, axis=0, weights=[e["peso"] for e in eventos])
		norma = np.linalg.norm(promedio)
		return promedio / norma if norma > 0 else None

	def _motivo(self, id_documento, fuentes):
		"""Dice cuál elemento del contexto se parece más al documento recomendado."""
		if not fuentes:
			return ""
		fila = self.documentos.index[self.documentos["id"] == id_documento][0]
		similitudes = [
			cosine_similarity(f["vector"].reshape(1, -1), self.matriz[fila].reshape(1, -1))[0, 0]
			for f in fuentes
		]
		return fuentes[int(np.argmax(similitudes))]["descripcion"]

	# Búsqueda y recomendación

	def buscar(self, consulta, k=3, area=None):
		"""Busca documentos por su similitud con la consulta, sin usar el historial."""
		vector = self.vectorizador.transformar([consulta])[0]
		if not vector.any():
			return self._tabla(np.zeros(len(self.documentos)), k)
		puntajes = self._aplicar_area(self._similitud(vector), area)
		return self._tabla(puntajes, k)

	def similares(self, id_documento, k=3):
		"""Devuelve los documentos más parecidos a uno dado, sin contar el mismo documento."""
		fila = self.documentos.index[self.documentos["id"] == id_documento]
		if len(fila) == 0:
			return self._tabla(np.zeros(len(self.documentos)), k)
		puntajes = self._similitud(self.matriz[fila[0]])
		puntajes[fila[0]] = -1
		return self._tabla(puntajes, k)

	def recomendar(self, usuario, consulta=None, k=3, excluir=(), area=None):
		"""
		Recomienda documentos mezclando la consulta actual (si hay) con el perfil
		de la persona. No recomienda documentos que ya leyó ni los de la lista excluir.
		Agrega la columna motivo para explicar cada recomendación.
		"""
		eventos = self._eventos_con_vector(usuario)
		perfil = self._promediar(eventos)

		vector_consulta = None
		if consulta:
			vector_consulta = self.vectorizador.transformar([consulta])[0]
			if not vector_consulta.any():
				vector_consulta = None

		if vector_consulta is None and perfil is None:
			tabla = self._tabla(np.zeros(len(self.documentos)), k)
			tabla["motivo"] = []
			return tabla

		if vector_consulta is not None and perfil is not None:
			puntajes = (PESO_CONSULTA_ACTUAL * self._similitud(vector_consulta)
						+ (1 - PESO_CONSULTA_ACTUAL) * self._similitud(perfil))
		elif vector_consulta is not None:
			puntajes = self._similitud(vector_consulta)
		else:
			puntajes = self._similitud(perfil)

		leidos = {e["valor"] for e in self.historial.get(usuario, []) if e["tipo"] == "lectura"}
		ocultos = set(excluir) | leidos
		puntajes[self.documentos["id"].isin(ocultos).values] = -1
		puntajes = self._aplicar_area(puntajes, area)

		fuentes = list(eventos)
		if vector_consulta is not None:
			fuentes.insert(0, {"descripcion": "responde a tu consulta actual", "vector": vector_consulta})

		tabla = self._tabla(puntajes, k)
		tabla["motivo"] = [self._motivo(id_doc, fuentes) for id_doc in tabla["id"]]
		return tabla

	# Evaluación

	def evaluar_similares(self, k=3):
		"""
		Mide qué parte de los k documentos parecidos pertenece a la misma área que
		el documento de origen. Devuelve el promedio por área.
		"""
		filas = []
		for _, doc in self.documentos.iterrows():
			tabla = self.similares(doc["id"], k)
			filas.append({"categoria": doc["categoria"],
						  "precision": (tabla["categoria"] == doc["categoria"]).sum() / k})
		return pd.DataFrame(filas).groupby("categoria", as_index=False)["precision"].mean()

	def evaluar_historial(self, k=3):
		"""
		Para cada historial de prueba, mide qué parte de las k recomendaciones
		es del área esperada. No toca el historial guardado.
		"""
		filas = []
		try:
			for area, consultas in ESCENARIOS:
				self.historial["_prueba"] = [{"tipo": "consulta", "valor": c, "fecha": ""} for c in consultas]
				tabla = self.recomendar("_prueba", k=k)
				filas.append({
					"area_esperada": area,
					"consultas": " | ".join(consultas),
					"recomendados": " | ".join(tabla["titulo"]),
					"precision": (tabla["categoria"] == area).sum() / k,
				})
		finally:
			self.historial.pop("_prueba", None)
		return pd.DataFrame(filas)


def crear_recomendador():
	"""Crea el recomendador con TF-IDF, o con embeddings si se activaron y funcionan."""
	if USAR_EMBEDDINGS:
		try:
			return Recomendador(VectorizadorEmbeddings())
		except Exception as error:
			print(f"No se pudieron usar los embeddings ({error}). Se usará TF-IDF.")
	return Recomendador()


# ---------------------------------------------------------------------------
# Interfaz Streamlit
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Preparando el recomendador...")
def recomendador_cacheado():
	"""El recomendador se crea una sola vez, no en cada interacción."""
	return crear_recomendador()


def texto_sin_titulo(texto):
	"""Documento sin la primera línea (el título), listo para mostrar en Markdown."""
	return "\n".join(texto.splitlines()[1:]).strip().replace("\n", "  \n")


def usuario_actual():
	return st.session_state.get("rec_usuario", "").strip() or "invitado"


def _al_cambiar_usuario():
	st.session_state.pop("rec_busqueda", None)
	st.session_state["rec_leyendo"] = None


def abrir_documento(id_documento):
	"""Callback de los botones Leer: anota la lectura y abre el documento."""
	recomendador_cacheado().registrar_lectura(usuario_actual(), id_documento)
	st.session_state["rec_leyendo"] = id_documento


def cerrar_documento():
	st.session_state["rec_leyendo"] = None


def mostrar_lista(tabla, prefijo, etiqueta="Puntaje"):
	"""Lista de documentos con un botón Leer. El prefijo evita llaves repetidas."""
	if tabla.empty:
		st.info("No hay documentos para mostrar.")
		return
	for _, fila in tabla.iterrows():
		with st.container(border=True):
			col_texto, col_boton = st.columns([6, 1])
			col_texto.markdown(f"**{fila['titulo']}**")
			col_texto.caption(f"Área: {nombre_area(fila['categoria'])}  ·  {etiqueta}: {fila['puntaje']}")
			col_texto.write(fila["resumen"])
			if "motivo" in tabla.columns and fila["motivo"]:
				col_texto.caption(f"💡 Motivo: {fila['motivo']}")
			col_boton.button(
				"📖 Leer",
				key=f"{prefijo}_{fila['id']}",
				on_click=abrir_documento,
				args=(fila["id"],),
			)


def mostrar_lector(recomendador):
	"""Documento abierto, con sugerencias de otros parecidos."""
	id_documento = st.session_state.get("rec_leyendo")
	if not id_documento:
		return
	coincidencias = recomendador.documentos[recomendador.documentos["id"] == id_documento]
	if coincidencias.empty:
		return
	fila = coincidencias.iloc[0]

	with st.container(border=True):
		st.subheader(fila["titulo"])
		st.caption(f"Área: {nombre_area(fila['categoria'])}  ·  Id: {fila['id']}")
		st.markdown(texto_sin_titulo(fila["texto"]))
		st.markdown("**Documentos parecidos a este**")
		mostrar_lista(recomendador.similares(id_documento, 3), "sim", "Similitud")
		st.button("✖ Cerrar documento", on_click=cerrar_documento)
	st.divider()


def _modo_buscar(recomendador, usuario):
	with st.form("rec_form_busqueda"):
		consulta = st.text_input("Escribe tu pregunta con tus propias palabras")
		enviado = st.form_submit_button("Buscar", type="primary")

	if enviado and consulta.strip():
		consulta = consulta.strip()
		tiene_historial = bool(recomendador.historial.get(usuario))
		resultados = recomendador.buscar(consulta, 3)
		recomendados = recomendador.recomendar(usuario, consulta, 3, excluir=resultados["id"])
		recomendador.registrar_consulta(usuario, consulta)
		st.session_state["rec_busqueda"] = (consulta, resultados, recomendados, tiene_historial)

	guardado = st.session_state.get("rec_busqueda")
	if not guardado:
		return
	consulta, resultados, recomendados, tiene_historial = guardado
	st.markdown(f"#### Resultados para: {consulta}")
	mostrar_lista(resultados, "res", "Similitud")
	if tiene_historial:
		st.markdown("#### También te pueden servir según tu historial")
		mostrar_lista(recomendados, "reco")
	else:
		st.info("Cuando tengas historial, aquí verás recomendaciones pensadas para ti.")


def _modo_segun_historial(recomendador, usuario):
	tabla = recomendador.recomendar(usuario, k=3)
	if tabla.empty:
		st.info("Todavía no hay historial para recomendar. Haz una búsqueda primero.")
	else:
		mostrar_lista(tabla, "his")


def _modo_explorar(recomendador):
	for categoria, grupo in recomendador.documentos.groupby("categoria"):
		with st.expander(nombre_area(categoria)):
			for _, fila in grupo.iterrows():
				col_titulo, col_boton = st.columns([6, 1])
				col_titulo.write(fila["titulo"])
				col_boton.button(
					"📖 Leer",
					key=f"exp_{fila['id']}",
					on_click=abrir_documento,
					args=(fila["id"],),
				)


def _modo_ver_historial(recomendador, usuario):
	eventos = recomendador.historial.get(usuario, [])
	if not eventos:
		st.info("Tu historial está vacío.")
		return

	filas = [{
		"Fecha": e["fecha"],
		"Tipo": "Lectura" if e["tipo"] == "lectura" else "Consulta",
		"Detalle": recomendador.titulo(e["valor"]) if e["tipo"] == "lectura" else e["valor"],
	} for e in reversed(eventos)]
	st.dataframe(pd.DataFrame(filas), hide_index=True)

	st.divider()
	confirmar = st.checkbox("Confirmo que quiero borrar mi historial")
	if st.button("🗑️ Borrar mi historial", disabled=not confirmar):
		recomendador.borrar_historial(usuario)
		st.session_state.pop("rec_busqueda", None)
		st.rerun()


def _modo_evaluar(recomendador):
	st.write("Mide qué tan bien recomienda el sistema (no toca tu historial guardado).")
	if not st.button("▶️ Ejecutar evaluación"):
		return

	with st.spinner("Evaluando..."):
		similares = recomendador.evaluar_similares(3)
		historial = recomendador.evaluar_historial(3)
		CARPETA_RESULTADOS.mkdir(exist_ok=True)
		similares.to_csv(CARPETA_RESULTADOS / "recomendador_similares.csv", index=False, encoding="utf-8")
		historial.to_csv(CARPETA_RESULTADOS / "recomendador_historial.csv", index=False, encoding="utf-8")

	st.markdown("**Documentos parecidos** — % de los 3 recomendados de la misma área que el documento de origen")
	st.dataframe(pd.DataFrame({
		"Área": similares["categoria"].map(nombre_area),
		"Precisión": similares["precision"].map("{:.0%}".format),
	}), hide_index=True)
	st.metric("Promedio", f"{similares['precision'].mean():.0%}")

	st.markdown("**Recomendaciones según el historial** — % de los 3 recomendados del área esperada")
	st.dataframe(pd.DataFrame({
		"Área esperada": historial["area_esperada"].map(nombre_area),
		"Consultas": historial["consultas"],
		"Recomendados": historial["recomendados"],
		"Precisión": historial["precision"].map("{:.0%}".format),
	}), hide_index=True)
	st.metric("Promedio", f"{historial['precision'].mean():.0%}")

	st.caption(f"Tablas guardadas en: {CARPETA_RESULTADOS}")


def pagina_recomendador():
	st.header("⭐ Recomendador de documentos")
	recomendador = recomendador_cacheado()

	col_nombre, col_info = st.columns([2, 3])
	col_nombre.text_input(
		"Tu nombre (para guardar tu historial)",
		key="rec_usuario",
		placeholder="invitado",
		on_change=_al_cambiar_usuario,
	)
	usuario = usuario_actual()
	total = len(recomendador.historial.get(usuario, []))
	col_info.markdown(
		f"<br>Método: **{recomendador.vectorizador.nombre}** · Persona: **{usuario}** · "
		f"Historial: **{total}** eventos",
		unsafe_allow_html=True,
	)

	mostrar_lector(recomendador)

	modo = st.radio(
		"Qué quieres hacer",
		["Buscar y recibir recomendaciones", "Según mi historial", "Explorar documentos",
		 "Mi historial", "Probar el recomendador"],
		horizontal=True,
	)

	if modo == "Buscar y recibir recomendaciones":
		_modo_buscar(recomendador, usuario)
	elif modo == "Según mi historial":
		_modo_segun_historial(recomendador, usuario)
	elif modo == "Explorar documentos":
		_modo_explorar(recomendador)
	elif modo == "Mi historial":
		_modo_ver_historial(recomendador, usuario)
	else:
		_modo_evaluar(recomendador)