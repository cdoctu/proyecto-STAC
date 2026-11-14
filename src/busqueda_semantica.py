"""
busqueda_semantica.py
Busca los documentos más parecidos a una pregunta escrita en lenguaje natural,
usando TF-IDF y similitud coseno.

Formas de usarlo (desde la carpeta principal del proyecto):
    python src/busqueda_semantica.py "como reporto un correo sospechoso"
    python src/busqueda_semantica.py "como reporto un correo sospechoso" -k 5 --raiz
    python src/busqueda_semantica.py            (modo interactivo, escribes varias preguntas)
    python src/busqueda_semantica.py --probar   (corre las consultas de prueba)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Se agrega la carpeta principal al path para poder importar "src.preprocesamiento" aunque el script se corra como "python src/busqueda_semantica.py".
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.preprocesamiento import limpiar_texto 

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



# Limpieza y reducción opcional a la raíz
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



# indexa, limpia todos los documentos y aplicacion de fit_transform
def cargar_documentos():
    filas = []
    for archivo in sorted(CARPETA_DOCS.rglob("*.txt")):
        texto = archivo.read_text(encoding="utf-8")
        filas.append({
            "id": archivo.stem,
            "titulo": texto.splitlines()[0],       
            "categoria": archivo.parent.name,     
            "texto": texto,
        })
    return pd.DataFrame(filas)


def crear_indice(usar_raiz=False):
    df = cargar_documentos()
    df["texto_limpio"] = df["texto"].apply(lambda t: limpiar(t, usar_raiz))

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matriz = vectorizador.fit_transform(df["texto_limpio"]) 
    return df, vectorizador, matriz


# Vectorizar la consulta, comparar y ordenar
def buscar(consulta, indice, k=3, usar_raiz=False):
    df, vectorizador, matriz = indice

    consulta_limpia = limpiar(consulta, usar_raiz)              
    vector_consulta = vectorizador.transform([consulta_limpia])  

    similitudes = cosine_similarity(vector_consulta, matriz).flatten()  # valor por documento
    orden = np.argsort(similitudes)[::-1][:k]                           # k = mas parecidos

    resultado = pd.DataFrame({
        "posicion": range(1, len(orden) + 1),
        "id": df["id"].iloc[orden].values,
        "titulo": df["titulo"].iloc[orden].values,
        "categoria": df["categoria"].iloc[orden].values,
        "similitud": similitudes[orden].round(4),
    })
    # Si la similitud es 0, el documento no comparte ninguna palabra con la consulta
    return resultado[resultado["similitud"] > 0].reset_index(drop=True)


def mostrar(consulta, resultado):
    print(f"\nConsulta: {consulta}")
    if resultado.empty:
        print("  No se encontraron documentos con palabras en común.")
    else:
        print(resultado.to_string(index=False))


# Prueba para varias consultas
def probar(k=3):
    indice_normal = crear_indice(usar_raiz=False)
    indice_raiz = crear_indice(usar_raiz=True)

    ids_existentes = set(indice_normal[0]["id"])
    for _, esperado in PRUEBAS:
        if esperado not in ids_existentes:
            print(f"AVISO: el documento esperado '{esperado}' no existe en data/documentos/. "
                  "Cambia la lista PRUEBAS al inicio de este archivo.")

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
    for _, f in tabla.iterrows():
        print(f"\nConsulta: {f['consulta']}")
        print(f"  Esperado : {f['esperado']}")
        print(f"  Sin raíz : {f['primero_sin_raiz']} (similitud {f['similitud_sin_raiz']})"
              f" -> {'ACIERTO' if f['acierto_sin_raiz'] else 'FALLO'}")
        print(f"  Con raíz : {f['primero_con_raiz']} (similitud {f['similitud_con_raiz']})"
              f" -> {'ACIERTO' if f['acierto_con_raiz'] else 'FALLO'}")

    print(f"\nAciertos en el primer lugar: sin raíz {tabla['acierto_sin_raiz'].sum()}/{len(tabla)}"
          f" | con raíz {tabla['acierto_con_raiz'].sum()}/{len(tabla)}")

    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    tabla.to_csv(CARPETA_RESULTADOS / "consultas_prueba.csv", index=False, encoding="utf-8")
    print(f"Tabla guardada en: {CARPETA_RESULTADOS / 'consultas_prueba.csv'}")


# Recibir la consulta 
def main():
    parser = argparse.ArgumentParser(description="Búsqueda semántica en los documentos de NovaTech")
    parser.add_argument("consulta", nargs="*", help="pregunta en lenguaje natural")
    parser.add_argument("-k", type=int, default=3, help="cuántos resultados mostrar (por defecto 3)")
    parser.add_argument("--raiz", action="store_true", help="reducir las palabras a su raíz")
    parser.add_argument("--probar", action="store_true", help="correr las consultas de prueba")
    args = parser.parse_args()

    if args.probar:
        probar(args.k)
        return

    indice = crear_indice(usar_raiz=args.raiz)

    if args.consulta:
        consulta = " ".join(args.consulta)
        mostrar(consulta, buscar(consulta, indice, args.k, args.raiz))
    else:
        print("Búsqueda en documentos de NovaTech. Escribe tu pregunta (o 'salir' para terminar).")
        while True:
            consulta = input("\n> ").strip()
            if consulta.lower() in ("salir", "exit", ""):
                break
            mostrar(consulta, buscar(consulta, indice, args.k, args.raiz))


if __name__ == "__main__":
    main()
