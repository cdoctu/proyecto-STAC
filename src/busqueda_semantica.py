import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


# Nombres para mostrar
NOMBRES_AREAS = {
    "recursos_humanos": "Recursos Humanos",
    "desarrollo_software": "Desarrollo de Software",
    "seguridad_informacion": "Seguridad de la Información",
    "soporte_clientes": "Soporte a Clientes",
    "gestion_proyectos": "Gestión de Proyectos",
}

# Preguntas de ejemplo 
EJEMPLOS_POR_AREA = {
    "recursos_humanos": ["¿Cómo pido mis vacaciones?", "¿Cuántos días de home office tengo?"],
    "desarrollo_software": ["¿Cómo se publica una nueva versión?", "¿Cómo se revisa el código de un compañero?"],
    "seguridad_informacion": ["¿Cómo reporto un correo sospechoso?", "¿Cada cuánto se cambian las contraseñas?"],
    "soporte_clientes": ["¿Cómo se atiende un ticket?", "¿Cuáles son los tiempos del SLA?"],
    "gestion_proyectos": ["¿Cómo se hace una retrospectiva?", "¿Qué hago si el cliente pide un cambio?"],
}


def nombre_area(carpeta):
    return NOMBRES_AREAS.get(carpeta, carpeta.replace("_", " ").title())


def resumir(texto, largo=110):
    """Primeras palabras del documento (sin el título), para mostrar en los resultados."""
    cuerpo = " ".join(texto.splitlines()[1:]).strip()
    if len(cuerpo) <= largo:
        return cuerpo
    return cuerpo[:largo].rsplit(" ", 1)[0] + "..."


# Limpieza

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


# 1. Indexar: limpiar todos los documentos y aplicar fit_transform
def cargar_documentos():
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


def crear_indice(usar_raiz=False):
    df = cargar_documentos()
    df["texto_limpio"] = df["texto"].apply(lambda t: limpiar(t, usar_raiz))

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    matriz = vectorizador.fit_transform(df["texto_limpio"])  # matriz documentos x términos
    return df, vectorizador, matriz


# Vectorizar la consulta, comparar y ordenar
def buscar(consulta, indice, k=3, usar_raiz=False, area=None):
    df, vectorizador, matriz = indice

    consulta_limpia = limpiar(consulta, usar_raiz)               
    vector_consulta = vectorizador.transform([consulta_limpia])  

    similitudes = cosine_similarity(vector_consulta, matriz).flatten()  
    if area:  
        similitudes = np.where(df["categoria"].values == area, similitudes, 0)
    orden = np.argsort(similitudes)[::-1][:k]                          

    resultado = pd.DataFrame({
        "posicion": range(1, len(orden) + 1),
        "id": df["id"].iloc[orden].values,
        "titulo": df["titulo"].iloc[orden].values,
        "categoria": df["categoria"].iloc[orden].values,
        "similitud": similitudes[orden].round(4),
    })
    return resultado[resultado["similitud"] > 0].reset_index(drop=True)


def mostrar(consulta, resultado):
    print(f"\nConsulta: {consulta}")
    if resultado.empty:
        print("  No se encontraron documentos con palabras en común.")
    else:
        print(resultado.to_string(index=False))


# Prueba con varias consultas
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


# Detalles técnicos: por qué salió cada resultado
def explicar(consulta, indice, resultado, usar_raiz=False, max_terminos=5):
    """Muestra qué palabras coincidieron y cuánto aportó cada una a la similitud."""
    df, vectorizador, matriz = indice
    terminos = vectorizador.get_feature_names_out()

    consulta_limpia = limpiar(consulta, usar_raiz)
    vector_consulta = vectorizador.transform([consulta_limpia])

    sin_vocabulario = [p for p in consulta_limpia.split() if p not in vectorizador.vocabulary_]

    print(f"\nConsulta limpia: {consulta_limpia if consulta_limpia else '(vacía)'}")
    if sin_vocabulario:
        print(f"Palabras que no aparecen en ningún documento (se ignoran): {', '.join(sin_vocabulario)}")
    if resultado.empty:
        print("Ningún documento comparte palabras con la consulta, por eso no hay resultados.")
        return

    for _, fila in resultado.iterrows():
        numero_doc = df.index[df["id"] == fila["id"]][0]
        aportes = vector_consulta.multiply(matriz[numero_doc]).toarray().flatten()
        mejores = np.argsort(aportes)[::-1][:max_terminos]

        print(f"\n#{fila['posicion']}  {fila['id']}  (similitud {fila['similitud']})")
        for i in mejores:
            if aportes[i] > 0:
                print(f"    {terminos[i]:<26} aporta {aportes[i]:.4f}")
        print(f"    Suma de todas las palabras en común = {aportes.sum():.4f}")


# Búsqueda pensada para las personas de las áreas
def mostrar_amigable(consulta, resultado, indice, area=None):
    """Muestra los resultados como una lista fácil de leer."""
    df = indice[0]
    donde = nombre_area(area) if area else "todas las áreas"
    print(f'\nResultados para "{consulta}" en {donde}:')
    if resultado.empty:
        print("\n  No encontré documentos con esas palabras.")
        print("  Tip: el buscador compara palabras. Prueba con otras palabras o con una sola palabra clave.")
        return
    print("  (La similitud va de 0, nada en común, a 1, idéntico.)")
    resumenes = df.set_index("id")["resumen"]
    for _, fila in resultado.iterrows():
        print(f"\n  {fila['posicion']}. {fila['titulo']}")
        print(f"     Área: {nombre_area(fila['categoria'])}  |  Similitud: {fila['similitud']}")
        print(f"     {resumenes[fila['id']]}")


def leer_documento(indice, referencia, resultado):
    """La referencia puede ser la posición (1, 2, 3...) del último resultado o el id del documento."""
    df = indice[0]
    if referencia.isdigit() and not resultado.empty and 1 <= int(referencia) <= len(resultado):
        referencia = resultado["id"].iloc[int(referencia) - 1]

    fila = df[df["id"] == referencia]
    if fila.empty:
        print("  No encontré ese documento.")
        return
    fila = fila.iloc[0]
    print(f"\n{fila['titulo']}")
    print(f"Área: {nombre_area(fila['categoria'])}  |  Id: {fila['id']}\n")
    print("\n".join(fila["texto"].splitlines()[1:]).strip())


def elegir_area(indice):
    """Muestra las áreas numeradas y devuelve la que elija la persona (o None si cancela)."""
    areas = sorted(indice[0]["categoria"].unique())
    print("\nElige tu área:")
    for i, area in enumerate(areas, 1):
        print(f"  {i}. {nombre_area(area)}")
    print("  0. Volver")
    opcion = input("\nNúmero de área: ").strip()
    if opcion.isdigit() and 1 <= int(opcion) <= len(areas):
        return areas[int(opcion) - 1]
    return None


def buscar_y_mostrar(e, area=None):
    """Pide una pregunta, busca (en todas las áreas o en una) y deja leer un resultado."""
    if area:
        print(f"\nBuscando solo en {nombre_area(area)}.")
        print("Ejemplos de preguntas:")
        for ejemplo in EJEMPLOS_POR_AREA.get(area, []):
            print(f"  - {ejemplo}")
    consulta = input("\nEscribe tu pregunta con tus propias palabras: ").strip()
    if not consulta:
        return

    resultado = buscar(consulta, e["indice"], e["k"], e["raiz"], area)
    e["consulta"], e["resultado"] = consulta, resultado
    mostrar_amigable(consulta, resultado, e["indice"], area)

    if not resultado.empty:
        ref = input("\nEscribe el número de un resultado para leerlo (Enter para volver): ").strip()
        if ref:
            leer_documento(e["indice"], ref, resultado)


def explorar_area(indice):
    """Lista los documentos de un área para leerlos sin necesidad de buscar."""
    area = elegir_area(indice)
    if area is None:
        return
    df = indice[0]
    grupo = df[df["categoria"] == area].reset_index(drop=True)
    print(f"\nDocumentos de {nombre_area(area)}:")
    for i, fila in grupo.iterrows():
        print(f"  {i + 1}. {fila['titulo']}")
    ref = input("\nEscribe el número de un documento para leerlo (Enter para volver): ").strip()
    if ref.isdigit() and 1 <= int(ref) <= len(grupo):
        leer_documento(indice, grupo["id"].iloc[int(ref) - 1], pd.DataFrame())


# Menús
def menu_avanzado(e):
    """Opciones técnicas, útiles para el reporte."""
    while True:
        print("\n--- Opciones avanzadas ---")
        print(f"  Resultados a mostrar: {e['k']} | Raíz de palabras: {'sí' if e['raiz'] else 'no'}")
        print("  1. Ver por qué salieron los últimos resultados")
        print("  2. Cambiar cuántos resultados mostrar")
        print("  3. Activar o desactivar la raíz de las palabras")
        print("  4. Correr las consultas de prueba")
        print("  0. Volver")
        opcion = input("\nElige una opción: ").strip()

        if opcion == "1":
            if e["consulta"] is None:
                print("  Primero haz una búsqueda.")
            else:
                explicar(e["consulta"], e["indice"], e["resultado"], e["raiz"])
        elif opcion == "2":
            valor = input("¿Cuántos resultados quieres ver? (1 a 10): ").strip()
            if valor.isdigit() and 1 <= int(valor) <= 10:
                e["k"] = int(valor)
                print(f"  Ahora se muestran {e['k']} resultados.")
            else:
                print("  Escribe un número del 1 al 10.")
        elif opcion == "3":
            e["raiz"] = not e["raiz"]
            e["indice"] = crear_indice(e["raiz"]) 
            e["consulta"], e["resultado"] = None, pd.DataFrame()
            print(f"  Raíz de palabras {'activada' if e['raiz'] else 'desactivada'}. Se volvió a indexar.")
        elif opcion == "4":
            probar(e["k"])
        elif opcion == "0":
            return
        else:
            print("  Opción no válida.")


def menu(k=3, usar_raiz=False, desde_main=False):
    e = {"k": k, "raiz": usar_raiz, "indice": crear_indice(usar_raiz),
         "consulta": None, "resultado": pd.DataFrame()}

    try:
        while True:
            print("\n" + "=" * 58)
            print("  BUSCADOR DE DOCUMENTOS · NovaTech Soluciones")
            print("=" * 58)
            print("  1. Buscar en todas las áreas")
            print("  2. Buscar solo en mi área")
            print("  3. Explorar los documentos de un área")
            print("  4. Opciones avanzadas")
            print(f"  0. {'Volver al menú principal' if desde_main else 'Salir'}")
            opcion = input("\nElige una opción: ").strip()

            if opcion == "1":
                buscar_y_mostrar(e)
            elif opcion == "2":
                area = elegir_area(e["indice"])
                if area:
                    buscar_y_mostrar(e, area)
            elif opcion == "3":
                explorar_area(e["indice"])
            elif opcion == "4":
                menu_avanzado(e)
            elif opcion == "0":
                break
            else:
                print("  Opción no válida.")
    except (EOFError, KeyboardInterrupt):
        pass
    if not desde_main:
        print("\nHasta luego.")


# Recibir la consulta
def main():
    parser = argparse.ArgumentParser(description="Búsqueda semántica en los documentos de NovaTech")
    parser.add_argument("consulta", nargs="*", help="pregunta en lenguaje natural")
    parser.add_argument("-k", type=int, default=3, help="cuántos resultados mostrar (por defecto 3)")
    parser.add_argument("--area", help="buscar solo en un área (por ejemplo recursos_humanos)")
    parser.add_argument("--raiz", action="store_true", help="reducir las palabras a su raíz")
    parser.add_argument("--explicar", action="store_true", help="mostrar por qué salió cada resultado")
    parser.add_argument("--probar", action="store_true", help="correr las consultas de prueba")
    args = parser.parse_args()

    if args.probar:
        probar(args.k)
    elif args.consulta:
        indice = crear_indice(usar_raiz=args.raiz)
        areas = sorted(indice[0]["categoria"].unique())
        if args.area and args.area not in areas:
            print(f"Área no válida: {args.area}. Opciones: {', '.join(areas)}")
            return
        consulta = " ".join(args.consulta)
        resultado = buscar(consulta, indice, args.k, args.raiz, args.area)
        mostrar(consulta, resultado)
        if args.explicar:
            explicar(consulta, indice, resultado, args.raiz)
    else:
        menu(args.k, args.raiz)


if __name__ == "__main__":
    main()