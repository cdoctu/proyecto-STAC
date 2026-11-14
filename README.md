# Proyecto STAC · NovaTech Soluciones

Programa en Python para trabajar con los documentos internos de NovaTech Soluciones. Incluye dos herramientas:

1. **Clasificador de documentos**: entrena un modelo (TF-IDF + Naive Bayes) que asigna cada documento a un área de la empresa.
2. **Buscador de documentos (búsqueda semántica)**: permite hacer preguntas en lenguaje natural y devuelve los documentos más parecidos, usando TF-IDF y similitud coseno.

---

## Requisitos

- **Python 3.10 o superior**
- Librerías:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`

Instalación:

```bash
pip install pandas numpy scikit-learn joblib matplotlib
```

Se recomienda usar un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
pip install pandas numpy scikit-learn joblib matplotlib
```

---

## Estructura del proyecto

```
proyecto/
├── main.py                        # Menú principal
├── src/
│   ├── preprocesamiento.py        # Limpieza de texto y carga de documentos
│   ├── clasificador.py            # Entrenamiento, evaluación y clasificación
│   └── busqueda_semantica.py      # Buscador de documentos
├── data/
│   ├── documentos/                # Documentos de entrenamiento / búsqueda
│   │   ├── recursos_humanos/
│   │   │   └── rh_01_politica_vacaciones.txt
│   │   ├── desarrollo_software/
│   │   ├── seguridad_informacion/
│   │   ├── soporte_clientes/
│   │   └── gestion_proyectos/
│   └── nuevos/                    # Documentos sin clasificar (.txt)
├── models/                        # Se crea solo; aquí se guarda clasificador.joblib
├── resultados/                    # Se crea solo; aquí se guarda consultas_prueba.csv
├── ontology/                      # Archivo OWL/OWX con el diagrama de ontología del proyecto
└── result_photos/                 # Capturas de pantalla del proyecto funcionando
```

- **`ontology/`**: contiene un único archivo OWL/OWX con el diagrama de ontología del proyecto. No es necesario para ejecutar el programa; sirve como documentación del modelo de conocimiento.
- **`result_photos/`**: contiene capturas de pantalla del proyecto en funcionamiento (clasificador y buscador). Es solo material de referencia y no interviene en la ejecución.

### Formato de los datos

- **`data/documentos/`**: cada subcarpeta es una **categoría** (área) y contiene archivos `.txt` en UTF-8. El nombre de la carpeta se usa como etiqueta del documento.
- **La primera línea de cada `.txt` es el título** del documento; el resto es el cuerpo. El buscador usa el título y un resumen del cuerpo para mostrar los resultados.
- **`data/nuevos/`**: archivos `.txt` en UTF-8 que se clasificarán con el modelo entrenado.

> Las carpetas `src/`, `data/` y el archivo `main.py` deben respetar esta organización, ya que las rutas se calculan de forma relativa a ella.

---

## Cómo iniciar el programa

Desde la carpeta raíz del proyecto (donde está `main.py`):

```bash
python main.py
```

Se mostrará el menú principal:

```
==========================================================
  PROYECTO STAC · NovaTech Soluciones
==========================================================
  1. Clasificador de documentos
  2. Buscador de documentos (búsqueda semántica)
  3. Ejecutar todo (clasificador y luego buscador)
  0. Salir
```

---

## Cómo usar el clasificador (opción 1)

Al elegir la opción **1**, el programa realiza automáticamente estos pasos:

1. Carga todos los `.txt` de `data/documentos/` y limpia el texto (minúsculas, sin acentos, sin signos ni números, sin *stopwords* en español; se conserva la letra ñ).
2. Divide los datos en entrenamiento (70 %) y prueba (30 %), de forma estratificada por categoría.
3. Ejecuta una **validación cruzada de 3 folds** sobre el conjunto de entrenamiento.
4. Entrena el modelo y muestra:
   - Exactitud (accuracy)
   - Reporte de clasificación (precisión, recall, F1)
   - Matriz de confusión en texto y en una **gráfica** (cierra la ventana de la gráfica para continuar).
5. Re-entrena el modelo con **todos** los documentos y lo guarda en `models/clasificador.joblib`.
6. Clasifica los archivos de `data/nuevos/` y muestra, para cada uno, la **categoría predicha** y el **porcentaje de confianza**.

Para clasificar documentos nuevos, solo colócalos como `.txt` en `data/nuevos/` y vuelve a ejecutar la opción 1.

---

## Cómo usar el buscador (opción 2)

Al elegir la opción **2**, se abre el menú del buscador:

```
  1. Buscar en todas las áreas
  2. Buscar solo en mi área
  3. Explorar los documentos de un área
  4. Opciones avanzadas
  0. Volver al menú principal
```

- **Buscar en todas las áreas**: escribe una pregunta con tus propias palabras (por ejemplo, *"¿Cómo reporto un correo sospechoso?"*). Verás los documentos más parecidos con su área, similitud (de 0 a 1) y un resumen. Puedes escribir el número de un resultado para leer el documento completo.
- **Buscar solo en mi área**: igual que la anterior, pero limitada a un área que elijas. Se muestran preguntas de ejemplo.
- **Explorar los documentos de un área**: lista todos los documentos de un área para leerlos sin necesidad de buscar.
- **Opciones avanzadas**:
  1. Ver por qué salieron los últimos resultados (qué palabras coincidieron y cuánto aportó cada una).
  2. Cambiar cuántos resultados mostrar (1 a 10).
  3. Activar o desactivar la reducción de palabras a su raíz (vuelve a indexar los documentos).
  4. Correr las consultas de prueba y guardar la tabla en `resultados/consultas_prueba.csv`.

> **Tip:** el buscador compara palabras, no significados. Si no hay resultados, prueba con otras palabras o con una sola palabra clave.

### Uso desde la línea de comandos

El buscador también se puede ejecutar de forma independiente, sin pasar por `main.py`:

```bash
# Abrir el menú interactivo del buscador
python src/busqueda_semantica.py

# Hacer una consulta directa
python src/busqueda_semantica.py ¿Cómo pido mis vacaciones?

# Mostrar 5 resultados
python src/busqueda_semantica.py -k 5 vacaciones

# Buscar solo en un área
python src/busqueda_semantica.py --area recursos_humanos vacaciones

# Reducir palabras a su raíz
python src/busqueda_semantica.py --raiz correos sospechosos

# Ver qué palabras explican cada resultado
python src/busqueda_semantica.py --explicar contraseñas

# Correr las consultas de prueba
python src/busqueda_semantica.py --probar
```

| Argumento     | Descripción                                                          |
|---------------|----------------------------------------------------------------------|
| `consulta`    | Pregunta en lenguaje natural (opcional; sin ella se abre el menú).   |
| `-k N`        | Número de resultados a mostrar (por defecto 3).                      |
| `--area AREA` | Limita la búsqueda a un área (nombre de la carpeta, ej. `recursos_humanos`). |
| `--raiz`      | Reduce las palabras a su raíz (por ejemplo, *correos* → *corr*).     |
| `--explicar`  | Muestra las palabras que coincidieron y su aporte a la similitud.    |
| `--probar`    | Ejecuta las consultas de prueba y guarda el resultado en CSV.        |

Áreas disponibles por defecto: `recursos_humanos`, `desarrollo_software`, `seguridad_informacion`, `soporte_clientes`, `gestion_proyectos`.

---

## Opción 3: ejecutar todo

Ejecuta primero el clasificador (opción 1) y, al terminar, abre el buscador (opción 2). Al salir del buscador se regresa al menú principal.

---

## Cómo funciona (resumen técnico)

- **Preprocesamiento** (`preprocesamiento.py`): normaliza el texto, elimina acentos (conservando la ñ), signos de puntuación, números y una lista de *stopwords* en español.
- **Clasificador** (`clasificador.py`): `Pipeline` de scikit-learn con `TfidfVectorizer` y `MultinomialNB`. El modelo se guarda y se puede recargar con `joblib`.
- **Buscador** (`busqueda_semantica.py`): indexa todos los documentos con `TfidfVectorizer` (unigramas y bigramas, `sublinear_tf=True`), convierte la consulta al mismo espacio vectorial y ordena los documentos por **similitud coseno**. Opcionalmente reduce las palabras a su raíz mediante una regla sencilla (quita el plural y la última vocal).

---

## Notas y solución de problemas

- **`ModuleNotFoundError: No module named 'src'`**: ejecuta `main.py` desde la carpeta raíz del proyecto.
- **`ValueError` o resultados vacíos al entrenar**: verifica que `data/documentos/` exista, tenga subcarpetas por categoría y cada categoría tenga suficientes archivos `.txt` (la división estratificada y la validación cruzada requieren varios documentos por categoría).
- **Aviso "el documento esperado ... no existe"** al correr las pruebas: edita la lista `PRUEBAS` al inicio de `src/busqueda_semantica.py` para que coincida con los nombres de tus documentos.
- **Sin gráfica en entornos sin pantalla** (por ejemplo, servidores): configura un *backend* de matplotlib no interactivo o ejecuta el programa en un equipo con entorno gráfico.
- Todos los archivos de texto deben estar codificados en **UTF-8**.