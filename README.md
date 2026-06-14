# Segmentador Óptimo de Contenido con LLM Local

Sistema que divide un texto en segmentos temáticamente coherentes usando
programación dinámica y un modelo de lenguaje local (Qwen 2.5-3B-Instruct).

---

## Requisitos

- Python 3.9 o superior
- El archivo del modelo: `qwen2.5-3b-instruct-q4_k_m.gguf`
- pip

---

## Instalación

```bash
# 1. Entrar a la carpeta del proyecto
cd "Escuela 2/IA 2/segmentador"

# 2. Crear entorno virtual
python3 -m venv venv

# 3. Activar el entorno virtual
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Copiar el modelo a la carpeta models/
cp /ruta/al/qwen2.5-3b-instruct-q4_k_m.gguf models/
```

> **Apple Silicon (opcional):** para acelerar la inferencia con GPU Metal:
> ```bash
> CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall
> ```

---

## Estructura del proyecto

```
segmentador/
├── app.py            # Servidor web Flask
├── main.py           # Interfaz de línea de comandos
├── segmenter.py      # Algoritmo de programación dinámica
├── llm_client.py     # Carga y consulta del modelo local
├── text_utils.py     # Tokenización de texto en oraciones
├── dataset.py        # Dataset de 5 documentos anotados
├── evaluate.py       # Evaluación con métricas Pk / WindowDiff / F1
├── requirements.txt
├── informe.tex       # Informe técnico en LaTeX
├── models/
│   └── qwen2.5-3b-instruct-q4_k_m.gguf   # modelo (colocar aquí)
└── templates/
    └── index.html    # Interfaz web
```

---

## Cómo correr el proyecto

### Interfaz web (recomendada)

```bash
source venv/bin/activate
python app.py
```

Abrir en el navegador: **http://127.0.0.1:5000**

El modelo tarda unos segundos en cargar al inicio. Una vez que la terminal
muestre `Modelo cargado.`, la interfaz está lista para usar.

### Línea de comandos

```bash
source venv/bin/activate

# Modo interactivo: pegar el texto y terminar con FIN
python main.py

# Desde archivo de texto
python main.py --file mi_texto.txt

# Opciones adicionales
python main.py --max-seg 8 --min-seg 2 --threads 6
```

### Evaluación sobre el dataset

```bash
source venv/bin/activate

# Evaluar todos los documentos con la configuración estándar
python evaluate.py

# Comparar todas las configuraciones experimentales
python evaluate.py --compare

# Evaluar un solo documento
python evaluate.py --id D01

# Ajustar parámetros
python evaluate.py --max-seg 8 --min-seg 2 --threads 6

# Guardar resultados en un archivo específico
python evaluate.py --compare --output mis_resultados.json
```

---

## Parámetros configurables

| Parámetro | Descripción | Valor por defecto |
|---|---|---|
| `--max-seg` | Máximo de oraciones por segmento | `6` |
| `--min-seg` | Mínimo de oraciones por segmento | `1` |
| `--threads` | Hilos de CPU para el modelo | `4` |
| `--output` | Archivo JSON donde guardar resultados | `resultados.json` |

---

## Texto de prueba

```
El calentamiento global es uno de los mayores desafíos del siglo XXI.
Las temperaturas promedio han aumentado aproximadamente 1.1 grados Celsius
desde la era preindustrial. El derretimiento de los glaciares eleva el nivel
del mar y amenaza a millones de personas en zonas costeras. Las olas de calor
son cada vez más frecuentes e intensas en todo el planeta.
La inteligencia artificial ha transformado radicalmente la industria tecnológica
en la última década. Los modelos de lenguaje grandes pueden generar texto, código
y análisis con una precisión sorprendente. Empresas de todo el mundo compiten por
desarrollar sistemas más capaces y eficientes. La automatización impulsada por IA
está redefiniendo el mercado laboral.
La dieta mediterránea es reconocida mundialmente por sus beneficios para la salud
cardiovascular. Se basa en el consumo abundante de frutas, verduras, legumbres y
aceite de oliva. El consumo moderado de pescado y vino tinto también forma parte
de sus principios. Numerosos estudios clínicos confirman que reduce el riesgo de
enfermedades crónicas.
```

El sistema debería identificar exactamente 3 segmentos: clima, IA y nutrición.
