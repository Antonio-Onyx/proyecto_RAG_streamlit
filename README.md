# Proyecto RAG con Streamlit

Esta aplicación es una web app construida con Streamlit para chatear con documentos propios usando un flujo RAG (Retrieval-Augmented Generation). El proyecto permite subir archivos, procesarlos en fragmentos, generar embeddings con Hugging Face, guardar y consultar los vectores en Supabase, y responder preguntas con distintos proveedores LLM como OpenAI, Groq o Gemini.

## Qué hace el proyecto

- Carga documentos desde la interfaz web.
- Procesa el contenido para dividirlo en chunks.
- Genera embeddings con el modelo `BAAI/bge-m3`.
- Guarda y recupera contexto desde Supabase Vector Store.
- Responde preguntas con soporte de RAG cuando hay un documento cargado.
- También puede funcionar como chat simple si no hay documento procesado.

## Requisitos previos

Antes de ejecutar la app conviene tener listo lo siguiente:

- Python 3.10 o superior.
- Una cuenta y proyecto en Supabase.
- Al menos una API key válida para el proveedor LLM que vayas a usar.
- Dependencias instalables desde `requirements.txt`.

## Variables de entorno

El proyecto usa un archivo `.env`. Debes definir, como mínimo:

```env
API_HOST=openai

OPENAI_API_KEY=tu_api_key
GROQ_API_KEY=tu_api_key
GEMINI_API_KEY=tu_api_key

SUPABASE_URL=tu_supabase_url
SUPABASE_KEY=tu_supabase_key
```

Notas:

- `API_HOST` puede ser `openai`, `groq` o `gemini`.
- No necesitas usar las tres API keys; basta con configurar la del proveedor seleccionado en `API_HOST`.
- La app depende de Supabase para almacenar y recuperar embeddings.

## Instalación

### 1. Crear el entorno virtual

En Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecutar la aplicación

Con el entorno virtual activo:

```bash
streamlit run app.py
```

Luego abre en el navegador la URL local que muestre Streamlit.

## Consideraciones importantes

- El flujo actual de procesamiento está pensado principalmente para documentos PDF.
- El proyecto usa `SupabaseVectorStore`, por lo que la tabla `documents` y la función `match_documents` deben existir en tu base de datos.
- La generación de embeddings está configurada con `device="cuda"`, así que actualmente espera una GPU NVIDIA compatible con CUDA. Si vas a correrlo solo en CPU, hay que ajustar esa configuración en `RAG_modules_logic.py`.

## Estructura principal

- `app.py`: interfaz Streamlit y flujo principal del chat.
- `RAG_modules_logic.py`: procesamiento de documentos, embeddings y almacenamiento vectorial.
- `client_llm_logic.py`: clientes para distintos proveedores LLM.
- `requirements.txt`: dependencias del proyecto.
