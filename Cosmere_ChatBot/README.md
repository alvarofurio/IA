## PARSING DE LOS DOCUMENTOS
EL directorio "wiki" contiene los archivos html raw. 

Ejecutar parsing.py -> Se crea el directorio "data" que contiene los archivos .txt depurados.

## CREACIÓN DE LA BASE DE DATOS
Ejecutar en una terminal distinta 'ollama pull nomic-embed-text'

Ejecutar create_database.py -> Crea una CHROMA DB (y su directorio chroma_DB asociado) con los vectores asociados a los .txt de "data" divididos en chunks.

## GENERACIÓN DE LA CONSULTA
Ejecutar query_data.py "query text" -> Genera una respuesta a la pregunta a partir de un contexto recuperado de la DB.