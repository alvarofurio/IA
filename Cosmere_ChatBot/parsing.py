import os
import re
from bs4 import BeautifulSoup
import shutil


INPUT_PATH = 'wiki'
OUTPUT_PATH = 'data'


# Reemplazar caracteres inválidos en el nombre del archivo
def limpiar_nombre(nombre):
    return re.sub(r'[<>:"/\\|?*]', '_', nombre)

'''
Extraer todo el texto dentro de etiquetas <p>,<h1>,<h2>,<h3>,<h4>,<h5>,<h6>
y maquetar:
    - Eliminadno los artículos NO Comere
    - Anadiendo saltos de linea y asteriscos entorno a los titulos segun el grado de importancia
    - Eliminando fragmentos del tipo [numero], [edit] 
    - Eliminando el "spoiler policy" y las cosas del final (Notes, Trivia...)
'''

def extraer_texto(name):
    with open(INPUT_PATH + '/' + name, 'r', encoding='utf-8') as file: 
        contenido_html = file.read()
    soup = BeautifulSoup(contenido_html, 'html.parser')

    if not ("title=\"Cosmere\"" in contenido_html): return 0

    nombre = limpiar_nombre(soup.find('h1').get_text())
    content = nombre.upper() + "\n\n"

    parrafos = soup.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for parrafo in parrafos:
        texto = parrafo.get_text(separator=" ").strip()

        # Usar expresiones regulares para eliminar fragmentos del tipo [numero], [edit]
        texto_limpio = re.sub(r'((| )\[\d+\](| ))|\[ edit \]', '', texto)
        texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
        texto_limpio = re.sub(r'\s\.', '.', texto_limpio)
        texto_limpio = re.sub(r'\s,', ',', texto_limpio)
        # Eliminar la spoiler policy
        if "spoilers" in texto_limpio: continue
        
        
        # Anadir los saltos de linea
        if parrafo.name == 'h2': 
                content += '\n\n'
                texto_limpio = "** " + texto_limpio.upper() + "**"
        if parrafo.name == 'h3': 
                content += '\n'
                texto_limpio = texto_limpio.upper()
        if parrafo.name in ['h4','h5','h6','p']: content += '\n'


        # Eliminar cosas del final
        if "NAVIGATION MENU" in texto_limpio or "TRIVIA" in texto_limpio or "NOTES" in texto_limpio: break
        
        content += texto_limpio
        if parrafo.name == 'p': content += '\n'
    
    f = open(OUTPUT_PATH + "/"+ name[:-5] + ".txt", "w", encoding='utf-8')
    f.write(content)
    f.close()


if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)
os.makedirs(OUTPUT_PATH)

pages = os.listdir(INPUT_PATH)
for p in pages: extraer_texto(p)

