from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import shutil
import glob


CHROMA_PATH = "chroma_DB"
DATA_PATH = "data"
MODEL_EMBEDDING = "nomic-embed-text"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    documents = []
    # Manually load each .txt file in the directory
    for file_path in glob.glob(f"{DATA_PATH}/*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    # Eliminar posibles finales de paginas que no tengan contenido
    chunks = [chunk for chunk in chunks if len(chunk.page_content) > 10]
 
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Eliminar la CHROMA DB si ya existia
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_model = OllamaEmbeddings(model=MODEL_EMBEDDING)
    db = None

    # Crear una nueva FAISS DB a partir de los documentos
    total_chunks = len(chunks); batch_size=100; failed_batches = []
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]; batch_number = i // batch_size + 1
        print(f"Processing batch {batch_number}/{(total_chunks - 1) // batch_size + 1}")
        try:
            if db is None: db = Chroma.from_documents(batch_chunks, embedding_model, persist_directory=CHROMA_PATH)
            else: db.add_documents(batch_chunks)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            failed_batches.append({'batch_number': batch_number, 'batch_chunks': batch_chunks, 'retries': 1})
            continue

    if db is not None:
        processed_chunks = total_chunks - len(failed_batches) * batch_size
        print(f"Saved {processed_chunks} chunks to {CHROMA_PATH}.")
    else:
        print("No data was saved to the vector store.")


    # Reintentar añadir los batches fallidos
    max_retries=5
    if failed_batches:
        print(f"\nRetrying failed batches...")
        while failed_batches:
            batch = failed_batches.pop(0)
            batch_number = batch['batch_number']; batch_chunks = batch['batch_chunks']; retries = batch['retries']

            print(f"Retrying batch {batch_number}, attempt {retries}/{max_retries}")
            try:
                if db is None: db = Chroma.from_documents(batch_chunks, embedding_model, persist_directory=CHROMA_PATH)
                else: db.add_documents(batch_chunks)
                print(f"Batch {batch_number} added successfully on retry {retries}.")
            except Exception as e:
                print(f"Batch {batch_number} failed on retry {retries}: {e}")
                if retries < max_retries:
                    batch['retries'] += 1
                    failed_batches.append(batch)
                else: print(f"Batch {batch_number} reached maximum retries. Skipping.")

    # Mpstrar los resultados
    if failed_batches:
        failed_batch_numbers = [batch['batch_number'] for batch in failed_batches]
        print(f"\nThe following batches failed after {max_retries} retries: {failed_batch_numbers}")
    else: print("\nAll batches processed successfully after retries.")

if __name__ == "__main__":
    main()