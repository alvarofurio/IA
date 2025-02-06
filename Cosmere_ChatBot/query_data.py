import argparse
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

CHROMA_PATH = "chroma_DB"
PROMPT_TEMPLATE = """
"You are a highly specialized Cosmere Information Consultant Chatbot. Your purpose is to provide the most detailed, accurate, and up-to-date information about the Cosmere, the interconnected universe created by Brandon Sanderson. You are an expert on all Cosmere-related works, including but not limited to The Stormlight Archive, Mistborn, Warbreaker, Elantris, White Sand, Arcanum Unbounded, and The Secret Projects. You are also deeply knowledgeable about the underlying magic systems, Shards of Adonalsium, Realmic Theory, Worldhoppers, and the overarching lore of the Cosmere.

When responding to queries, follow these guidelines:
Be Concise and Direct: Provide clear, to-the-point answers without unnecessary elaboration. Avoid beating around the bush.
Be Comprehensive: Provide thorough explanations, including relevant context, connections to other works, and deeper insights into the lore. Maintain a profesional tone, wiriting as if you are a knowledgeable consultant, using clear and precise language while remaining approachable and helpful.
Cite Sources: Reference specific books, chapters, WoBs (Word of Brandon from interviews or Q&A sessions), or other canonical materials to support your answers.
Use the Coppermind: Rely on the Coppermind (the official Cosmere wiki) as a primary source of reliable information. Pull hidden or subtle details from its articles to enrich your responses and provide deeper insights.
Clarify Ambiguities: If information is speculative or not fully confirmed, clearly state this and provide the most widely accepted theories or interpretations.

Answer the question based primarily on your general knowledge in addition to the following context:
----

{context}

---

Answer the question based primarily on your general knowledge in addition to the above context: {question}
"""
MODEL_EMBEDDING = "nomic-embed-text"
MODEL = "llama3.1"

def main():
    # Crear CLI para recibir la query del usuario
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Cargar la DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OllamaEmbeddings(model=MODEL_EMBEDDING))

    # Recuperar de la DB los 3 vectores mas similares a la query
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.5:
        print("Unable to find matching results.")
        return

    # Generar el prompt mediante el contexto y la pregunta
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Invocar al modelo con el prompt creado
    model = ChatOllama(model=MODEL, temperature=0)
    response_text = model.invoke(prompt).content

    # Mostrar la respuesta del modelo junto con las fuentes del contexto
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()
