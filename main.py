from scripts.parse_data import EmbeddingGenerator
from models.llama_rag import LlamaRAG

def main():
    # try:
    #     embedding_generator = EmbeddingGenerator()

    #     missing_types = embedding_generator.check_embeddings()
    #     if missing_types:
    #         print(f"Missing embeddings detected: {missing_types}. Generating...")
    #         embedding_generator.generate_and_store_embeddings(missing_types)
    #     else:
    #         print("All embeddings are already up-to-date.")

    # except Exception as e:
    #     print(f"Error during embedding generation: {e}")
    #     return
    
    # Step 2: Initialize LlamaRAG
    print("Initializing LlamaRAG...")
    llama_rag = LlamaRAG()
    
    # Step 3: Start the Llama RAG pipeline
    print("Starting Llama RAG pipeline...")
    while True:
        # Get user input
        query = input("\nYou: ")
        
        # Exit condition
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Handle the query
        try:
            response = llama_rag.handle_query(query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error handling query: {e}")

if __name__ == "__main__":
    main()