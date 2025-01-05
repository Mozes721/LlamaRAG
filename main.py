from scripts.prepare_data import generate_embeddings
from models.llama_rag import LlamaRAG

def main():
    try:
        generate_embeddings()
        print("Data preparation complete.")
    except Exception as e:
        print(f"Error preparing data: {e}")
        return
    
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