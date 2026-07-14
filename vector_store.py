import os
import hashlib
import re
from typing import Any
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Explicitly load the .env file at startup
load_dotenv()

class VectorStoreManager:
    def __init__(self, persist_directory="db", collection_name="agentic_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Pull the API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-001",
                google_api_key=api_key
            )
        else:
            self.embeddings = None
        
        # Initialize the local Chroma vector database
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def build_database(self, chunks):
        """Converts text to vectors and saves them to the disk."""
        if self.embeddings is None:
            raise ValueError("GOOGLE_API_KEY is required to create document embeddings.")

        if not chunks:
            print("No chunks found. Database update skipped.")
            return

        print(f"Converting {len(chunks)} chunks into vectors and saving to database...")
        
        # Separate the text from the metadata
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        ids = [
            hashlib.sha1(
                f"{metadata.get('source')}|{metadata.get('page')}|{metadata.get('chunk')}|{text}".encode("utf-8")
            ).hexdigest()
            for text, metadata in zip(texts, metadatas)
        ]

        for source in {metadata.get("source") for metadata in metadatas if metadata.get("source")}:
            try:
                self.vector_store._collection.delete(where={"source": source})
            except Exception:
                pass
        
        # Add the documents to Chroma
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print(f"Successfully secured memory inside the '{self.persist_directory}/' folder!")

    def add_generated_knowledge(self, query, answer, references=None):
        """Stores an LLM-generated report as searchable knowledge for later reuse."""
        if self.embeddings is None:
            raise ValueError("GOOGLE_API_KEY is required to store generated knowledge.")

        slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:60] or "generated_answer"
        source = f"generated_{slug}.txt"
        reference_text = "\n".join(
            item.get("label") or item.get("source", "")
            for item in references or []
            if item.get("label") or item.get("source")
        )
        content = (
            f"Generated knowledge for query: {query}\n\n"
            f"{answer.strip()}\n\n"
            f"Reference sources:\n{reference_text or 'LLM general knowledge'}"
        )
        chunks = []
        chunk_size = 1500
        for index in range(0, len(content), chunk_size):
            chunks.append({
                "text": content[index:index + chunk_size],
                "metadata": {
                    "source": source,
                    "page": 1,
                    "chunk": (index // chunk_size) + 1,
                    "type": "generated_knowledge",
                    "query": query,
                },
            })

        self.build_database(chunks)
        return source

    def sync_database(self, chunks):
        """Replaces the collection with chunks from the current data folder."""
        self.clear()
        if chunks:
            self.build_database(chunks)

    def search(self, query, k=3):
        """Retrieves the top k most relevant chunks for a user's question."""
        if self.embeddings is None:
            return []

        print(f"\nSearching memory for: '{query}'")
        
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        return formatted_results

    def clear(self):
        """Completely resets the Chroma collection."""
        try:
            count = self.vector_store._collection.count()

            # Delete the collection
            self.vector_store.delete_collection()

            # Recreate it
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )

            print(f"Deleted {count} chunks.")
            return count

        except Exception as e:
            print("Clear error:", e)
            return 0

    def count(self):
        """Returns how many searchable chunks are currently stored."""
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def sources(self):
        """Returns a compact list of source files represented in the vector store."""
        try:
            data: dict[str, Any] = self.vector_store._collection.get(include=["metadatas"])
            metadatas = data.get("metadatas") or []
        except Exception:
            return []

        source_map = {}
        for metadata in metadatas:
            if not metadata:
                continue
            source = metadata.get("source", "Unknown")
            entry = source_map.setdefault(source, {"source": source, "chunks": 0, "pages": set()})
            entry["chunks"] += 1
            if metadata.get("page"):
                entry["pages"].add(metadata["page"])

        sources = []
        for entry in source_map.values():
            pages = sorted(entry["pages"])
            sources.append({
                "source": entry["source"],
                "chunks": entry["chunks"],
                "pages": pages,
                "page_count": len(pages)
            })

        return sorted(sources, key=lambda item: item["source"].lower())

# Test pipeline integration
if __name__ == "__main__":
    from document_processor import DocumentProcessor
    
    # 1. Run Ingestion logic to grab chunks
    processor = DocumentProcessor()
    raw_chunks = processor.process_all_documents()
    
    # 2. Run Database logic to embed and store them
    db_manager = VectorStoreManager()
    
    if raw_chunks:
        db_manager.build_database(raw_chunks)
        
        # 3. Test search execution on your CV data
        test_query = "What is the professional background or skills mentioned?" 
        results = db_manager.search(test_query)
        
        print("\n--- Top Search Results ---")
        for res in results:
            source = res['metadata'].get('source', 'Unknown')
            page = res['metadata'].get('page', 'N/A')
            
            print(f"Source: {source} (Page {page}) | Distance Score: {res['score']:.4f}")
            print(f"Excerpt: {res['content'][:150]}...\n")
