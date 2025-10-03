# Standard library
import os
from typing import List

# Third-party - Environment
from dotenv import load_dotenv

# Third-party - LangChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Third-party - Document processing
import PyPDF2
import docx

# Local imports
from vectordb import VectorDB

# Load environment variables
load_dotenv()


def load_documents(data_dir: str = "data") -> List[dict]:
    """
    Load documents from directory and subdirectories.
    
    Supports: .txt, .pdf, .docx files
    
    Args:
        data_dir: Root directory containing documents
        
    Returns:
        List of dicts with 'content' and 'metadata' keys
    """
    results = []
    
    for root, _, files in os.walk(data_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            text = ""
            
            if filename.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    
            elif filename.lower().endswith(".pdf"):
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
                        
            elif filename.lower().endswith(".docx"):
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                
            else:
                continue
            
            if text.strip():
                results.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "path": file_path
                    }
                })
    
    print(f"Loaded {len(results)} documents")
    return results


class RAGAssistant:
    """
    RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant with LLM, vector DB, and prompt chain."""
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        self.vector_db = VectorDB()

        self.prompt_template = ChatPromptTemplate.from_template("""
You are a knowledgeable research assistant specializing in providing accurate, 
well-sourced answers based on provided documents.

## Your Role & Behavior:
- Analyze the context carefully before answering
- Be precise and cite specific information from the context
- Maintain a professional yet approachable tone
- If information is ambiguous, acknowledge the nuance

## Scope & Boundaries:
- ONLY answer based on the provided context below
- If the context doesn't contain relevant information, clearly state: 
  "I don't have enough information in the provided documents to answer this question."
- Do not make assumptions or add information not present in the context
- Do not use external knowledge beyond what's provided

## Safety & Ethics:
- If asked something harmful or unethical, politely decline
- Respect privacy and confidentiality of any sensitive information
- Present information objectively without bias

## Output Format:
- Provide clear, structured answers
- Use bullet points for lists when appropriate
- Keep responses concise but complete
- If relevant, mention which document/source the information comes from

---

Context:
{context}

---

Question: {question}

Answer:
""")

        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    def _initialize_llm(self):
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List[dict]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of dicts with 'content' and 'metadata' keys
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant with retrieval-augmented generation.

        Args:
            input: User's question
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM based on retrieved context
        """
        search_results = self.vector_db.search(input, n_results=n_results)
        retrieved_chunks = search_results.get("documents", [])

        if not retrieved_chunks:
            return "I couldn't find relevant information in the documents to answer your question."

        context = "\n\n".join(retrieved_chunks)

        llm_answer = self.chain.invoke({
            "context": context,
            "question": input
        })

        return llm_answer


def main():
    """Main function to run the interactive RAG assistant."""
    try:
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        print("\nLoading documents...")
        documents = load_documents()
        
        if not documents:
            print("No documents found in data/ directory. Please add some documents.")
            return
        
        print(f"Loaded {len(documents)} documents")
        assistant.add_documents(documents)

        print("\n" + "="*50)
        print("RAG Assistant Ready! Ask me anything or type 'quit' to exit.")
        print("="*50 + "\n")

        while True:
            question = input("Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            print("\nThinking...\n")
            answer = assistant.invoke(question)
            print(f"Answer: {answer}\n")
            print("-" * 50 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("1. Created a .env file with at least one API key")
        print("2. Added documents to the data/ directory")
        print("\nSupported API keys:")
        print("  - OPENAI_API_KEY (OpenAI GPT models)")
        print("  - GROQ_API_KEY (Groq Llama models)")
        print("  - GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()