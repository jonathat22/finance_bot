import streamlit as st
import openai
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from openai import OpenAI
from google import genai
from transformers import AutoTokenizer
import tiktoken
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import os
import tempfile
import io
import time
import random
from typing import Optional

# Load environment variables
load_dotenv()

def exponential_backoff_retry(func, max_retries=5, base_delay=1):
    """Retry function with exponential backoff for rate limiting"""
    for attempt in range(max_retries):
        try:
            return func()
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            
            # Calculate delay with exponential backoff and jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            st.warning(f"Rate limit hit. Retrying in {delay:.1f} seconds... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        except Exception as e:
            # For non-rate-limit errors, don't retry
            raise e
    
    raise Exception("Max retries exceeded")

def extract(uploaded_document):
    """Extract content from uploaded document using Docling"""
    converter = DocumentConverter()
    result = converter.convert(uploaded_document)
    return result

def get_chunks(extracted_data):
    """Get chunks from extracted data using HybridChunker"""
    tokenizer = OpenAITokenizer(
        tokenizer=tiktoken.encoding_for_model("gpt-4o"),
        max_tokens=8191,  # context window length required for OpenAI tokenizers
    )
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True
    )
    chunk_iter = chunker.chunk(dl_doc=extracted_data.document)
    chunks = list(chunk_iter)
    return chunks

def process_chunks(chunks):
    """Process chunks into structured format"""
    processed_chunks = [
        {
            "text": chunk.text,
            "metadata": {
                "filename": chunk.meta.origin.filename,
                "page_numbers": [
                    page_no
                    for page_no in sorted(
                        set(
                            prov.page_no
                            for item in chunk.meta.doc_items
                            for prov in item.prov
                        )
                    )
                ]
                or None,
                "title": chunk.meta.headings[0] if chunk.meta.headings else None,
            },
        }
        for chunk in chunks
    ]
    return processed_chunks

def embed_chunks(chunks, table_name="docling"):
    """Embed chunks and store in LanceDB"""
    uri = "db/docling_db"
    db = lancedb.connect(uri)
    func = get_registry().get("sentence-transformers").create()

    class ChunkMetadata(LanceModel):
        filename: str | None
        page_numbers: list[int] | None
        title: str | None

    class Chunks(LanceModel):
        text: str = func.SourceField()
        vector: Vector(func.ndims()) = func.VectorField()
        metadata: ChunkMetadata

    table = db.create_table(
        table_name,
        schema=Chunks,
        mode="overwrite"
    )

    processed_chunks = process_chunks(chunks)
    table.add(processed_chunks)
    
    return table

def query_documents(question, table_name="docling", chat_history=None):
    """Query the document database with a question, including chat history for context"""
    try:
        uri = "db/docling_db"
        db = lancedb.connect(uri)
        table = db.open_table(table_name)
        
        results = table.search(question).limit(5).to_pandas()
        
        if results.empty:
            return "No relevant information found in the uploaded documents."
        
        # Prepare context from retrieved chunks, organizing by document
        context = ""
        
        for idx, row in results.iterrows():
            context += f"Document: {row['metadata']['filename']}\n"
            if row['metadata']['title']:
                context += f"Section: {row['metadata']['title']}\n"
            if row['metadata']['page_numbers']:
                context += f"Pages: {row['metadata']['page_numbers']}\n"
            context += f"Content: {row['text']}\n\n"
        
        # Build conversation history for context
        conversation_context = ""
        if chat_history:
            conversation_context = "\n\nPrevious conversation:\n"
            # Include last 3 exchanges for context (to avoid token limits)
            recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
            for msg in recent_history:
                if msg['role'] == 'user':
                    conversation_context += f"Human: {msg['content']}\n"
                else:
                    conversation_context += f"Assistant: {msg['content']}\n"
        
        # Use Gemini to generate response with retry logic
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        def make_gemini_request():
        
            prompt = f"Context from documents:\n{context}{conversation_context}\n\nCurrent question: {question}"
            
            request_gemini = client.models.generate_content(
                model="gemini-2.0-flash",
                config={
                    "system_instruction": "You are a financial analyst assistant. Answer questions based on the provided document context and conversation history. Be specific and reference the documents when possible. If the user is asking a follow-up question, consider the previous conversation context.",
                    "temperature": 0.1,
                },
                contents=prompt
            )
            return request_gemini
        
        # Use exponential backoff for rate limiting
        response = exponential_backoff_retry(make_gemini_request, max_retries=5, base_delay=1)
        
        return response.text
        
    except Exception as e:
        return f"Error querying documents: {str(e)}"

def process_uploaded_files(uploaded_files):
    """Process uploaded files and embed them"""
    if not uploaded_files:
        return False, "No files uploaded"
    
    try:
        all_chunks = []
        processed_files = []
        
        for uploaded_file in uploaded_files:
                
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Extract content
                extracted_data = extract(tmp_file_path)
                
                # Get chunks
                chunks = get_chunks(extracted_data)
                all_chunks.extend(chunks)
                processed_files.append(uploaded_file.name)
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
        
        if all_chunks:
            # Embed all chunks
            table = embed_chunks(all_chunks)
            return True, f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}"
        else:
            return False, "No documents were successfully processed."
            
    except Exception as e:
        return False, f"Error processing files: {str(e)}"

def main():
    st.set_page_config(
        page_title="Financial Document Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Financial Document Analyzer 📊")
    st.subheader("Upload financial statements and ask questions about them")
    
    # Initialize session state
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose financial documents",
            type=['pdf', 'docx', 'xlsx', 'pptx', 'txt'],
            accept_multiple_files=True,
            help="Upload financial statements, reports, or any relevant documents"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            for file in uploaded_files:
                st.write(f"- {file.name}")
        
        # Process button
        if st.button("Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents... This may take a moment."):
                    success, message = process_uploaded_files(uploaded_files)
                    
                    if success:
                        st.success(message)
                        st.session_state.documents_processed = True
                        # Update with all documents in database
                        st.session_state.processed_files = [f.name for f in uploaded_files]
                    else:
                        st.error(message)
            else:
                st.warning("Please upload files first.")
        
        # Status
        if st.session_state.documents_processed:
            st.success("✅ Documents are processed and ready for questions!")
            st.write("**Processed files:**")
            for file in st.session_state.processed_files:
                st.write(f"- {file}")
    
    with col2:
        st.header("Chat with Your Documents")
        
        if not st.session_state.documents_processed:
            st.info("👈 Please upload and process documents first to start chatting!")
        else:
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your financial documents...", disabled=not st.session_state.documents_processed):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                
                # Generate and display assistant response
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Searching documents and generating response..."):
                            response = query_documents(prompt, chat_history=st.session_state.messages[:-1])
                            st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("Clear Chat", type="secondary"):
                st.session_state.messages = []
                st.rerun()
    
    # Sidebar with information
    with st.sidebar:
        st.header("About")
        st.write(
            "This application uses advanced document processing to analyze financial statements. "
            "Upload your documents and ask questions to get insights from your financial data."
        )
        
        st.subheader("How to use:")
        st.write(
            "1. Upload your financial documents using the file uploader\n"
            "2. Click 'Process Documents' to analyze and index the content\n"
            "3. Ask questions about your documents in natural language\n"
            "4. Get detailed answers with references to your documents"
        )
        
        st.subheader("Supported file types:")
        st.write("• PDF files\n• Word documents (.docx)\n• Excel files (.xlsx)\n• PowerPoint files (.pptx)\n• Text files (.txt)")
        
        if st.button("Reset Application"):
            st.session_state.documents_processed = False
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
