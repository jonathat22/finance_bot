import streamlit as st
import openai
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from google import genai
import tiktoken
import lancedb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from dotenv import load_dotenv
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
import os
import tempfile
import time
import random
import re
from typing import Tuple

# Load environment variables
load_dotenv()

def detect_input_type(user_input: str) -> Tuple[str, str]:
    """
    Detect if the input is a ticker symbol or a question.
    
    Returns:
        Tuple[str, str]: (input_type, normalized_input)
        - input_type: "ticker" or "question"
        - normalized_input: cleaned/normalized version of the input
    """
    user_input = user_input.strip()
    
    # Ticker symbol patterns:
    # - 1-5 uppercase letters (e.g., AAPL, MSFT, BRK.B, GOOGL)
    # - May contain a dot for different share classes
    # - No spaces or special characters except dots
    ticker_pattern = r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$'
    
    # Normalize to uppercase for ticker checking
    normalized = user_input.upper()
    
    if re.match(ticker_pattern, normalized):
        return "ticker", normalized
    else:
        return "question", user_input

def validate_input(user_input: str) -> Tuple[bool, str]:
    """
    Validate user input and provide feedback.
    
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if not user_input or not user_input.strip():
        return False, "Please enter a ticker symbol or question."
    
    if len(user_input.strip()) < 2:
        return False, "Input too short. Please enter a valid ticker symbol or question."
    
    return True, ""

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
            embed_chunks(all_chunks)
            return True, f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}"
        else:
            return False, "No documents were successfully processed."
            
    except Exception as e:
        return False, f"Error processing files: {str(e)}"

def main():
    st.set_page_config(
        page_title="Financial Research Assistant",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Financial Research Assistant 📊")
    st.subheader("Search companies by ticker or question, or upload your own documents")
    
    # Initialize session state
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_company" not in st.session_state:
        st.session_state.current_company = None
    if "input_type" not in st.session_state:
        st.session_state.input_type = None
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Data Source")
        
        # Create tabs for different input methods
        tab1, tab2 = st.tabs(["🔍 Search Company", "📤 Upload Documents"])
        
        with tab1:
            st.write("Enter a company ticker symbol or ask a question about a company.")
            
            # Text input for ticker or question
            user_input = st.text_input(
                "Ticker or Question",
                placeholder="e.g., AAPL or What is Apple's revenue?",
                help="Enter a stock ticker (e.g., AAPL, MSFT) or ask a question about a company",
                key="search_input"
            )
            
            # Display input type detection
            if user_input:
                input_type, normalized = detect_input_type(user_input)
                if input_type == "ticker":
                    st.info(f"🎯 Detected ticker symbol: **{normalized}**")
                else:
                    st.info(f"💬 Detected question: *{user_input}*")
            
            # Search & Analyze button
            search_button = st.button("🚀 Search & Analyze", type="primary", use_container_width=True)
            
            if search_button:
                is_valid, error_msg = validate_input(user_input)
                
                if not is_valid:
                    st.error(error_msg)
                else:
                    input_type, normalized = detect_input_type(user_input)
                    st.session_state.input_type = input_type
                    st.session_state.user_query = user_input
                    
                    with st.spinner("🔍 Searching for financial documents..."):
                        # Placeholder for document search functionality (to be implemented)
                        st.warning("⚠️ Document search functionality will be implemented in the next step.")
                        st.info(f"Input Type: {input_type}\nNormalized: {normalized if input_type == 'ticker' else user_input}")
            
            # Display current search status
            if st.session_state.current_company:
                st.success(f"✅ Currently analyzing: **{st.session_state.current_company}**")
                if st.session_state.processed_files:
                    st.write("**Documents loaded:**")
                    for file in st.session_state.processed_files:
                        st.write(f"- {file}")
        
        with tab2:
            st.write("Upload your own financial documents for analysis.")
            
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
            if st.button("Process Documents", type="primary", use_container_width=True):
                if uploaded_files:
                    with st.spinner("Processing documents... This may take a moment."):
                        success, message = process_uploaded_files(uploaded_files)
                        
                        if success:
                            st.success(message)
                            st.session_state.documents_processed = True
                            # Update with all documents in database
                            st.session_state.processed_files = [f.name for f in uploaded_files]
                            st.session_state.current_company = "Custom Upload"
                        else:
                            st.error(message)
                else:
                    st.warning("Please upload files first.")
            
            # Status
            if st.session_state.documents_processed and st.session_state.current_company == "Custom Upload":
                st.success("✅ Documents are processed and ready for questions!")
                st.write("**Processed files:**")
                for file in st.session_state.processed_files:
                    st.write(f"- {file}")
    
    with col2:
        # Dynamic header based on current state
        if st.session_state.current_company:
            st.header(f"Chat: {st.session_state.current_company}")
        else:
            st.header("Financial Analysis Chat")
        
        if not st.session_state.documents_processed:
            st.info("👈 Search for a company or upload documents to start chatting!")
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
            "This application uses AI-powered web search and document processing to analyze financial statements. "
            "Search for companies by ticker symbol, ask questions, or upload your own documents."
        )
        
        st.subheader("How to use:")
        
        st.write("**Option 1: Search Company**")
        st.write(
            "1. Enter a ticker symbol (e.g., AAPL) or question\n"
            "2. Click 'Search & Analyze' to find financial documents\n"
            "3. Ask questions about the company's financials\n"
            "4. Get AI-powered analysis and insights"
        )
        
        st.write("**Option 2: Upload Documents**")
        st.write(
            "1. Upload your financial documents\n"
            "2. Click 'Process Documents' to analyze content\n"
            "3. Ask questions about your documents\n"
            "4. Get detailed answers with references"
        )
        
        st.subheader("Supported formats:")
        st.write("• Ticker symbols (e.g., AAPL, MSFT, BRK.B)\n• Natural language questions\n• PDF, DOCX, XLSX, PPTX, TXT files")
        
        st.divider()
        
        if st.button("Reset Application", use_container_width=True):
            st.session_state.documents_processed = False
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.session_state.current_company = None
            st.session_state.input_type = None
            st.session_state.user_query = ""
            st.rerun()

if __name__ == "__main__":
    main()
