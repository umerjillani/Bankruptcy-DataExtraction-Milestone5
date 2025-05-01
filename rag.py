# -*- coding: utf-8 -*-
"""
Modified RAG System with OCR capabilities
"""

import os
import sys
import fitz  # PyMuPDF
import tempfile
from pathlib import Path
import io
import re
from PIL import Image
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import time
import hashlib
import json


# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import chromadb
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_page_as_image(pdf_path, page_num):
    """Extract PDF page as image"""
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return Image.open(io.BytesIO(pix.tobytes("png")))
    finally:
        doc.close()

def perform_ocr_on_page(pdf_path, page_num, ocr_methods, lang="eng"):
    """OCR processing for single page"""
    try:
        image = extract_page_as_image(pdf_path, page_num)
        if not image:
            return None

        text = ""
        if 'pytesseract' in ocr_methods:
            import pytesseract
            text = pytesseract.image_to_string(image, lang=lang)
            
        if not text.strip() and 'easyocr' in ocr_methods:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(np.array(image))
            text = "\n".join([t[1] for t in results])

        return {
            "page_content": text,
            "metadata": {
                "source": pdf_path,
                "page": page_num,
                "is_ocr": True
            }
        }
        
    except Exception as e:
        logger.error(f"Page {page_num+1} OCR failed: {str(e)}")
        return None

def extract_image_text_concurrent(pdf_path, page_nums, ocr_methods, max_workers=4, lang="eng"):
    """Concurrent OCR processing"""
    extracted = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(perform_ocr_on_page, pdf_path, num, ocr_methods, lang): num 
            for num in page_nums
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    extracted.append(Document(**result))
            except Exception as e:
                logger.error(f"OCR failed: {str(e)}")
    
    return extracted

def get_document_hash(pdf_path):
    """Generate a hash of the document content"""
    return hashlib.md5(open(pdf_path, 'rb').read()).hexdigest()

def document_exists(pdf_path, fingerprint_dir):
    """Check if document has already been processed by comparing hash"""
    doc_hash = get_document_hash(pdf_path)
    fingerprint_path = os.path.join(fingerprint_dir, f"{doc_hash}.json")
    
    if os.path.exists(fingerprint_path):
        with open(fingerprint_path, 'r') as f:
            data = json.load(f)
            return data.get('vector_store_path')
    return None

def save_document_fingerprint(pdf_path, vector_store_path, fingerprint_dir):
    """Save document fingerprint and vector store location"""
    doc_hash = get_document_hash(pdf_path)
    os.makedirs(fingerprint_dir, exist_ok=True)
    
    fingerprint_path = os.path.join(fingerprint_dir, f"{doc_hash}.json")
    data = {
        'pdf_path': pdf_path,
        'vector_store_path': vector_store_path,
        'timestamp': time.time()
    }
    
    with open(fingerprint_path, 'w') as f:
        json.dump(data, f)

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.embedding = OpenAIEmbeddings()
        self.vector_store_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
        
        # Configuration
        self.ocr_enabled = True
        self.chunk_size = 1024
        self.chunk_overlap = 100
        self.max_concurrent_ocr = 4
        self.tesseract_lang = "eng"
            
    def ingest(self, pdf_path, fingerprint_dir=None):
        """Modified ingestion with document fingerprinting"""
        try:
            # Cleanup any existing resources
            self.cleanup()
            
            # Check if document has already been processed
            if fingerprint_dir:
                existing_store = document_exists(pdf_path, fingerprint_dir)
                if existing_store and os.path.exists(existing_store):
                    print(f"📄 Document already processed, reusing vector store: {existing_store}")
                    self.load_existing_vector_store(existing_store)
                    return self
            
            # Create new vector store for new document
            vector_store_dir = self._create_new_vector_store_instance()
            
            print(f"🔍 Processing pages in {os.path.basename(pdf_path)}...")
            
            # Extract text from PDF
            pages = self._extract_text_and_ocr(pdf_path)
            if not pages:
                raise ValueError("No text extracted from PDF")
                
            # Split text into chunks
            chunks = self._split_text(pages)
            self._create_vector_store(chunks)
            print(f"✅ Successfully processed {len(chunks)} chunks")
            
            # Save fingerprint if directory is provided
            if fingerprint_dir:
                save_document_fingerprint(pdf_path, vector_store_dir, fingerprint_dir)
            
            return self
            
        except Exception as e:
            print(f"❌ Ingestion failed: {str(e)}")
            raise
    
    def ask(self, query):
        """Execute query against processed documents"""
        try:
            # Validate vector store
            if not self.vector_store:
                raise ValueError("No documents ingested - vector store is not initialized")
                
            # Initialize chain if needed
            if not self.chain:
                print("🔗 Initializing query chain...")
                self._initialize_chain()
                
            # Additional verification
            if not hasattr(self.vector_store, 'as_retriever'):
                raise ValueError("Vector store is invalid - missing as_retriever method")
                
            # Execute query
            print(f"🔍 Executing query: {query}")
            result = self.chain.invoke({"input": query})
            
            return {
                "answer": result["answer"],
                "sources": [doc.metadata for doc in result["context"]]
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            print(f"❌ Query error: {str(e)}")
            raise
                
    def set_vector_store_dir(self, directory):
        """Set the vector store directory explicitly"""
        self.vector_store_dir = directory
        return self

    def load_existing_vector_store(self, directory):
        """Load an existing vector store from a directory"""
        try:
            print(f"🔄 Loading vector store from {directory}...")
            self.vector_store_dir = directory
            
            # Make sure the directory exists
            if not os.path.exists(directory):
                raise ValueError(f"Vector store directory does not exist: {directory}")
            
            # Clear any existing vector store
            self.vector_store = None
            self.chain = None
            
            # Initialize with properly configured client
            client = chromadb.PersistentClient(
                path=directory,
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get the collection names in the directory
            collections = client.list_collections()
            if not collections:
                raise ValueError(f"No collections found in vector store directory: {directory}")
            
            # Use the first collection
            collection_name = collections[0].name
            print(f"📚 Found collection: {collection_name}")
            
            # Initialize the vector store with the collection
            self.vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=self.embedding,
                persist_directory=directory
            )
            
            # Verify the vector store has documents
            try:
                # Try to get one document to verify the store works
                retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})
                docs = retriever.get_relevant_documents("test query")
                print(f"✓ Vector store verified with {len(docs)} test results")
            except Exception as e:
                print(f"⚠️ Vector store verification warning: {str(e)}")
            
            # Initialize the chain
            self._initialize_chain()
            
            print(f"✅ Successfully loaded existing vector store from {directory}")
            return self
            
        except Exception as e:
            print(f"❌ Failed to load vector store: {str(e)}")
            # Reset to avoid partial initialization
            self.vector_store = None
            self.chain = None
            raise            
            
    # ------------ Core Processing Methods ------------
    def _extract_text_and_ocr(self, pdf_path):
        """Combined text extraction and OCR processing"""
        pages = self._extract_pdf_text(pdf_path)
        
        if self.ocr_enabled:
            image_pages = self._get_image_pages(pdf_path)
            if image_pages:
                ocr_texts = self._process_ocr(pdf_path, image_pages)
                pages.extend(ocr_texts)
                
        return pages

    def _extract_pdf_text(self, pdf_path):
        """Text extraction using PyMuPDF"""
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            try:
                text = doc.load_page(page_num).get_text()
                if text.strip():
                    pages.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num, "is_ocr": False}
                    ))
            except Exception as e:
                logger.warning(f"Page {page_num+1} error: {str(e)}")
        doc.close()
        return pages

    def _process_ocr(self, pdf_path, page_nums):
        """OCR processing for image-based pages"""
        ocr_methods = self._get_ocr_methods()
        return extract_image_text_concurrent(
            pdf_path, 
            page_nums,
            ocr_methods,
            self.max_concurrent_ocr,
            self.tesseract_lang
        )

    # ------------ Helper Methods ------------
    def _get_ocr_methods(self):
        """Check available OCR methods"""
        methods = []
        try:
            import pytesseract
            methods.append('pytesseract')
        except ImportError:
            pass
            
        try:
            import easyocr
            methods.append('easyocr')
        except ImportError:
            pass
            
        return methods

    def _get_image_pages(self, pdf_path):
        """Identify pages containing images"""
        doc = fitz.open(pdf_path)
        image_pages = [num for num in range(len(doc)) if doc[num].get_images()]
        doc.close()
        return image_pages

    def _split_text(self, documents):
        """Split documents into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return splitter.split_documents(documents)
    
    def cleanup(self):
        """Clean up resources between runs"""
        if hasattr(self, 'vector_store') and self.vector_store is not None:
            # Delete the collection if possible
            try:
                if hasattr(self.vector_store, '_client'):
                    self.vector_store._client.reset()
            except Exception as e:
                logger.warning(f"Vector store cleanup warning: {str(e)}")
            
            # Reset vector_store
            self.vector_store = None
        
        # Reset chain
        self.chain = None
        
    def _create_vector_store(self, chunks):
        """Create Chroma vector store with proper client configuration"""
        # Important: Use unique collection name to avoid conflicts
        collection_name = f"collection_{int(time.time())}"
        
        # Create client with explicit settings
        client = chromadb.PersistentClient(
            path=self.vector_store_dir,
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True  # Important for cleanup
            )
        )
        
        # Create the vector store with the client and collection
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.vector_store_dir,
            client=client,  # Use our configured client
            collection_name=collection_name  # Use unique collection name
        )

    def _clear_vector_store(self):
        """Completely reset vector store between PDFs by removing the directory"""
        try:
            if os.path.exists(self.vector_store_dir):
                # Delete the entire directory
                shutil.rmtree(self.vector_store_dir)
                # Create an empty directory
                os.makedirs(self.vector_store_dir)
                print(f"♻️ Vector store reset at {self.vector_store_dir}")
            else:
                # Create the directory if it doesn't exist
                os.makedirs(self.vector_store_dir, exist_ok=True)
                print(f"📁 Created new vector store directory at {self.vector_store_dir}")
        except Exception as e:
            print(f"⚠️ Vector store reset failed: {str(e)}")
            raise
        
    def cleanup_old_vector_stores(self, base_dir, max_stores=5):
        """Delete old vector stores, keeping only the most recent ones"""
        pattern = re.compile(r'chroma_db_\d+')
        stores = []
        
        # Find all vector store directories
        for item in os.listdir(base_dir):
            if pattern.match(item) and os.path.isdir(os.path.join(base_dir, item)):
                timestamp = int(item.split('_')[-1])
                stores.append((timestamp, os.path.join(base_dir, item)))
        
        # Sort by timestamp (newest first)
        stores.sort(reverse=True)
        
        # Delete old stores beyond max_stores
        for _, dir_path in stores[max_stores:]:
            try:
                shutil.rmtree(dir_path)
                print(f"🧹 Deleted old vector store: {dir_path}")
            except Exception as e:
                print(f"⚠️ Failed to delete old vector store: {str(e)}")
                
    def _create_new_vector_store_instance(self):
        """Create a new vector store with a unique path for each PDF"""
        # Generate a unique directory name using timestamp
        unique_dir = os.path.join(
            os.path.dirname(self.vector_store_dir), 
            f"chroma_db_{int(time.time())}"
        )
        
        # Update the vector store directory path
        self.vector_store_dir = unique_dir
        
        # Ensure directory exists
        os.makedirs(unique_dir, exist_ok=True)
        print(f"📁 Created new vector store at {unique_dir}")
        
        # No need to clear anything as we're using a fresh directory
        return unique_dir
    
    def _initialize_chain(self):
        """Initialize LangChain processing chain"""
        try:
            if not self.vector_store:
                raise ValueError("Cannot initialize chain - vector store not loaded")
                
            model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            
            prompt = PromptTemplate.from_template(
                """Answer the question based only on the context:
                Context: {context}
                Question: {input}
                Answer:"""
            )
            
            document_chain = create_stuff_documents_chain(model, prompt)
            
            # Make sure the retriever is properly configured
            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
            )
            
            self.chain = create_retrieval_chain(retriever, document_chain)
            print("✅ Query chain initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize chain: {str(e)}")
            self.chain = None
            raise