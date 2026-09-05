import os
import PyPDF2
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    
    ALLOWED_EXTENSIONS = {'pdf', 'txt'}
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in DocumentProcessor.ALLOWED_EXTENSIONS
    
    @staticmethod
    def extract_text_from_pdf(filepath: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Get total pages for logging
                num_pages = len(pdf_reader.pages)
                logger.info(f"Processing PDF with {num_pages} pages")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF: {str(e)}")
            raise
    
    @staticmethod
    def extract_text_from_txt(filepath: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting TXT: {str(e)}")
            raise
    
    @staticmethod
    def process_document(filepath: str) -> str:
        """Main method to process any supported document"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        file_ext = filepath.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            return DocumentProcessor.extract_text_from_pdf(filepath)
        elif file_ext == 'txt':
            return DocumentProcessor.extract_text_from_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks