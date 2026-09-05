from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import logging
from config.settings import config
from app.document_processor import DocumentProcessor
from app.qa_engine import QAEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='frontend', static_folder='frontend')
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

qa_engine = QAEngine(
    api_key=config.OPENAI_API_KEY,
    persist_directory=config.CHROMA_DB_PATH
)

current_document_loaded = False


@app.route('/')
def home():
    """Serve homepage"""
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    global current_document_loaded
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not DocumentProcessor.allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use PDF or TXT'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        try:
            text = DocumentProcessor.process_document(filepath)
            chunks = DocumentProcessor.chunk_text(text)
            
            # Ingest into Q&A engine
            qa_engine.ingest_documents(
                texts=chunks,
                metadata={"filename": filename}
            )
            
            qa_engine.initialize_qa_chain()
            
            current_document_loaded = True
            
            return jsonify({
                'status': 'success',
                'message': f'Document "{filename}" uploaded and processed',
                'chunks': len(chunks)
            }), 200
        
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return jsonify({'error': f'Processing error: {str(e)}'}), 500
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    global current_document_loaded
    
    try:
        if not current_document_loaded:
            return jsonify({'error': 'Please upload a document first'}), 400
        
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        answer, sources = qa_engine.answer_question(question)
        
        return jsonify({
            'status': 'success',
            'answer': answer,
            'sources': sources
        }), 200
    
    except Exception as e:
        logger.error(f"Q&A error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'document_loaded': current_document_loaded
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Smart Document Q&A Bot")
    app.run(debug=config.DEBUG, port=5000)