from flask import Flask, render_template, request, jsonify, send_file
import os
from PyPDF2 import PdfReader, PdfWriter
import uuid

from utils import ArgumentParser, ConfigLoader, LOG
from model import GLMModel, OpenAIModel
from translator import PDFTranslator

app = Flask(__name__)

# 文件存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model_type = "openai"
glm_model_url = ""
timeout = 300
openai_model = "gpt-3.5-turbo"
openai_api_key = os.getenv("OPENAI_API_KEY")
book = ""
file_format = "markdown"
target_language = "chinese"


@app.route('/')
def index():
    return render_template('index.html')

# API 路由
@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "File is not a PDF"}), 400
    
    file_id = str(uuid.uuid4())  # 生成唯一 ID
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
    file.save(input_path)
    
    return jsonify({"file_id": file_id, "message": "File uploaded successfully!"}), 200

@app.route('/api/convert/<file_id>', methods=['GET'])
def convert_pdf(file_id):
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")
    output_path = os.path.join(OUTPUT_FOLDER, f"converted_{file_id}.pdf")
    
    if not os.path.exists(input_path):
        return jsonify({"error": "File not found"}), 404
    
    try:
        convert_pdf_to_chinese(input_path, output_path)
        return jsonify({"file_id": file_id, "message": "Conversion successful!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/download/<file_id>', methods=['GET'])
def download_pdf(file_id):
    output_path = os.path.join(OUTPUT_FOLDER, f"converted_{file_id}.pdf")
    if not os.path.exists(output_path):
        return jsonify({"error": "Converted file not found"}), 404
    return send_file(output_path, as_attachment=True)


@app.route('/api/translate', methods=['POST'])
def translate_pdf():
    model_type = request.form.get('model_type')
    glm_model_url = request.form.get('glm_model_url')
    timeout = request.form.get('timeout')
    openai_model = request.form.get('openai_model')
    book = request.files.get('book')
    file_format = request.form.get('file_format')
    target_language = request.form.get('target_language')
    markdown_preview = request.form.get('markdown-preview')
    # 验证必要参数
    if model_type == "GLMModel" and not glm_model_url:
        return jsonify({"error": "GLM Model URL is required for GLMModel"}), 400
    if model_type == "OpenAIModel" and (not openai_model or not openai_api_key):
        return jsonify({"error": "OpenAI Model and API Key are required for OpenAIModel"}), 400

    # 保存上传的文件并执行翻译逻辑
    book_path = os.path.join(UPLOAD_FOLDER, book.filename)
    book.save(book_path)
    output_file_path = os.path.join(OUTPUT_FOLDER, f"converted_{book.filename}")
    if file_format == "markdown":
        output_file_path = output_file_path.replace('.pdf', '.md')
    # 实例化 PDFTranslator 类，并调用 translate_pdf() 方法
    translator = PDFTranslator(model_type)
    translator.translate_pdf(book_path, file_format, target_language, output_file_path)

    if file_format == "markdown":
        #write markdown to preview
        with open(output_file_path, 'r', encoding='utf-8') as f:
            markdown_preview = f.read()
    return jsonify({"message": "Translation completed successfully!", "markdown_preview": markdown_preview})

    #return jsonify({"message": "Translation started successfully!"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
