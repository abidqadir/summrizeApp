from flask import Flask, render_template, request, redirect, url_for
import os
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the summarization model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Summarize the extracted text
def summarize_text(text, max_length=500, min_length=100):
    summary = summarizer(text, max_length=max_length, min_length=min_length, truncation=True)
    return summary[0]['summary_text']

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Handle file upload and summarization
@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.pdf'):
            # Save the file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Extract text and summarize
            pdf_text = extract_text_from_pdf(file_path)
            summary = summarize_text(pdf_text)

            # Render the template with the summary
            return render_template('index.html', summary=summary)

    return redirect(url_for('index'))

