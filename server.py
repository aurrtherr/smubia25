from flask import Flask, request, jsonify, render_template
import logging
from PyPDF2 import PdfReader
from html import escape
from newspaper import Article
import requests
import spacy
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from spacy.matcher import Matcher
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import time
import os

# Initialize the Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file upload size to 16 MB

matplotlib.use('Agg')  # ✅ Fix Tkinter issue

# Configure logging for debugging purposes
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text.strip() if text else "No text found in PDF."
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return None

def extract_text_from_url(url):
    """Extracts text from a webpage (news article)."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip() if article.text else "No readable text found on the page."
    except Exception as e:
        logging.error(f"Failed to fetch article text from URL: {e}")
        return None
    
@app.route("/")
def home():
    nltk.download('punkt_tab')
    return render_template("index.html", time_now=int(time.time()))

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles the analysis of data sent by the front-end.
    Supports file uploads (.txt, .pdf), plain text, and URL-based extraction.
    """
    try:

        logging.info(f"Received request: {request.form if request.form else request.files}")
        
        # Handle file upload
        if 'file' in request.files:
            logging.info("[HANDLE FILE]")
            file = request.files['file']
            # Process .txt files
            if file.filename.endswith('.txt'):
                content = file.read().decode('utf-8')
                extract_knowledge_graph(content)
                summary = summarize_text(content)
                # return jsonify({"type": "file", "content": escape(content)})
                return jsonify({'Summary': summary})
            # Process .pdf files
            elif file.filename.endswith('.pdf'):
                content = extract_text_from_pdf(file)
                extract_knowledge_graph(content)
                summary = summarize_text(content)
                if content:
                    # return jsonify({"type": "file", "content": escape(content)})
                    return jsonify({'Summary': summary})
                return jsonify({"error": "Failed to extract text from PDF"}), 500

        # Handle text input
        elif 'text' in request.form:
            logging.info("[TEXT INPUT]")
            text = request.form['text']
            extract_knowledge_graph(text)
            summary = summarize_text(text)
            # return jsonify({"type": "text", "content": escape(text)})
            return jsonify({'Summary': summary})

        # Handle URL input (web scraping)
        elif 'url' in request.form:
            logging.info("[HANDLE URL]")
            url = request.form['url'].strip()
            if not url.startswith(("http://", "https://")):
                return jsonify({"error": "Invalid URL format. Please include http:// or https://"}), 400
            
            extracted_text = extract_text_from_url(url)
            extract_knowledge_graph(extracted_text)
            getWordCloud(extracted_text)
            summary = summarize_text(extracted_text)
            if extracted_text:
                # return jsonify({"type": "url", "content": escape(extracted_text)})
                return jsonify({'Summary': summary})
            return jsonify({"error": "Failed to extract text from URL"}), 500

        return jsonify({"error": "No valid input provided."}), 400

    except Exception as e:
        logging.info("[ANALYZE EXCEPTION]")
        logging.error(f"Error occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

def extract_knowledge_graph(text):
    logging.info("[EXTRACT KNOWLEDGE GRAPH]")
    nlp = spacy.load('en_core_web_sm')
    sentences = sent_tokenize(text)

     # Ensure the directory exists
    image_path = "static/images/entity_network.png"

    # ✅ Delete existing image before saving a new one
    if os.path.exists(image_path):
        os.remove(image_path)

    def get_entities(sent):
        ent1, ent2 = "", ""
        prv_tok_dep, prv_tok_text = "", ""
        prefix, modifier = "", ""

        for tok in nlp(sent):
            if tok.dep_ != "punct":
                if tok.dep_ == "compound":
                    prefix = tok.text if prv_tok_dep != "compound" else prv_tok_text + " " + tok.text
                if tok.dep_.endswith("mod"):
                    modifier = tok.text if prv_tok_dep != "compound" else prv_tok_text + " " + tok.text
                if "subj" in tok.dep_:
                    ent1 = (modifier + " " + prefix + " " + tok.text).strip()
                    prefix, modifier = "", ""
                if "obj" in tok.dep_:
                    ent2 = (modifier + " " + prefix + " " + tok.text).strip()
            prv_tok_dep, prv_tok_text = tok.dep_, tok.text

        return [ent1, ent2]

    def get_relation(sent):
        doc = nlp(sent)
        matcher = Matcher(nlp.vocab)
        pattern = [{'DEP': 'ROOT'}, {'DEP': 'prep', 'OP': "?"}, {'DEP': 'agent', 'OP': "?"}, {'POS': 'ADJ', 'OP': "?"}]
        matcher.add("matching_1", [pattern])
        matches = matcher(doc)
        return doc[matches[-1][1]:matches[-1][2]].text if matches else ""

    entity_pairs = [get_entities(sentence) for sentence in sentences]
    relations = [get_relation(sentence) for sentence in sentences]

    kg_df = pd.DataFrame({
        'source': [pair[0] for pair in entity_pairs],
        'target': [pair[1] for pair in entity_pairs],
        'edge': relations
    })

    kg_df = kg_df[(kg_df['source'] != '') & (kg_df['target'] != '')]

    G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr="edge", create_using=nx.MultiDiGraph())
    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G, k=1.2, seed=62)
    pos = nx.rescale_layout_dict(pos, scale=2)

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='black', linewidths=5, font_size=10)

    edge_labels = {(row['source'], row['target']): row['edge'] for _, row in kg_df.iterrows()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)

    plt.savefig(image_path)
    plt.close()

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from wordcloud import STOPWORDS

def getWordCloud(text):

    # Ensure the directory exists
    image_path = "static/images/word_cloud.png"

    # ✅ Delete existing image before saving a new one
    if os.path.exists(image_path):
        os.remove(image_path)

    wordcloud = WordCloud(
        width=1000, height=500,
        background_color='white',
        stopwords=STOPWORDS,
        max_words=50,
        colormap='coolwarm'
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.savefig(image_path)
    plt.close()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_text(text, max_chunk_length=800):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    final_summary = []
    for chunk in chunks:
        try:
            prompt = (
                f"Summarize the content provided clearly and concisely, "
                f"focusing on the key points and details whilst avoiding redundant information: {chunk}"
            )
            inputs = tokenizer(prompt, max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=20, num_beams=5, repetition_penalty=3.0, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            final_summary.append(summary)
        except Exception as e:
            final_summary.append(f"Error summarizing chunk: {e}")
    
    summary_compiled = " ".join(final_summary)
    summary = summary_compiled.replace("..", ".").strip()
    return summary[0].capitalize() + summary[1:] if summary else summary



# Run the Flask server
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT not set
    app.run(host="0.0.0.0", port=port, debug=True)
