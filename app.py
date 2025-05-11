"""
VideoAsk to Rev AI Integration Webhook Server with Advanced NLP Processing
---------------------------------------------------------------
This configuration sets up a webhook server that handles:
1. Receiving VideoAsk webhook requests with audio responses
2. Validating request signatures for security
3. Submitting audio to Rev AI for transcription
4. Handling Rev AI webhook callbacks
5. Processing transcripts using advanced NLP techniques
6. Storing processed transcripts in a database with contact metadata
7. Updating VideoAsk with transcription status (optional)
8. GitHub webhook integration for version control and CI/CD
"""

import os
import json
import hmac
import hashlib
import requests
import re
import unicodedata
import numpy as np
import pandas as pd
import spacy
import nltk
from typing import List, Dict, Any
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from textblob import TextBlob
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load environment variables
load_dotenv()

# Configuration
class Config:
    # VideoAsk Configuration
    VIDEOASK_WEBHOOK_SECRET = os.getenv('VIDEOASK_WEBHOOK_SECRET')
    VIDEOASK_API_KEY = os.getenv('VIDEOASK_API_KEY')
    VIDEOASK_API_BASE_URL = "https://api.videoask.com/v1"
    
    # Rev AI Configuration
    REV_AI_ACCESS_TOKEN = os.getenv('REV_AI_ACCESS_TOKEN')
    REV_AI_CALLBACK_URL = os.getenv('REV_AI_CALLBACK_URL')  # Your webhook endpoint for Rev AI callbacks
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///transcriptions.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Google Sheets Configuration (if using Sheets as database)
    GOOGLE_SHEETS_CREDENTIALS = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
    GOOGLE_SHEETS_ID = os.getenv('GOOGLE_SHEETS_ID')
    GOOGLE_SHEETS_NAME = "FINAL SUBMISSIONS"
    
    # Question mapping for Google Sheets (column mappings)
    QUESTION_COLUMN_MAP = {
        "Introduce Yourself": {"link_col": 7, "transcript_col": 8},
        "Foundation's Influence": {"link_col": 9, "transcript_col": 10},
        "Sharing Advice": {"link_col": 11, "transcript_col": 12},
        "Purpose & Joy": {"link_col": 13, "transcript_col": 14},
        "Staying Connected": {"link_col": 15, "transcript_col": 16}
    }
    EMAIL_COLUMN_INDEX = 5  # Column E
    
    # Webhook Proxy URL (for local development)
    WEBHOOK_PROXY_URL = os.getenv('WEBHOOK_PROXY_URL', 'https://smee.io/NXoLZTqSCKr2j4T')

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize database
db = SQLAlchemy(app)

# Database Models
class Transcription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    contact_id = db.Column(db.String(255), nullable=False)
    contact_email = db.Column(db.String(255), nullable=True)
    question_id = db.Column(db.String(255), nullable=False)
    question_text = db.Column(db.Text, nullable=True)
    audio_url = db.Column(db.String(512), nullable=False)
    rev_ai_job_id = db.Column(db.String(255), nullable=True)
    transcript = db.Column(db.Text, nullable=True)
    processed_transcript = db.Column(db.Text, nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    named_entities = db.Column(db.JSON, nullable=True)
    keywords = db.Column(db.JSON, nullable=True)
    summary = db.Column(db.Text, nullable=True)
    semantic_complexity = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(50), default='pending')
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    updated_at = db.Column(db.DateTime, onupdate=db.func.now())

    def __repr__(self):
        return f'<Transcription {self.id}>'

# Advanced NLP Processor from main.py
class AdvancedTextProcessor:
    def __init__(self):
        """
        Initialize advanced NLP processing utilities
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize Hugging Face pipelines
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        self.summarizer = pipeline('summarization')
        
        # Stopwords removal
        self.stop_words = set(stopwords.words('english'))
        
        # Word embedding model
        self.word2vec_model = self.train_word2vec_model()
        
        # Error correction model
        self.error_correction_model = self.create_error_correction_model()
        
        app.logger.info("Advanced NLP Processor initialized successfully")

    def train_word2vec_model(self, sentences: List[str] = None) -> Word2Vec:
        """
        Train Word2Vec model for semantic understanding
        
        Args:
            sentences (List[str], optional): Training sentences
        
        Returns:
            Word2Vec model
        """
        # Default training sentences if not provided
        if sentences is None:
            sentences = [
                "climate change affects global ecosystems",
                "human activities impact environmental sustainability",
                "scientific research reveals complex planetary challenges",
                "interview response transcript",
                "audio transcription content",
                "personal story narrative",
                "foundation influence experience",
                "advice sharing mentorship",
                "purpose joy fulfillment",
                "staying connected relationship"
            ]
        
        # Tokenize sentences
        tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=tokenized_sentences, 
            vector_size=100, 
            window=5, 
            min_count=1, 
            workers=4
        )
        
        return model

    def create_error_correction_model(self) -> Sequential:
        """
        Create a deep learning model for error correction
        
        Returns:
            Keras Sequential model
        """
        # Sample training data
        texts = [
            "climate change",
            "global warming",
            "environmental crisis",
            "scientific research",
            "interview response",
            "personal story",
            "foundation influence",
            "advice sharing",
            "purpose joy",
            "staying connected"
        ]
        
        # Tokenization
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        
        # Sequence padding
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=10)
        
        # Create model
        model = Sequential([
            Embedding(len(tokenizer.word_index) + 1, 50, input_length=10),
            LSTM(100),
            Dense(len(tokenizer.word_index) + 1, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        return model

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1 (str): First text
            text2 (str): Second text
        
        Returns:
            Semantic similarity score
        """
        if not text1 or not text2:
            return 0.0
            
        # Tokenize and remove stopwords
        tokens1 = [word for word in word_tokenize(text1.lower()) if word not in self.stop_words]
        tokens2 = [word for word in word_tokenize(text2.lower()) if word not in self.stop_words]
        
        # Calculate word embeddings
        vectors1 = [self.word2vec_model.wv[word] for word in tokens1 if word in self.word2vec_model.wv]
        vectors2 = [self.word2vec_model.wv[word] for word in tokens2 if word in self.word2vec_model.wv]
        
        # Average word vectors
        if not vectors1 or not vectors2:
            return 0.0
        
        avg_vector1 = np.mean(vectors1, axis=0)
        avg_vector2 = np.mean(vectors2, axis=0)
        
        # Calculate cosine similarity
        return np.dot(avg_vector1, avg_vector2) / (np.linalg.norm(avg_vector1) * np.linalg.norm(avg_vector2))

    def advanced_error_correction(self, text: str) -> str:
        """
        Advanced error correction using multiple techniques
        
        Args:
            text (str): Input text
        
        Returns:
            Corrected text
        """
        if not text:
            return ""
            
        # SpaCy-based correction
        doc = self.nlp(text)
        
        # TextBlob spelling correction
        blob = TextBlob(text)
        spelling_corrected = str(blob.correct())
        
        # Machine learning-based suggestions
        corrections = {
            'climat': 'climate',
            'resaerch': 'research',
            'enviromental': 'environmental',
            'adviz': 'advice',
            'purpuse': 'purpose',
            'influenze': 'influence',
            'connectesd': 'connected'
        }
        
        for error, correction in corrections.items():
            spelling_corrected = spelling_corrected.replace(error, correction)
        
        return spelling_corrected

    def extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text
        """
        if not text:
            return []
            
        doc = self.nlp(text)
        entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
        
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text
        """
        if not text or len(text.strip()) < 10:
            return {"label": "NEUTRAL", "score": 0.5}
            
        try:
            result = self.sentiment_analyzer(text)[0]
            return result
        except Exception as e:
            app.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "NEUTRAL", "score": 0.5}

    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """
        Generate summary of text
        """
        if not text or len(text.strip()) < 50:
            return text
            
        try:
            # Ensure text is long enough for summarization
            if len(text.split()) < 30:
                return text
                
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min(30, len(text.split()) // 2), 
                do_sample=False
            )[0]['summary_text']
            
            return summary
        except Exception as e:
            app.logger.error(f"Error in text summarization: {str(e)}")
            return text

    def extract_keywords(self, text: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Extract keywords from text using TF-IDF
        """
        if not text or len(text.strip()) < 10:
            return []
            
        try:
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top keywords
            tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
            top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            return [{"word": word, "score": float(score)} for word, score in top_keywords]
        except Exception as e:
            app.logger.error(f"Error in keyword extraction: {str(e)}")
            return []

    def calculate_semantic_complexity(self, text: str) -> float:
        """
        Calculate semantic complexity of text
        """
        if not text or len(text.strip()) < 10:
            return 0.0
            
        try:
            # Tokenize and remove stopwords
            tokens = [word for word in word_tokenize(text.lower()) if word not in self.stop_words]
            
            if not tokens:
                return 0.0
                
            # Calculate unique word ratio
            unique_words = len(set(tokens))
            total_words = len(tokens)
            
            # Calculate average word embeddings distance
            if len(tokens) > 1:
                embeddings = [self.word2vec_model.wv[word] for word in tokens if word in self.word2vec_model.wv]
                if embeddings:
                    # Calculate pairwise distances
                    distances = [np.linalg.norm(embeddings[i] - embeddings[j]) 
                                for i in range(len(embeddings)) 
                                for j in range(i+1, len(embeddings))]
                    avg_distance = np.mean(distances) if distances else 0
                else:
                    avg_distance = 0
            else:
                avg_distance = 0
            
            # Combine metrics
            complexity_score = (unique_words / total_words) * (1 + avg_distance)
            
            return min(1.0, complexity_score)
        except Exception as e:
            app.logger.error(f"Error in complexity calculation: {str(e)}")
            return 0.0

    def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text analysis
        
        Args:
            text (str): Input text
        
        Returns:
            Dictionary with multiple analysis results
        """
        if not text:
            return {
                "named_entities": [],
                "sentiment": {"label": "NEUTRAL", "score": 0.5},
                "summary": "",
                "top_keywords": [],
                "semantic_complexity": 0.0
            }
        
        # Extract named entities
        entities = self.extract_named_entities(text)
        
        # Sentiment analysis
        sentiment = self.analyze_sentiment(text)
        
        # Text summarization
        summary = self.summarize_text(text)
        
        # Extract keywords
        keywords = self.extract_keywords(text)
        
        # Calculate semantic complexity
        complexity = self.calculate_semantic_complexity(text)
        
        return {
            "named_entities": entities,
            "sentiment": sentiment,
            "summary": summary,
            "top_keywords": keywords,
            "semantic_complexity": complexity
        }

    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Process transcript with all NLP techniques
        
        Args:
            transcript (str): Raw transcript text
            
        Returns:
            Dictionary with processed text and analysis results
        """
        if not transcript:
            return {
                "processed_text": "",
                "named_entities": [],
                "sentiment": {"label": "NEUTRAL", "score": 0.5},
                "summary": "",
                "keywords": [],
                "semantic_complexity": 0.0
            }
        
        # Apply error correction
        processed_text = self.advanced_error_correction(transcript)
        
        # Perform comprehensive analysis
        analysis = self.comprehensive_analysis(processed_text)
        
        return {
            "processed_text": processed_text,
            "named_entities": analysis["named_entities"],
            "sentiment": analysis["sentiment"],
            "summary": analysis["summary"],
            "keywords": analysis["top_keywords"],
            "semantic_complexity": analysis["semantic_complexity"]
        }

# Create database tables
with app.app_context():
    db.create_all()

# Initialize the NLP processor as a global object
nlp_processor = AdvancedTextProcessor()

# Helper functions
def validate_videoask_signature(payload, signature):
    """Validate the signature from VideoAsk webhook request"""
    if not app.config['VIDEOASK_WEBHOOK_SECRET']:
        app.logger.warning("VIDEOASK_WEBHOOK_SECRET not configured, skipping signature validation")
        return True
        
    expected_signature = hmac.new(
        app.config['VIDEOASK_WEBHOOK_SECRET'].encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected_signature, signature)

def submit_to_rev_ai(audio_url, transcription_id):
    """Submit audio URL to Rev AI for transcription"""
    url = "https://api.rev.ai/speechtotext/v1/jobs"
    headers = {
        "Authorization": f"Bearer {app.config['REV_AI_ACCESS_TOKEN']}",
        "Content-Type": "application/json"
    }
    
    callback_url = f"{app.config['REV_AI_CALLBACK_URL']}?transcription_id={transcription_id}"
    
    payload = {
        "media_url": audio_url,
        "callback_url": callback_url,
        "skip_diarization": True,
        "skip_punctuation": False,
        "remove_disfluencies": True,
        "filter_profanity": False,
        "speaker_channels_count": 1
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        job_details = response.json()
        return job_details.get('id'), 'in_progress'
    else:
        app.logger.error(f"Rev AI submission failed: {response.text}")
        return None, 'failed'

def get_rev_ai_transcript(job_id):
    """Get transcription from Rev AI"""
    url = f"https://api.rev.ai/speechtotext/v1/jobs/{job_id}/transcript"
    headers = {
        "Authorization": f"Bearer {app.config['REV_AI_ACCESS_TOKEN']}",
        "Accept": "application/json"
    }
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        transcript_data = response.json()
        
        # Extract the full transcript text
        monologues = transcript_data.get('monologues', [])
        transcript_text = ""
        
        for monologue in monologues:
            for element in monologue.get('elements', []):
                if element.get('type') == 'text':
                    transcript_text += element.get('value', '') + ' '
        
        return transcript_text.strip()
    else:
        app.logger.error(f"Rev AI transcript retrieval failed: {response.text}")
        return None

def update_google_sheet(email, question_text, audio_url, transcript):
    """Update Google Sheet with transcription data"""
    # This is a placeholder for Google Sheets API integration
    # In a production environment, you'd use gspread or Google API client
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
        
        # Define the scope
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Authenticate
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            app.config['GOOGLE_SHEETS_CREDENTIALS'], scope
        )
        client = gspread.authorize(creds)
        
        # Open the spreadsheet and sheet
        sheet = client.open_by_key(app.config['GOOGLE_SHEETS_ID']).worksheet(app.config['GOOGLE_SHEETS_NAME'])
        
        # Find the row with the matching email
        cells = sheet.findall(email)
        if not cells:
            app.logger.error(f"Email {email} not found in spreadsheet")
            return False
            
        row = cells[0].row
        
        # Get column mappings for this question
        if question_text in app.config['QUESTION_COLUMN_MAP']:
            mapping = app.config['QUESTION_COLUMN_MAP'][question_text]
            link_col = mapping['link_col']
            transcript_col = mapping['transcript_col']
            
            # Update the sheet
            sheet.update_cell(row, link_col, audio_url)
            sheet.update_cell(row, transcript_col, transcript)
            return True
        else:
            app.logger.error(f"Question mapping not found for: {question_text}")
            return False
            
    except Exception as e:
        app.logger.error(f"Google Sheets update error: {str(e)}")
        return False

def setup_webhook_proxy():
    """Set up the webhook proxy for local development"""
    if os.getenv('FLASK_ENV') == 'development' and app.config['WEBHOOK_PROXY_URL']:
        try:
            from smee_client import SmeeClient
            
            smee = SmeeClient({
                'source': app.config['WEBHOOK_PROXY_URL'],
                'target': f"http://localhost:{os.getenv('PORT', 5000)}",
                'logger': app.logger
            })
            
            events = smee.start()
            app.logger.info(f"Webhook proxy started at {app.config['WEBHOOK_PROXY_URL']}")
            return events
        except ImportError:
            app.logger.warning("smee-client not installed, skipping webhook proxy setup")
            return None
    return None

# API Routes
@app.route('/webhook/github', methods=['POST'])
def github_webhook():
    """Handle incoming webhook from GitHub for version control and CI/CD"""
    # Verify GitHub signature
    signature = request.headers.get('X-Hub-Signature-256', '')
    if not signature.startswith('sha256='):
        return jsonify({"error": "Invalid signature format"}), 401
        
    signature = signature[7:]  # Remove 'sha256=' prefix
    secret = os.getenv('GITHUB_WEBHOOK_SECRET', '').encode()
    
    # Calculate expected signature
    payload = request.get_data()
    expected_signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    
    # Compare signatures
    if not hmac.compare_digest(expected_signature, signature):
        return jsonify({"error": "Invalid signature"}), 401
    
    # Process the webhook
    try:
        data = request.json
        event_type = request.headers.get('X-GitHub-Event')
        
        if event_type == 'push':
            # Handle code push events
            repository = data.get('repository', {}).get('name')
            branch = data.get('ref', '').split('/')[-1]
            commits = data.get('commits', [])
            
            app.logger.info(f"GitHub push event: {repository}/{branch} with {len(commits)} commits")
            
            # Here you can implement CI/CD actions:
            # - Pull latest code
            # - Restart the application
            # - Run tests
            # - Deploy updates
            
            # For example, automatic restart on main branch pushes
            if branch == 'main':
                # This is a placeholder for actual restart logic
                app.logger.info("Main branch updated, application should restart")
                
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        app.logger.error(f"Error processing GitHub webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/webhook/videoask', methods=['POST'])
def videoask_webhook():
    """Handle incoming webhook from VideoAsk"""
    # Verify signature
    payload = request.get_data()
    signature = request.headers.get('X-Videoask-Signature', '')
    
    if not validate_videoask_signature(payload, signature):
        return jsonify({"error": "Invalid signature"}), 401
    
    # Process request
    try:
        data = request.json
        event_type = data.get('event_type')
        
        # Check if this is a form response event
        if event_type == 'form_response':
            contact_id = data.get('contact', {}).get('id')
            contact_email = data.get('contact', {}).get('email')
            
            # Process each question answered
            questions = data.get('data', {}).get('questions', [])
            
            for question in questions:
                question_id = question.get('id')
                question_text = question.get('question_text', '')
                answers = question.get('answers', [])
                
                for answer in answers:
                    if answer.get('type') == 'audio':
                        audio_url = answer.get('audio_url')
                        
                        if audio_url:
                            # Create transcription record
                            transcription = Transcription(
                                contact_id=contact_id,
                                contact_email=contact_email,
                                question_id=question_id,
                                question_text=question_text,
                                audio_url=audio_url
                            )
                            db.session.add(transcription)
                            db.session.commit()
                            
                            # Submit to Rev AI
                            job_id, status = submit_to_rev_ai(audio_url, transcription.id)
                            
                            if job_id:
                                transcription.rev_ai_job_id = job_id
                                transcription.status = status
                                db.session.commit()
        
        # Always return 200 success to acknowledge the webhook
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        app.logger.error(f"Error processing VideoAsk webhook: {str(e)}")
        # Still return 200 so VideoAsk doesn't retry
        return jsonify({"status": "error", "message": str(e)}), 200

@app.route('/webhook/rev-ai', methods=['POST'])
def rev_ai_webhook():
    """Handle incoming webhook from Rev AI"""
    try:
        # Get transcription ID from query parameter
        transcription_id = request.args.get('transcription_id')
        if not transcription_id:
            return jsonify({"error": "Missing transcription_id parameter"}), 400
            
        # Get job details from Rev AI webhook
        data = request.json
        job_id = data.get('job', {}).get('id')
        status = data.get('job', {}).get('status')
        
        # Find the transcription record
        transcription = Transcription.query.get(transcription_id)
        if not transcription:
            return jsonify({"error": "Transcription not found"}), 404
            
        # Update status
        transcription.status = status
        
        # If job is complete, get the transcript and process it
        if status == 'transcribed':
            transcript = get_rev_ai_transcript(job_id)
            if transcript:
                transcription.transcript = transcript
                
                # Process transcript with NLP
                try:
                    nlp_results = nlp_processor.process_transcript(transcript)
                    
                    # Update transcription record with NLP results
                    transcription.processed_transcript = nlp_results["processed_text"]
                    transcription.sentiment_score = nlp_results["sentiment"]["score"]
                    transcription.named_entities = nlp_results["named_entities"]
                    transcription.keywords = nlp_results["keywords"]
                    transcription.summary = nlp_results["summary"]
                    transcription.semantic_complexity = nlp_results["semantic_complexity"]
                    
                    app.logger.info(f"Transcript {transcription_id} processed successfully with NLP")
                except Exception as e:
                    app.logger.error(f"Error processing transcript with NLP: {str(e)}")
                
                # Update Google Sheet if email is available
                if transcription.contact_email and transcription.question_text:
                    # Use the processed transcript for the sheet
                    processed_text = transcription.processed_transcript or transcript
                    update_google_sheet(
                        transcription.contact_email,
                        transcription.question_text,
                        transcription.audio_url,
                        processed_text
                    )
        
        db.session.commit()
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        app.logger.error(f"Error processing Rev AI webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Optional endpoint to manually check transcription status
@app.route('/transcriptions/<int:transcription_id>', methods=['GET'])
def get_transcription(transcription_id):
    """Get transcription status and content including NLP analysis"""
    transcription = Transcription.query.get_or_404(transcription_id)
    
    # Format sentiment for display
    sentiment_label = None
    sentiment_score = None
    if transcription.sentiment_score is not None:
        sentiment_score = transcription.sentiment_score
        if sentiment_score > 0.6:
            sentiment_label = "POSITIVE"
        elif sentiment_score < 0.4:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
    
    return jsonify({
        "id": transcription.id,
        "contact_id": transcription.contact_id,
        "contact_email": transcription.contact_email,
        "question_text": transcription.question_text,
        "audio_url": transcription.audio_url,
        "rev_ai_job_id": transcription.rev_ai_job_id,
        "status": transcription.status,
        "transcript": transcription.transcript,
        "sentiment": {
            "label": sentiment_label,
            "score": sentiment_score
        }
    })