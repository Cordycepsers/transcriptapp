import os
import json
import hmac
import hashlib
import requests
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

# Initialize NLP Processor
class NLPProcessor:
    def __init__(self):
        """
        Initialize NLP processing utilities
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
        self.word2vec_model = self.initialize_word2vec_model()
        
        app.logger.info("NLP Processor initialized successfully")
    
    def initialize_word2vec_model(self) -> Word2Vec:
        """
        Initialize Word2Vec model for semantic understanding
        """
        # Default training sentences
        sentences = [
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
        
        # Create Word2Vec model
        model = Word2Vec(
            sentences=tokenized_sentences, 
            vector_size=100, 
            window=5, 
            min_count=1, 
            workers=4
        )
        
        return model
    
    def correct_text(self, text: str) -> str:
        """
        Apply text correction to transcription
        """
        if not text:
            return ""
            
        # SpaCy-based correction
        doc = self.nlp(text)
        
        # TextBlob spelling correction
        blob = TextBlob(text)
        spelling_corrected = str(blob.correct())
        
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
            
            # Calculate average word length
            avg_word_length = sum(len(word) for word in tokens) / total_words
            
            # Calculate sentence complexity metrics
            sentences = text.split('.')
            avg_sentence_length = sum(len(sentence.split()) for sentence in sentences if sentence.strip()) / max(1, len([s for s in sentences if s.strip()]))
            
            # Combine metrics
            complexity_score = (unique_words / total_words) * (avg_word_length / 5) * (avg_sentence_length / 10)
            
            return min(1.0, complexity_score)
        except Exception as e:
            app.logger.error(f"Error in complexity calculation: {str(e)}")
            return 0.0
    
    def process_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Process transcript with all NLP techniques
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
        
        processed_text = self.correct_text(transcript)
        named_entities = self.extract_named_entities(processed_text)
        sentiment = self.analyze_sentiment(processed_text)
        summary = self.summarize_text(processed_text)
        keywords = self.extract_keywords(processed_text)
        semantic_complexity = self.calculate_semantic_complexity(processed_text)
        
        return {
            "processed_text": processed_text,
            "named_entities": named_entities,
            "sentiment": sentiment,
            "summary": summary,
            "keywords": keywords,
            "semantic_complexity": semantic_complexity
        }

# Create database tables
with app.app_context():
    db.create_all()

# Initialize the NLP processor as a global object
nlp_processor = NLPProcessor()
