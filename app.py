"""
VideoAsk to Rev AI Integration Webhook Server
"""

import os
from flask import Flask
from dotenv import load_dotenv
from videoask_rev_ai_config import Config, db, app
from main import AdvancedTextProcessor

# Enhanced NLP processor that integrates both codebases
class EnhancedNLPProcessor:
    def __init__(self):
        """Initialize the enhanced NLP processor"""
        # Import the original processor from videoask_rev_ai_config
        from videoask_rev_ai_config import NLPProcessor
        self.base_processor = NLPProcessor()
        
        # Import the advanced processor from main.py
        self.advanced_processor = AdvancedTextProcessor()
    
    def process_transcript(self, transcript):
        """Process transcript with enhanced NLP capabilities"""
        if not transcript:
            return self.base_processor.process_transcript(transcript)
        
        # Get base processing results
        base_results = self.base_processor.process_transcript(transcript)
        
        # Add advanced processing
        advanced_analysis = self.advanced_processor.comprehensive_analysis(transcript)
        
        # Merge results
        enhanced_results = base_results.copy()
        enhanced_results["semantic_similarity"] = {}
        enhanced_results["advanced_named_entities"] = advanced_analysis["named_entities"]
        enhanced_results["advanced_summary"] = advanced_analysis["summary"]
        
        return enhanced_results

# Replace the NLP processor with the enhanced version
from videoask_rev_ai_config import nlp_processor
nlp_processor = EnhancedNLPProcessor()

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Set up database
    with app.app_context():
        db.create_all()
    
    # Run the app
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG", "False").lower() == "true")
