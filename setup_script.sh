#!/bin/bash
# Setup script for VideoAsk-RevAI integration project

echo "Setting up VideoAsk-RevAI integration environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download spaCy model
python -m spacy download en_core_web_sm

# Setup environment file
if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please edit .env file with your actual configuration values."
fi

# Install Node.js dependencies for webhook forwarding
npm install --save smee-client dotenv

# Create tests directory if it doesn't exist
mkdir -p tests

# Initialize Git repository if not already initialized
if [ ! -d .git ]; then
    git init
    echo "Git repository initialized."
    echo "venv/" > .gitignore
    echo "__pycache__/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo "*.pyo" >> .gitignore
    echo "*.sqlite3" >> .gitignore
    echo ".env" >> .gitignore
    echo "*.db" >> .gitignore
    echo "node_modules/" >> .gitignore
    echo "*.log" >> .gitignore
fi

# Create database directory
mkdir -p instance

echo "Setup complete! To start the application:"
echo "1. Edit the .env file with your configuration"
echo "2. Run 'python app.py' to start the Flask server"
echo "3. In a separate terminal, run 'node smee-setup.js' to forward webhooks"
