# VideoAsk to Rev AI Integration

This project creates a webhook server that connects VideoAsk responses with Rev AI for transcription, processes the transcripts using advanced NLP, and stores results in both a database and Google Sheets.

## Features

- 🎤 **VideoAsk Webhook Integration**: Process video/audio responses
- 🔄 **Rev AI Transcription**: Convert audio to text with professional accuracy
- 🧠 **Advanced NLP Processing**: Extract insights from transcriptions
- 📊 **Google Sheets Integration**: Automatically update spreadsheets
- 🔄 **GitHub Webhook Integration**: Enable CI/CD and version control
- 🔧 **Local Development with Smee.io**: Test webhooks locally

## Setup Instructions

### Using GitHub Codespaces (Recommended)

1. Click on the "Code" button in this repository
2. Select the "Codespaces" tab
3. Click "Create codespace on main"
4. Wait for the environment to build (this may take a few minutes)
5. Once loaded, the environment will:
   - Install all dependencies
   - Set up the Smee.io webhook forwarding
   - Start the Flask application

### Local Development Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/videoask-revai-integration.git
   cd videoask-revai-integration
   ```

2. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. Edit the `.env` file with your configuration values

4. Start the Flask server:
   ```
   python app.py
   ```

5. In a separate terminal, start the webhook forwarding:
   ```
   node smee-setup.js
   ```

## Webhook Configuration

### VideoAsk Webhook Setup

1. Go to your VideoAsk project settings
2. Navigate to Integrations > Custom Webhook
3. Enter your endpoint URL (or the Smee.io URL for local development)
4. Set the secret in your `.env` file

### Rev AI Webhook Setup

1. In your Rev AI dashboard, go to Settings > API
2. Configure the callback URL to point to your `/webhook/rev-ai` endpoint
3. Add `?transcription_id={id}` parameter for identification

### GitHub Webhook Setup

1. In your GitHub repository, go to Settings > Webhooks
2. Add new webhook with the Smee.io URL: `https://smee.io/NXoLZTqSCKr2j4T`
3. Select events: Push, Pull requests
4. Set the secret in your `.env` file

## Environment Variables

Copy `.env.example` to `.env` and fill in your values:

- `VIDEOASK_WEBHOOK_SECRET`: Secret for VideoAsk webhook verification
- `VIDEOASK_API_KEY`: API key for VideoAsk
- `REV_AI_ACCESS_TOKEN`: Access token for Rev AI
- `REV_AI_CALLBACK_URL`: URL for Rev AI callbacks
- `DATABASE_URL`: Database connection string
- `GOOGLE_SHEETS_CREDENTIALS`: Path to Google API credentials JSON
- `GOOGLE_SHEETS_ID`: ID of your Google Sheet
- `GITHUB_WEBHOOK_SECRET`: Secret for GitHub webhook verification

## Development with Smee.io

This project uses Smee.io to forward webhooks to your local environment:

```
# Using the CLI
npm install --global smee-client
smee -u https://smee.io/NXoLZTqSCKr2j4T -t http://localhost:5000/webhook/github

# Or using the Node.js script
node smee-setup.js
```

## Deployment

The project includes a GitHub Action workflow for CI/CD in `.github/workflows/ci-cd.yml`. It:

1. Runs tests and linting checks
2. Builds a Docker image
3. Pushes the image to GitHub Container Registry
4. Sends a deployment notification

## Architecture

![System Architecture](https://via.placeholder.com/800x400?text=VideoAsk+RevAI+Integration+Architecture)

- **Flask Server**: Core application handling webhooks and business logic
- **SQLAlchemy**: ORM for database operations
- **NLP Processor**: Combines standard and advanced text processing
- **Webhook Handlers**: Process events from VideoAsk, Rev AI, and GitHub

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

MIT License
