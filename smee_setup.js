/**
 * Webhook forwarding setup script using smee-client
 * 
 * This script sets up a smee client to forward GitHub webhooks to your local dev environment
 */

const SmeeClient = require('smee-client');
require('dotenv').config();

// Get the webhook proxy URL from environment variables or use the default
const source = process.env.WEBHOOK_PROXY_URL || 'https://smee.io/NXoLZTqSCKr2j4T';
const target = process.env.WEBHOOK_TARGET || 'http://localhost:5000/webhook/github';

console.log(`Starting webhook forwarding from ${source} to ${target}`);

// Create smee client
const smee = new SmeeClient({
  source,
  target,
  logger: console
});

// Start forwarding
const events = smee.start();

// Handle termination signals
process.on('SIGINT', function() {
  console.log('Stopping webhook forwarding');
  events.close();
  process.exit();
});

console.log('Webhook forwarding is active. Press Ctrl+C to stop.');
