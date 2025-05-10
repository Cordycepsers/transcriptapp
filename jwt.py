#!/usr/bin/env python3
import sys
import time

import jwt

# Get PEM file path
if len(sys.argv) > 1:
    pem = sys.argv[1]
else:
    pem = "/transcriptapp/venv/lib/python3.12/site-packages/pip/_vendor/certifi/cacert.pem"

# Get the Client ID
if len(sys.argv) > 2:
    client_id = sys.argv[2]
else:
    client_id = "v23liNJaYkrqEwIxJu9"
    # client_id = "Iv23liNJaYkrqEwIxJu9")

# Open PEM
with open(pem, 'rb') as pem_file:
    signing_key = pem_file.read()

payload = {
    # Issued at time
    'iat': int(time.time()),
    # JWT expiration time (10 minutes maximum)
    'exp': int(time.time()) + 600,
    
    # GitHub App's client ID
    'iss': client_id
}

# Create JWT
encoded_jwt = jwt.encode(payload, signing_key, algorithm='RS256')

print(f"JWT: {encoded_jwt}")
