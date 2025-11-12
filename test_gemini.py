#!/usr/bin/env python3
"""Simple test script to verify Gemini API is working via litellm."""

import os
from dotenv import load_dotenv
import litellm

# Load environment variables
load_dotenv()

# Check if API key is set
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(f"GEMINI_API_KEY present: {bool(GEMINI_API_KEY)}")
if GEMINI_API_KEY:
    print(f"GEMINI_API_KEY length: {len(GEMINI_API_KEY)}")
    print(f"GEMINI_API_KEY prefix: {GEMINI_API_KEY[:10]}...")
else:
    print("ERROR: GEMINI_API_KEY not found in environment")
    exit(1)

print("\n" + "="*60)
print("Testing Gemini API via litellm...")
print("="*60)

# Test 1: Simple text generation
print("\nTest 1: Simple text generation")
try:
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
        api_key=GEMINI_API_KEY,
        temperature=0.0,
    )
    print(f"✓ Success! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: With gemini-2.5-pro
print("\nTest 2: With gemini-2.5-pro model")
try:
    response = litellm.completion(
        model="gemini/gemini-2.5-pro",
        messages=[{"role": "user", "content": "Say 'hello from pro' and nothing else."}],
        api_key=GEMINI_API_KEY,
        temperature=0.0,
    )
    print(f"✓ Success! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Check if litellm can find the key automatically
print("\nTest 3: Without explicit api_key (should use env var)")
try:
    response = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Say 'auto key works' and nothing else."}],
        temperature=0.0,
    )
    print(f"✓ Success! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)
