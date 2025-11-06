#!/usr/bin/env python3
"""Simple local server for the spec editor with save functionality."""

from flask import Flask, send_from_directory, request, jsonify, make_response
from flask_cors import CORS
from pathlib import Path
import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for local development

# Disable caching for all responses
@app.after_request
def add_no_cache_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Base directory (parent of editor/)
BASE_DIR = Path(__file__).parent.parent

# Save spec endpoint (must be before catch-all route)
@app.route('/api/save-spec', methods=['POST'])
def save_spec():
    try:
        data = request.json
        spec_name = data.get('specName')
        spec_content = data.get('spec')

        if not spec_name or not spec_content:
            return jsonify({'error': 'Missing specName or spec'}), 400

        # Construct the path to the spec file
        spec_path = BASE_DIR / 'datasets' / 'specs' / spec_name / 'spec.json'

        # Ensure the directory exists
        spec_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the spec file
        with open(spec_path, 'w') as f:
            json.dump(spec_content, f, indent=2)

        return jsonify({'success': True, 'path': str(spec_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Read SVG file
@app.route('/api/read-svg', methods=['GET'])
def read_svg():
    try:
        spec_name = request.args.get('specName')
        filename = request.args.get('filename')

        if not spec_name or not filename:
            return jsonify({'error': 'Missing specName or filename'}), 400

        # Construct the path to the SVG file
        svg_path = BASE_DIR / 'datasets' / 'specs' / spec_name / filename

        if not svg_path.exists():
            return jsonify({'error': 'SVG file not found'}), 404

        # Read the SVG file
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_content = f.read()

        return jsonify({'success': True, 'content': svg_content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Write SVG file
@app.route('/api/save-svg', methods=['POST'])
def save_svg():
    try:
        data = request.json
        spec_name = data.get('specName')
        filename = data.get('filename')
        content = data.get('content')

        if not spec_name or not filename or content is None:
            return jsonify({'error': 'Missing specName, filename, or content'}), 400

        # Construct the path to the SVG file
        svg_path = BASE_DIR / 'datasets' / 'specs' / spec_name / filename

        # Ensure the directory exists
        svg_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the SVG file
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return jsonify({'success': True, 'path': str(svg_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Convert image to background
@app.route('/api/make-background', methods=['POST'])
def make_background():
    try:
        data = request.json
        spec_name = data.get('specName')
        filename = data.get('filename')

        if not spec_name or not filename:
            return jsonify({'error': 'Missing specName or filename'}), 400

        spec_dir = BASE_DIR / 'datasets' / 'specs' / spec_name
        source_path = spec_dir / filename
        dest_path = spec_dir / 'background.png'

        if not source_path.exists():
            return jsonify({'error': 'Source image not found'}), 404

        # Copy the image to background.png
        import shutil
        shutil.copy2(source_path, dest_path)

        return jsonify({'success': True, 'path': str(dest_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Remove background from image
@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    try:
        data = request.json
        spec_name = data.get('specName')
        filename = data.get('filename')

        if not spec_name or not filename:
            return jsonify({'error': 'Missing specName or filename'}), 400

        spec_dir = BASE_DIR / 'datasets' / 'specs' / spec_name
        image_path = spec_dir / filename

        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404

        # Get FAL API key
        fal_api_key = os.getenv('FAL_API_KEY')
        if not fal_api_key:
            return jsonify({'error': 'FAL_API_KEY not configured'}), 500

        # Convert image to data URL
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        # Determine mime type
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
        }
        mime_type = mime_types.get(image_path.suffix.lower(), 'image/png')
        image_url = f"data:{mime_type};base64,{image_data}"

        # Call Bria RMBG API
        headers = {'Authorization': f'Key {fal_api_key}'}
        payload = {'image_url': image_url}

        response = requests.post(
            'https://fal.run/fal-ai/bria/background/remove',
            json=payload,
            headers=headers,
            timeout=120
        )
        response.raise_for_status()

        # Handle async job or immediate response
        if response.status_code == 200:
            result = response.json()
        elif response.status_code == 202:
            job = response.json()
            status_url = job.get('status_url') or job.get('response_url')

            # Poll for completion
            while True:
                import time
                time.sleep(2)
                status_response = requests.get(status_url, headers=headers, timeout=120)
                status_response.raise_for_status()
                result = status_response.json()
                state = (result.get('status') or result.get('state') or '').lower()

                if state in ('completed', 'success', 'succeeded'):
                    break
                if state in ('failed', 'error'):
                    return jsonify({'error': f'Background removal failed: {result}'}), 500
        else:
            return jsonify({'error': f'Unexpected response: {response.status_code}'}), 500

        # Download the result image
        if 'image' in result and 'url' in result['image']:
            result_url = result['image']['url']
        elif 'images' in result and len(result['images']) > 0:
            result_url = result['images'][0]['url']
        else:
            return jsonify({'error': 'No output image in result'}), 500

        # Download and save
        image_response = requests.get(result_url, timeout=60)
        image_response.raise_for_status()

        # Save back to the same file
        with open(image_path, 'wb') as f:
            f.write(image_response.content)

        return jsonify({'success': True, 'path': str(image_path)})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Create new SVG file
@app.route('/api/create-svg', methods=['POST'])
def create_svg():
    try:
        data = request.json
        spec_name = data.get('specName')

        if not spec_name:
            return jsonify({'error': 'Missing specName'}), 400

        # Get existing SVG files to determine next index
        spec_dir = BASE_DIR / 'datasets' / 'specs' / spec_name
        spec_dir.mkdir(parents=True, exist_ok=True)

        # Find next available SVG number
        existing_svgs = list(spec_dir.glob('svg-*.svg'))
        if existing_svgs:
            indices = []
            for svg in existing_svgs:
                try:
                    idx = int(svg.stem.split('-')[1])
                    indices.append(idx)
                except:
                    pass
            next_idx = max(indices) + 1 if indices else 1
        else:
            next_idx = 1

        # Create default SVG content
        default_svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" fill="#3498db" rx="10"/>
  <text x="50" y="55" font-size="24" fill="white" text-anchor="middle" font-family="Arial">SVG</text>
</svg>'''

        # Save with next available filename
        filename = f'svg-{next_idx}.svg'
        file_path = spec_dir / filename
        file_path.write_text(default_svg, encoding='utf-8')

        return jsonify({'success': True, 'filename': filename, 'path': str(file_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Upload image file
@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    try:
        spec_name = request.form.get('specName')

        if not spec_name:
            return jsonify({'error': 'Missing specName'}), 400

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Get existing asset files to determine next index
        spec_dir = BASE_DIR / 'datasets' / 'specs' / spec_name
        spec_dir.mkdir(parents=True, exist_ok=True)

        # Find next available asset number
        existing_assets = list(spec_dir.glob('asset-*.png'))
        if existing_assets:
            indices = []
            for asset in existing_assets:
                try:
                    idx = int(asset.stem.split('-')[1])
                    indices.append(idx)
                except:
                    pass
            next_idx = max(indices) + 1 if indices else 1
        else:
            next_idx = 1

        # Save with next available filename
        filename = f'asset-{next_idx}.png'
        file_path = spec_dir / filename
        file.save(str(file_path))

        return jsonify({'success': True, 'filename': filename, 'path': str(file_path)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve static files from editor directory
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    # Serve files from editor directory or parent datasets directory
    if path.startswith('datasets/'):
        # Serve from parent directory for datasets
        return send_from_directory(BASE_DIR, path)
    else:
        # Serve from editor directory
        return send_from_directory('.', path)

if __name__ == '__main__':
    print("Starting spec editor server...")
    print("Open http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop")
    # Disable reloader to avoid path issues
    app.run(host='localhost', port=5001, debug=False)
