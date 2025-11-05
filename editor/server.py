#!/usr/bin/env python3
"""Simple local server for the spec editor with save functionality."""

from flask import Flask, send_from_directory, request, jsonify, make_response
from flask_cors import CORS
from pathlib import Path
import json

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
