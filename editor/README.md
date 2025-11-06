# Spec Editor

Interactive web-based editor for viewing and editing design specifications.

## Quick Start

### Easy Way (Double-Click)

**Mac/Linux:** Double-click `launch.sh`
**Windows:** Double-click `launch.bat`
**Any OS:** Double-click `launch.py`

The launcher will:
- ✓ Check dependencies and install if needed
- ✓ Start the server on port 5001
- ✓ Open your browser automatically

### Command Line

```bash
python3 launch.py
```

## First Time Setup

**Requirements:**
- Python 3.8 or higher

The launcher will automatically install these packages if missing:
- flask
- flask-cors
- requests
- python-dotenv

## Using the Editor

Once the server starts, your browser will open to `http://localhost:5001`

**Interface:**
- **Left Sidebar** - List of all designs (click to view)
- **Center** - Original image vs Reconstruction preview
- **Right Panel** - Properties (appears when you select an element)
- **Bottom Right** - Control buttons

**Editing:**
- Click any element to select it
- Drag to move
- Drag corner handles to resize
- Edit properties in right panel
- Press Delete/Backspace to remove

**Layer Control:**
- Use layer buttons to reorder elements
- ⬆️ To Front / ↑ Forward / ↓ Backward / ⬇️ To Back

**Image Tools:**
- **Remove Background** - AI-powered background removal
- **Make Background** - Convert image to canvas background
- **Download Image** - Save the image file

**Saving:**
- Changes auto-detect
- Click "Save Spec" button when ready
- Button highlights when there are unsaved changes

## Stopping the Server

Press `Ctrl+C` in the terminal window

## Troubleshooting

**Port Already in Use:**
```bash
# Mac/Linux
lsof -ti:5001 | xargs kill -9

# Windows
netstat -ano | findstr :5001
# Then: taskkill /PID <pid> /F
```

**Dependencies Won't Install:**
```bash
pip install flask flask-cors requests python-dotenv
```

**Browser Doesn't Open:**
Manually navigate to: http://localhost:5001

## Environment Variables (Optional)

For the **Remove Background** feature, create a `.env` file in the project root:

```
FAL_API_KEY=your_fal_key_here
```

Get a free API key at: https://fal.ai/

**Without an API key you can still:**
- ✓ View and edit all specs
- ✓ Edit text/images/SVG elements
- ✓ Move, resize, and reorder layers
- ✓ Add new SVGs and images
- ✓ Save all changes
- ✗ Remove backgrounds from images (requires FAL_API_KEY)

**Note:** GEMINI_API_KEY is NOT needed for the editor. It's only used by other scripts for aesthetic evaluation and SVG generation.
