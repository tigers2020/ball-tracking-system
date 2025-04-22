# Stereo Image Player

A simple application for viewing and playing stereo image pairs. Built with PySide6 and OpenCV.

## Features

- View stereo image pairs side by side
- Play/pause/stop functionality for image sequences
- Adjustable playback speed (FPS)
- Automatic generation of frames_info.xml for image directories
- Drag and drop support for easy loading of image folders

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stereo-image-player.git
cd stereo-image-player
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the application

```bash
python main.py
```

### Loading images

1. Click on `File > Open Folder...` or press `Ctrl+O`
2. Select a folder containing stereo image pairs

The application will look for image files and group them into left/right pairs. If no explicit naming convention is found (like `_L`/`_R` suffixes or `left`/`right` in the name), it will pair them by order (even/odd indices).

### Playback controls

- **Play/Pause**: Start or pause the playback
- **Stop**: Stop the playback and return to the first frame
- **Previous/Next**: Navigate to the previous or next frame
- **Slider**: Drag to navigate to a specific frame
- **FPS**: Adjust the playback speed by changing the FPS value

## File Structure

- `main.py`: Application entry point
- `src/`: Main source code directory
  - `models/`: Data models
  - `views/`: UI components
  - `controllers/`: Controllers connecting models and views
  - `utils/`: Utility functions and constants
  - `resources/`: Resource files (icons, etc.)

## Development

### Running tests

```bash
pytest
```

### Building an executable

```bash
pyinstaller --name "StereoImagePlayer" --windowed --onefile main.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PySide6 for the UI framework
- OpenCV for image processing capabilities 