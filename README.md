# Stereo Ball Tracking System

An application for viewing stereo image pairs and tracking a ball in stereo images. Built with PySide6 and OpenCV.

## Features

- View stereo image pairs side by side
- Play/pause/stop functionality for image sequences
- Adjustable playback speed (FPS)
- Advanced ball tracking with HSV color filtering
- Region of Interest (ROI) based processing
- Kalman filtering for stable ball trajectory prediction
- Save tracking data in JSON and XML formats
- Visual overlays for tracking visualization

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/stereo-ball-tracking.git
cd stereo-ball-tracking
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

4. Install the package in development mode (recommended for development):

```bash
pip install -e .
```

This ensures all modules are correctly resolved regardless of your working directory.

## Usage

### Running the application

```bash
python main.py
```

To run in demo mode with sample data:

```bash
python main.py --demo
```

### Loading images

1. Click on `File > Open Folder...` or press `Ctrl+O`
2. Select a folder containing stereo image pairs

The application will look for image files and group them into left/right pairs. If no explicit naming convention is found (like `_L`/`_R` suffixes or `left`/`right` in the name), it will pair them by order (even/odd indices).

### Ball Tracking

1. Load a sequence of stereo images
2. Click the "Ball Tracking" button in the control panel
3. Adjust HSV values to isolate the ball color using the HSV Settings dialog
4. Enable/disable ROI as needed
5. Ball tracking data will be automatically saved to the `tracking_data/` directory

### HSV Color Tuning

For optimal ball detection, you need to tune the HSV parameters to match the ball color:

1. Click "Settings" in the ball tracking panel
2. Adjust the HSV sliders:
   - H (Hue): Color tone (0-179 in OpenCV)
   - S (Saturation): Color intensity (0-255)
   - V (Value): Brightness (0-255)
3. Watch the mask preview to see if the ball is well isolated
4. Adjust blur and morphological operations if needed
5. Click "Apply" to save settings

### ROI Settings

Region of Interest settings allow for faster processing and more accurate detection:

1. Enable ROI in the settings
2. Set appropriate width and height (recommended: 150x150 pixels)
3. Enable auto-center to make the ROI follow the detected object

### Camera Calibration

For 3D reconstruction, camera parameters can be set:

1. Go to Settings > Camera Settings
2. Enter the camera parameters, focal length, and baseline
3. Calibration parameters are used for 3D position estimation

## File Structure

- `main.py`: Application entry point
- `src/`: Main source code directory
  - `models/`: Data models
  - `views/`: UI components
  - `controllers/`: Controllers connecting models and views
  - `services/`: Core image processing and tracking services
  - `utils/`: Utility functions and constants
  - `resources/`: Resource files (icons, etc.)
- `config/`: Configuration files
- `test_data/`: Sample test data
- `tracking_data/`: Directory where tracking results are saved

## Development

### Running tests

```bash
pytest
```

For performance tests of HSV processing:

```bash
python test_hsv_performance.py
```

### Troubleshooting

If you encounter issues with ball detection:

1. Check that HSV values correctly isolate the ball
2. Ensure ROI is enabled and properly sized
3. Verify that Hough circle parameters match the expected ball size
4. Check the console logs for warnings about detection rates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PySide6 for the UI framework
- OpenCV for image processing and ball detection
- Kalman filtering for stable trajectory prediction 