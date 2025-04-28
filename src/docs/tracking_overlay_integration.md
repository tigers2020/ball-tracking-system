# Tracking Overlay Integration Guide

This guide explains how to integrate the new tracking coordinates overlay into your application. The tracking overlay shows 2D and 3D coordinates in real-time from HSV, Hough, and Kalman detection sources.

## Overview

The tracking overlay system consists of these components:

1. **TrackingOverlay** - UI widget that displays coordinate data
2. **CoordinateCombiner** - Service that combines coordinates from different sources
3. **TrackingCoordinatesController** - Controller that connects to BallTrackingController and updates the UI
4. **Integration utilities** - Helper functions to connect everything together

## Files

- `src/views/tracking_overlay.py` - The UI widget for displaying coordinates
- `src/services/coordinate_combiner.py` - Service for combining and triangulating coordinates
- `src/controllers/tracking_coordinates_controller.py` - Controller for updating tracking data
- `src/controllers/tracking_integration.py` - Helper functions for integration

## Integration Steps

### 1. Add to Existing ImageView

The tracking overlay is already integrated into the `ImageView` class. No additional code is needed if you're using this class as-is.

### 2. Manual Integration in Main Application

If you need to manually integrate the tracking overlay, follow these steps:

1. Import required modules:

```python
from src.controllers.tracking_integration import setup_tracking_overlay
```

2. Set up the tracking overlay in your main application code:

```python
# Assuming you have:
# - app: Your main application window
# - ball_tracker: BallTrackingController instance
# - config_manager: ConfigManager instance
# - image_view: ImageView instance

# Initialize tracking overlay
tracking_coord_controller = setup_tracking_overlay(
    app_window=app,
    ball_tracking_controller=ball_tracker,
    config_manager=config_manager,
    image_view=image_view
)
```

### 3. Custom UI Integration

If you have a custom UI layout and need to integrate the tracking overlay separately:

```python
from src.views.tracking_overlay import TrackingOverlay
from src.controllers.tracking_integration import connect_existing_tracking_overlay

# Create tracking overlay widget
tracking_overlay = TrackingOverlay()

# Add to your layout
your_layout.addWidget(tracking_overlay)

# Connect to controllers
tracking_coord_controller = connect_existing_tracking_overlay(
    tracking_overlay=tracking_overlay,
    ball_tracking_controller=ball_tracker,
    config_manager=config_manager
)
```

## Customization Options

### Changing Update Interval

You can adjust how often the tracking overlay updates:

```python
# Update every 200ms instead of the default
tracking_coord_controller.set_update_interval(200)
```

### Toggle Visibility

You can enable/disable the tracking overlay programmatically:

```python
# If using ImageView
image_view.enable_tracking_overlay(True)  # or False to hide

# Direct access to widget
tracking_overlay.setVisible(True)  # or False to hide
```

## Troubleshooting

### No Data Displayed

If the tracking overlay shows "No tracking" or doesn't display coordinates:

1. Check if BallTrackingController is properly initialized and enabled
2. Verify that detection signals are being emitted
3. Check logs for errors related to tracking or coordinate calculation
4. Make sure camera settings are correctly loaded in config_manager

### Performance Issues

If the overlay updates cause performance problems:

1. Increase the update interval (default is 100ms)
2. Use `tracking_coord_controller.stop_updates()` during intensive operations
3. Re-enable with `tracking_coord_controller.start_updates()` when done

## Integration Example

Here's a complete integration example for a main application:

```python
from src.controllers.ball_tracking_controller import BallTrackingController
from src.controllers.tracking_integration import setup_tracking_overlay
from src.utils.config_manager import ConfigManager
from src.views.image_view import ImageView

class MainApplication:
    def __init__(self):
        # Initialize components
        self.config_manager = ConfigManager()
        self.ball_tracker = BallTrackingController(None, self.config_manager)
        self.image_view = ImageView()
        
        # Set up tracking overlay
        self.tracking_controller = setup_tracking_overlay(
            app_window=self,
            ball_tracking_controller=self.ball_tracker,
            config_manager=self.config_manager,
            image_view=self.image_view
        )
        
        # Connect image view to ball tracking controller
        self.image_view.connect_ball_tracking_controller(self.ball_tracker)
``` 