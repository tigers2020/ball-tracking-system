# Unified Visualization System

This module provides a unified interface for drawing visual elements on different backends (OpenCV and Qt).

## Key Features

- Common interface for both OpenCV and Qt visualization
- Factory pattern for creating appropriate visualizer
- Support for points, lines, grids, ROIs, circles, trajectories, and predictions
- Consistent styling using constants from `src.utils.constants`

## Usage

### Basic Usage

```python
from src.views.visualization import VisualizerFactory

# For OpenCV backend
img = np.zeros((600, 800, 3), dtype=np.uint8)  # Create blank image
visualizer = VisualizerFactory.create(backend="opencv")
img = visualizer.draw_point(img, (100, 100))
img = visualizer.draw_circle(img, (200, 200), 50)

# For Qt backend
scene = QGraphicsScene()
visualizer = VisualizerFactory.create(backend="qt", scene=scene)
visualizer.draw_point(scene, (100, 100))
visualizer.draw_circle(scene, (200, 200), 50)
```

### Available Methods

All visualizers implement the `IVisualizer` interface with these methods:

- `draw_point(raw, point, color=None, radius=5, thickness=-1, label=None, cross_size=10)`
- `draw_points(raw, points, color=None, radius=5, thickness=-1, labels=None, numbered=False)`
- `draw_line(raw, pt1, pt2, color=None, thickness=2, dashed=False)`
- `draw_grid_lines(raw, points, rows, cols, color=None, thickness=2, dashed=False)`
- `draw_roi(raw, roi, color=None, thickness=TRACKING.ROI_THICKNESS, show_center=True)`
- `draw_circle(raw, center, radius, color=None, thickness=TRACKING.CIRCLE_THICKNESS, show_center=True, label=None)`
- `draw_circles(raw, circles, color=None, thickness=TRACKING.CIRCLE_THICKNESS, label_circles=False)`
- `draw_prediction(raw, current_pos, predicted_pos, arrow_color=None, thickness=TRACKING.PREDICTION_THICKNESS, draw_uncertainty=False, uncertainty_radius=TRACKING.UNCERTAINTY_RADIUS)`
- `draw_trajectory(raw, positions, color=None, thickness=TRACKING.TRAJECTORY_THICKNESS, max_points=TRACKING.TRAJECTORY_MAX_POINTS)`

### Legacy Support

For backward compatibility, the original direct drawing functions are still available:

```python
from src.views.visualization import draw_centroid, draw_roi, draw_circles, draw_prediction, draw_trajectory

# Use direct function calls
img = draw_centroid(img, (100, 100))
img = draw_roi(img, (50, 50, 100, 100))
```

## Implementation Details

- The `OpenCVVisualizer` wraps existing drawing functions from `src.utils.viz_utils`
- The `QtVisualizer` creates Qt graphics items and adds them to a `QGraphicsScene`
- The `VisualizerFactory` provides a unified creation mechanism for both visualizers

## Example

See `examples/visualizer_example.py` for a full example showing both backends.

## Migration Guide

To migrate from direct function calls to the new unified interface:

1. Replace imports:
   ```python
   # Old
   from src.views.visualization import draw_centroid, draw_roi
   
   # New
   from src.views.visualization import VisualizerFactory
   visualizer = VisualizerFactory.create(backend="opencv")
   ```

2. Replace function calls:
   ```python
   # Old
   img = draw_centroid(img, (100, 100))
   
   # New
   img = visualizer.draw_point(img, (100, 100))
   ``` 