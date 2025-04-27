#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of the unified visualizer interface.
Shows how to use both OpenCV and Qt visualizers with the same code.
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import Qt

from src.views.visualization import VisualizerFactory

def run_opencv_example():
    """Run an example using the OpenCV visualizer."""
    # Create a blank image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Create an OpenCV visualizer
    visualizer = VisualizerFactory.create(backend="opencv")
    
    # Draw various elements
    points = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200), (300, 200)]
    img = visualizer.draw_points(img, points, numbered=True)
    img = visualizer.draw_grid_lines(img, points, rows=2, cols=3)
    
    # Draw ROI
    roi = (400, 100, 200, 150)
    img = visualizer.draw_roi(img, roi, show_center=True)
    
    # Draw circle
    img = visualizer.draw_circle(img, (500, 400), 80, label="Ball")
    
    # Draw trajectory
    trajectory = [(100, 400), (150, 380), (200, 370), (250, 390), (300, 420), (350, 440)]
    img = visualizer.draw_trajectory(img, trajectory)
    
    # Draw prediction
    img = visualizer.draw_prediction(
        img, 
        current_pos=(350, 440), 
        predicted_pos=(400, 450),
        draw_uncertainty=True
    )
    
    # Display the result
    cv2.imshow("OpenCV Visualizer Example", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

class QtVisualizerDemo(QMainWindow):
    """Demo application for Qt visualizer."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt Visualizer Example")
        self.resize(800, 600)
        
        # Create a QGraphicsScene and View
        self.scene = QGraphicsScene(0, 0, 800, 600)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(self.view.RenderHint.Antialiasing)
        
        # Create layout and set central widget
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Create a Qt visualizer
        self.visualizer = VisualizerFactory.create(backend="qt", scene=self.scene)
        
        # Draw the same elements as in the OpenCV example
        self.draw_demo_elements()
    
    def draw_demo_elements(self):
        """Draw demo visualization elements."""
        # Draw points
        points = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200), (300, 200)]
        self.visualizer.draw_points(self.scene, points, numbered=True)
        self.visualizer.draw_grid_lines(self.scene, points, rows=2, cols=3)
        
        # Draw ROI
        roi = (400, 100, 200, 150)
        self.visualizer.draw_roi(self.scene, roi, show_center=True)
        
        # Draw circle
        self.visualizer.draw_circle(self.scene, (500, 400), 80, label="Ball")
        
        # Draw trajectory
        trajectory = [(100, 400), (150, 380), (200, 370), (250, 390), (300, 420), (350, 440)]
        self.visualizer.draw_trajectory(self.scene, trajectory)
        
        # Draw prediction
        self.visualizer.draw_prediction(
            self.scene, 
            current_pos=(350, 440), 
            predicted_pos=(400, 450),
            draw_uncertainty=True
        )

def run_qt_example():
    """Run an example using the Qt visualizer."""
    app = QApplication([])
    window = QtVisualizerDemo()
    window.show()
    app.exec()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "qt":
        print("Running Qt visualizer example...")
        run_qt_example()
    else:
        print("Running OpenCV visualizer example...")
        run_opencv_example()
        
    print("Run with 'python visualizer_example.py qt' to see the Qt example.") 