#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Project Information Tab module.
This module contains the ProjectInfoTab class which provides information about 
current features, coming soon features, and development plan.
"""

import logging
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QIcon, QPixmap, QLinearGradient, QBrush, QPalette
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QTabWidget, QTextBrowser, QFrame
)

from src.utils.ui_constants import Layout
from src.utils.ui_theme import Colors, StyleManager

logger = logging.getLogger(__name__)

class StyledTextBrowser(QTextBrowser):
    """Custom styled text browser for fancy project info display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("""
            QTextBrowser {
                background-color: #1a1a2e;
                color: #e0e0e0;
                border: none;
                font-family: 'Segoe UI', Arial, sans-serif;
                padding: 10px;
            }
            QScrollBar:vertical {
                background-color: #1a1a2e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a4a82;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6aaa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

class FancyTabWidget(QTabWidget):
    """Custom styled tab widget with fancy appearance."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background-color: #151525;
                border-radius: 15px;
                box-shadow: 0 8px 30px rgba(0,0,0,0.5);
            }
            QTabWidget::tab-bar {
                alignment: center;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1c1c35, stop:1 #252555);
                color: #b0b0d0;
                border: none;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
                min-width: 180px;
                padding: 15px 20px;
                margin-right: 8px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #5d48e0, stop:1 #9048e0);
                color: #ffffff;
                font-size: 15px;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2f2f65, stop:1 #3a3a85);
                color: #d0d0ff;
            }
        """)

class ProjectInfoTab(QWidget):
    """
    Tab for displaying project information, current features, coming soon features, and development plan.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the project information tab.
        
        Args:
            parent (QWidget, optional): Parent widget
        """
        super().__init__(parent)
        
        # Set background color for entire tab
        self.setStyleSheet("""
            QWidget#projectInfoTab {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                                          stop:0 #12122a, stop:0.5 #151530, stop:1 #0d0d1f);
            }
            QLabel#titleLabel {
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-weight: bold;
                font-size: 28px;
                padding: 15px;
                letter-spacing: 1px;
            }
        """)
        self.setObjectName("projectInfoTab")
        
        # Set up the UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the user interface."""
        # Create main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(Layout.MARGIN+10, Layout.MARGIN+10, Layout.MARGIN+10, Layout.MARGIN+10)
        main_layout.setSpacing(Layout.SPACING+5)
        
        # Create title with gradient underline
        title_container = QWidget()
        title_container.setStyleSheet("""
            background-color: rgba(20, 20, 45, 0.7);
            border-radius: 15px;
            margin: 10px;
        """)
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(20, 15, 20, 15)
        
        title_label = QLabel("Ball Tracking System Overview")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        
        # Add gradient line
        gradient_line = QFrame()
        gradient_line.setFrameShape(QFrame.HLine)
        gradient_line.setFixedHeight(4)
        gradient_line.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                      stop:0 #5d48e0, stop:0.5 #b545e0, stop:1 #4848e0);
            border-radius: 2px;
            margin: 5px 50px;
        """)
        title_layout.addWidget(gradient_line)
        
        main_layout.addWidget(title_container)
        
        # Create nested tabs with custom styling
        self.nested_tabs = FancyTabWidget()
        main_layout.addWidget(self.nested_tabs)
        
        # Create Current Features tab
        self.current_features_widget = self._create_features_tab(
            self._get_current_features_html()
        )
        self.nested_tabs.addTab(self.current_features_widget, "Current Features")
        
        # Create Coming Soon tab
        self.coming_soon_widget = self._create_features_tab(
            self._get_coming_soon_features_html()
        )
        self.nested_tabs.addTab(self.coming_soon_widget, "Coming Soon")
        
        # Create Development Plan tab
        self.dev_plan_widget = self._create_features_tab(
            self._get_development_plan_html()
        )
        self.nested_tabs.addTab(self.dev_plan_widget, "Development Plan")
    
    def _create_features_tab(self, html_content):
        """
        Create a scrollable tab with HTML content.
        
        Args:
            html_content (str): HTML content to display
            
        Returns:
            QWidget: Tab widget
        """
        # Create widget and layout
        widget = QWidget()
        widget.setStyleSheet("""
            background-color: #151525;
            border-radius: 15px;
        """)
        
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Create custom text browser with HTML content
        text_browser = StyledTextBrowser()
        text_browser.setStyleSheet("""
            QTextBrowser {
                background-color: #131328;
                color: #e0e0e0;
                border: none;
                font-family: 'Segoe UI', Arial, sans-serif;
                padding: 15px;
                border-radius: 12px;
            }
            QScrollBar:vertical {
                background-color: #1a1a35;
                width: 14px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a4a92;
                min-height: 40px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6aaa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        text_browser.setOpenExternalLinks(True)
        text_browser.setHtml(html_content)
        
        # Create scroll area with custom styling
        scroll_area = QScrollArea()
        scroll_area.setWidget(text_browser)
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
                border-radius: 15px;
            }
        """)
        
        layout.addWidget(scroll_area)
        
        return widget
    
    def _get_current_features_html(self):
        """
        Get HTML content for current features.
        
        Returns:
            str: HTML content
        """
        return """
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                line-height: 1.6; 
                background-color: #131328; 
                color: #e0e0e0;
                margin: 0;
                padding: 15px;
            }
            .section-title {
                color: #a088ff;
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 25px;
                padding-bottom: 10px;
                position: relative;
                letter-spacing: 1px;
            }
            .section-title:after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 80px;
                height: 3px;
                background: linear-gradient(90deg, #5d48e0, #b545e0);
                border-radius: 3px;
            }
            .features-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 25px;
                padding: 10px 5px;
            }
            .feature-card {
                background: linear-gradient(145deg, #191940, #151533);
                border-radius: 15px;
                padding: 25px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.3);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                border: 1px solid rgba(100, 100, 255, 0.1);
                position: relative;
                overflow: hidden;
            }
            .feature-card:before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 5px;
                height: 100%;
                background: linear-gradient(to bottom, #5d48e0, #b545e0);
                border-top-left-radius: 15px;
                border-bottom-left-radius: 15px;
            }
            .feature-card:hover {
                transform: translateY(-10px) scale(1.02);
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
                border: 1px solid rgba(100, 100, 255, 0.2);
            }
            .feature-card:hover:after {
                opacity: 1;
            }
            .feature-card:after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: radial-gradient(circle at top right, rgba(120, 100, 255, 0.1), transparent 70%);
                opacity: 0;
                transition: opacity 0.4s ease;
            }
            .feature-header {
                display: flex;
                align-items: center;
                margin-bottom: 18px;
                position: relative;
                z-index: 1;
            }
            .feature-icon {
                width: 45px;
                height: 45px;
                background: linear-gradient(135deg, #5d48e0, #9048e0);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-right: 15px;
                font-weight: bold;
                color: white;
                font-size: 20px;
                box-shadow: 0 5px 15px rgba(93, 72, 224, 0.4);
            }
            .feature { 
                font-weight: bold; 
                color: #80d8ff; 
                font-size: 18px;
                flex: 1;
                letter-spacing: 0.5px;
            }
            .description { 
                color: #c5d0e0; 
                padding: 15px 0 5px 60px;
                border-top: 1px solid rgba(100, 100, 255, 0.15);
                margin-top: 10px;
                font-size: 15px;
                position: relative;
                z-index: 1;
                line-height: 1.7;
            }
            .link { 
                color: #b545e0; 
                text-decoration: none;
                transition: all 0.3s ease;
                display: inline-block;
                margin-top: 12px;
                padding-left: 60px;
                font-weight: bold;
                position: relative;
                z-index: 1;
            }
            .link:hover { 
                color: #d27fff; 
                text-decoration: underline;
                transform: translateX(5px);
            }
        </style>
        
        <h1 class="section-title">Current System Capabilities</h1>
        
        <div class="features-container">
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">1</div>
                    <div class="feature">Stereo Image Pair Display</div>
                </div>
                <div class="description">Side-by-side view of left/right camera image pairs with synchronized display and high-resolution rendering</div>
                <a href="https://github.com/tigers2020/ball-tracking-system" class="link">GitHub Repository &rarr;</a>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">2</div>
                    <div class="feature">Advanced Playback Controls</div>
                </div>
                <div class="description">Comprehensive image sequence playback control (Play/Pause/Stop) with precise frame rate (FPS) adjustment and timeline navigation</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">3</div>
                    <div class="feature">Multi-stage Ball Detection Pipeline</div>
                </div>
                <div class="description">Sophisticated BallTrackingController with adaptive HSV masks, intelligent ROI processing, Hough circle detection, and Kalman filter stabilization</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">4</div>
                    <div class="feature">3D Triangulation Engine</div>
                </div>
                <div class="description">High-precision CalculationService for world-to-court coordinate transformation with sub-centimeter accuracy</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">5</div>
                    <div class="feature">PnP-based Camera Calibration</div>
                </div>
                <div class="description">Interactive calibration tab and PnP calibration widgets with automated optimization and correction tools</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">6</div>
                    <div class="feature">Comprehensive Game Analysis</div>
                </div>
                <div class="description">Integrated GameAnalyzer with advanced BounceDetector, net crossing detection, landing prediction, and real-time in/out detection</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">7</div>
                    <div class="feature">Tracking Data Storage</div>
                </div>
                <div class="description">Automatic saving in JSON and XML formats to the tracking_data directory with metadata enrichment and versioning</div>
            </div>
            
            <div class="feature-card">
                <div class="feature-header">
                    <div class="feature-icon">8</div>
                    <div class="feature">Rich Visualization Suite</div>
                </div>
                <div class="description">Professional-grade visualization components including InfoView, ImageView, and BounceOverlay with real-time trajectory rendering</div>
            </div>
        </div>
        """
    
    def _get_coming_soon_features_html(self):
        """
        Get HTML content for coming soon features.
        
        Returns:
            str: HTML content
        """
        return """
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                line-height: 1.6; 
                background-color: #131328; 
                color: #e0e0e0;
                margin: 0;
                padding: 15px;
            }
            .section-title {
                color: #a088ff;
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 25px;
                padding-bottom: 10px;
                position: relative;
                letter-spacing: 1px;
            }
            .section-title:after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 80px;
                height: 3px;
                background: linear-gradient(90deg, #5d48e0, #b545e0);
                border-radius: 3px;
            }
            .coming-soon-container {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 30px;
                padding: 20px 10px;
            }
            .feature-item {
                background: linear-gradient(145deg, #191940, #151533);
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 15px 35px rgba(0,0,0,0.3);
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                position: relative;
                border: 1px solid rgba(100, 100, 255, 0.1);
                display: flex;
                flex-direction: column;
                height: 100%;
            }
            .feature-item:hover {
                transform: translateY(-10px) scale(1.02);
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
                border: 1px solid rgba(100, 100, 255, 0.2);
            }
            .feature-item:hover .feature-title {
                background-position: -100% 0;
            }
            .feature-item:after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 5px;
                background: linear-gradient(90deg, #9048e0, #5d48e0);
            }
            .feature-content {
                padding: 25px;
                display: flex;
                flex-direction: column;
                flex-grow: 1;
                position: relative;
                z-index: 1;
            }
            .feature-content:before {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 150px;
                height: 150px;
                background: radial-gradient(circle, rgba(144, 72, 224, 0.1), transparent 70%);
                z-index: -1;
                border-radius: 50%;
            }
            .feature-headline {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            .feature-title { 
                font-weight: bold; 
                font-size: 18px;
                flex: 1;
                letter-spacing: 0.5px;
                background: linear-gradient(to right, #ff7eb9, #ff4aa5, #ff7eb9);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-size: 200% 100%;
                transition: background-position 0.8s ease;
            }
            .tag { 
                display: inline-block; 
                background: linear-gradient(90deg, #2d3b62, #3e5ca9);
                color: #ffffff; 
                padding: 5px 12px; 
                border-radius: 25px; 
                font-size: 13px;
                font-weight: bold;
                box-shadow: 0 5px 15px rgba(45, 59, 98, 0.4);
                letter-spacing: 0.5px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s ease;
            }
            .feature-item:hover .tag {
                transform: scale(1.05);
                box-shadow: 0 7px 20px rgba(45, 59, 98, 0.5);
            }
            .feature-description { 
                color: #c5d0e0; 
                border-top: 1px solid rgba(100, 100, 255, 0.15);
                padding-top: 15px;
                margin-top: 10px;
                line-height: 1.7;
                font-size: 15px;
                flex-grow: 1;
            }
            @media (max-width: 768px) {
                .coming-soon-container {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        
        <h1 class="section-title">Upcoming Enhancements</h1>
        
        <div class="coming-soon-container">
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">Ball Position-Based System Accuracy Analysis</div>
                        <span class="tag">Q2 2025</span>
                    </div>
                    <div class="feature-description">Advanced tracking error calculation and visualization for X, Y, Z axes in static camera environments with comprehensive statistical analysis</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">Camera Position Impact Analysis</div>
                        <span class="tag">Q2 2025</span>
                    </div>
                    <div class="feature-description">Sophisticated error evaluation based on baseline and camera angle variations with optimized placement recommendations</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">Coefficient of Restitution Calculation</div>
                        <span class="tag">Q2 2025</span>
                    </div>
                    <div class="feature-description">High-precision calculation and visualization of vertical velocity ratio before and after bounces with surface material analysis</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">LED Visualization System</div>
                        <span class="tag">Q2 2025</span>
                    </div>
                    <div class="feature-description">Responsive visual feedback with green LED for inbound, 10 Hz 50% duty-cycle red LED flashing for out-of-bounds determinations</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">5-Shot Serve/Volley In/Out Detection Logic</div>
                        <span class="tag">Q2 2025</span>
                    </div>
                    <div class="feature-description">Intelligent classification and detailed summary of 5 consecutive serves/volleys with pattern recognition and performance metrics</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">Instant Replay Functionality</div>
                        <span class="tag">Q3 2025</span>
                    </div>
                    <div class="feature-description">Comprehensive frame range selection playback and intuitive control UI with multiple viewing angles and slow-motion capabilities</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">Automated Report Generation</div>
                        <span class="tag">Q3 2025</span>
                    </div>
                    <div class="feature-description">Sophisticated PDF/HTML generation of accuracy graphs, restitution coefficients, and in/out statistics with customizable templates</div>
                </div>
            </div>
            
            <div class="feature-item">
                <div class="feature-content">
                    <div class="feature-headline">
                        <div class="feature-title">Documentation Enhancement</div>
                        <span class="tag">Ongoing</span>
                    </div>
                    <div class="feature-description">Comprehensive completion of user manuals, API documents, and interactive code architecture diagrams with searchable knowledge base</div>
                </div>
            </div>
        </div>
        """
    
    def _get_development_plan_html(self):
        """
        Get HTML content for development plan.
        
        Returns:
            str: HTML content
        """
        return """
        <style>
            body { 
                font-family: 'Segoe UI', Arial, sans-serif; 
                line-height: 1.6; 
                background-color: #131328; 
                color: #e0e0e0;
                margin: 0;
                padding: 15px;
            }
            .section-title {
                color: #a088ff;
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 25px;
                padding-bottom: 10px;
                position: relative;
                letter-spacing: 1px;
            }
            .section-title:after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 80px;
                height: 3px;
                background: linear-gradient(90deg, #5d48e0, #b545e0);
                border-radius: 3px;
            }
            .dev-plan-container {
                background: linear-gradient(145deg, #191940, #151533);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 15px 35px rgba(0,0,0,0.3);
                margin-bottom: 30px;
                padding: 0 0 10px 0;
                border: 1px solid rgba(100, 100, 255, 0.1);
                position: relative;
            }
            .dev-plan-container:before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 5px;
                background: linear-gradient(90deg, #5d48e0, #b545e0);
                z-index: 5;
            }
            .plan-header {
                background: linear-gradient(90deg, #293363, #3d5198);
                color: white;
                font-weight: bold;
                display: grid;
                grid-template-columns: 0.5fr 2fr 1.5fr 1fr;
                padding: 18px 20px;
                font-size: 15px;
                position: sticky;
                top: 0;
                z-index: 10;
                letter-spacing: 0.5px;
                text-transform: uppercase;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .plan-row {
                display: grid;
                grid-template-columns: 0.5fr 2fr 1.5fr 1fr;
                padding: 18px 20px;
                border-bottom: 1px solid rgba(100, 100, 255, 0.1);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .plan-row:hover {
                background-color: rgba(100, 100, 255, 0.05);
                transform: translateY(-2px);
            }
            .plan-row:hover:after {
                transform: translateY(0);
                opacity: 1;
            }
            .plan-row:after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 100%;
                height: 2px;
                background: linear-gradient(90deg, transparent, #5d48e0, #b545e0, transparent);
                transform: translateY(2px);
                opacity: 0;
                transition: all 0.3s ease;
            }
            .plan-row:last-child {
                border-bottom: none;
            }
            .phase-cell {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .phase-number {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #5d48e0, #9048e0);
                border-radius: 50%;
                font-weight: bold;
                color: white;
                font-size: 18px;
                box-shadow: 0 5px 15px rgba(93, 72, 224, 0.4);
                position: relative;
                z-index: 1;
                overflow: hidden;
            }
            .phase-number:after {
                content: '';
                position: absolute;
                width: 100%;
                height: 100%;
                background: linear-gradient(rgba(255, 255, 255, 0.2), transparent);
                top: 0;
                left: 0;
                border-radius: 50%;
            }
            .tasks-cell {
                padding-right: 15px;
            }
            .task-title {
                color: #80d8ff;
                font-weight: bold;
                margin-bottom: 10px;
                font-size: 16px;
                letter-spacing: 0.5px;
                position: relative;
                display: inline-block;
            }
            .task-title:after {
                content: '';
                position: absolute;
                bottom: -5px;
                left: 0;
                width: 0;
                height: 2px;
                background: linear-gradient(90deg, #5d48e0, transparent);
                transition: width 0.3s ease;
            }
            .plan-row:hover .task-title:after {
                width: 100%;
            }
            .task-details {
                color: #c5d0e0;
                padding-left: 15px;
                font-size: 14px;
                line-height: 1.8;
                border-left: 2px solid rgba(100, 100, 255, 0.2);
                margin-left: 5px;
            }
            .deliverables-cell {
                color: #c5d0e0;
                font-size: 14px;
                display: flex;
                align-items: center;
                padding-left: 10px;
                border-left: 1px dashed rgba(100, 100, 255, 0.15);
                line-height: 1.6;
            }
            .duration-cell {
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ffd966;
                font-weight: bold;
                font-size: 15px;
                letter-spacing: 0.5px;
            }
            
            .summary-panel {
                background: linear-gradient(145deg, #191940, #151533);
                border-radius: 16px;
                padding: 25px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.3);
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(100, 100, 255, 0.1);
            }
            .summary-panel:before {
                content: '';
                position: absolute;
                left: 0;
                top: 0;
                width: 5px;
                height: 100%;
                background: linear-gradient(to bottom, #5d48e0, #b545e0);
            }
            .summary-panel:after {
                content: '';
                position: absolute;
                top: 0;
                right: 0;
                width: 150px;
                height: 150px;
                background: radial-gradient(circle, rgba(144, 72, 224, 0.1), transparent 70%);
                z-index: 0;
                border-radius: 50%;
            }
            .summary-section {
                margin-bottom: 20px;
                position: relative;
                z-index: 1;
                padding-left: 15px;
            }
            .summary-title {
                color: #80d8ff;
                font-weight: bold;
                margin-bottom: 12px;
                font-size: 17px;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
            }
            .summary-title:before {
                content: '\u25C8';
                margin-right: 10px;
                color: #9048e0;
                font-size: 14px;
            }
            .summary-content {
                color: #c5d0e0;
                margin-left: 20px;
                font-size: 15px;
                line-height: 1.8;
            }
            .priority-item {
                color: #c5d0e0;
                margin: 12px 0;
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                justify-content: center;
            }
            .priority-arrow {
                color: #ff7eb9;
                margin: 0 8px;
                font-size: 16px;
            }
            .small-phase-number {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 28px;
                height: 28px;
                background: linear-gradient(135deg, #5d48e0, #9048e0);
                border-radius: 50%;
                font-weight: bold;
                color: white;
                font-size: 14px;
                box-shadow: 0 3px 10px rgba(93, 72, 224, 0.4);
                position: relative;
                z-index: 1;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            .small-phase-number:hover {
                transform: scale(1.15);
                box-shadow: 0 5px 15px rgba(93, 72, 224, 0.6);
            }
            .milestone {
                background: rgba(45, 45, 80, 0.5);
                padding: 15px 20px;
                margin: 15px 0;
                border-radius: 10px;
                border: 1px solid rgba(100, 100, 255, 0.15);
                position: relative;
                transition: all 0.3s ease;
            }
            .milestone:hover {
                background: rgba(50, 50, 90, 0.5);
                transform: translateY(-3px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            .milestone-name {
                color: #ffd966;
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 5px;
                letter-spacing: 0.5px;
            }
            .milestone-date {
                color: #ff7eb9;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            .milestone-description {
                margin-top: 8px;
                color: #c5d0e0;
            }
        </style>
        
        <h1 class="section-title">Development Roadmap</h1>
        
        <div class="dev-plan-container">
            <div class="plan-header">
                <div>Phase</div>
                <div>Key Tasks</div>
                <div>Deliverables</div>
                <div>Timeline</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">1</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Accuracy Analysis Module Design</div>
                    <div class="task-details">
                        - Define error calculation functions (static)<br>
                        - Design input/output data formats<br>
                        - Establish accuracy metrics and validation methods
                    </div>
                </div>
                <div class="deliverables-cell">
                    Comprehensive accuracy analysis module specification with validation methodology
                </div>
                <div class="duration-cell">1 week</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">2</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Accuracy Analysis Implementation & Visualization</div>
                    <div class="task-details">
                        - Implement error calculation logic<br>
                        - Develop matplotlib-based graph components<br>
                        - Create interactive error visualization dashboard
                    </div>
                </div>
                <div class="deliverables-cell">
                    Interactive accuracy graph UI with real-time error analysis and module test code
                </div>
                <div class="duration-cell">1 week</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">3</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Camera Motion Accuracy Extension</div>
                    <div class="task-details">
                        - Baseline change simulation functionality<br>
                        - Results comparison and layout<br>
                        - Camera positioning optimization algorithms
                    </div>
                </div>
                <div class="deliverables-cell">
                    Advanced motion analysis module with placement recommendations and simulation settings UI
                </div>
                <div class="duration-cell">1 week</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">4</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Restitution Coefficient Calculation Logic</div>
                    <div class="task-details">
                        - e = |v_after_z|/|v_before_z| calculation<br>
                        - Add e field to BounceEvent<br>
                        - Surface material analysis integration
                    </div>
                </div>
                <div class="deliverables-cell">
                    High-precision restitution coefficient report with surface comparison and UI display component
                </div>
                <div class="duration-cell">4 days</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">5</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">LED Visualization Component Implementation</div>
                    <div class="task-details">
                        - LED widget development<br>
                        - in_out_detected event integration<br>
                        - Configurable visual indicators
                    </div>
                </div>
                <div class="deliverables-cell">
                    Responsive LED simulation widget with customizable alert patterns
                </div>
                <div class="duration-cell">3 days</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">6</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Serve/Volley In/Out Detection Logic Development</div>
                    <div class="task-details">
                        - Add set logic to GameAnalyzer<br>
                        - Results statistics UI<br>
                        - Advanced pattern recognition algorithms
                    </div>
                </div>
                <div class="deliverables-cell">
                    Intelligent 5-shot judgment module with performance metrics and interactive statistics panel
                </div>
                <div class="duration-cell">1 week</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">7</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Instant Replay Feature Implementation</div>
                    <div class="task-details">
                        - Playback range markers, controller<br>
                        - Memory-based frame buffer<br>
                        - Multiple viewing angles and slow-motion capabilities
                    </div>
                </div>
                <div class="deliverables-cell">
                    Comprehensive Instant Replay widget with advanced playback controls and integration tests
                </div>
                <div class="duration-cell">1 week</div>
            </div>
            
            <div class="plan-row">
                <div class="phase-cell">
                    <div class="phase-number">8</div>
                </div>
                <div class="tasks-cell">
                    <div class="task-title">Integration Testing and Documentation</div>
                    <div class="task-details">
                        - Write E2E test scenarios<br>
                        - Create user/developer guides<br>
                        - Build interactive documentation platform
                    </div>
                </div>
                <div class="deliverables-cell">
                    Comprehensive test report, interactive user manuals, and developer documentation
                </div>
                <div class="duration-cell">1 week</div>
            </div>
        </div>
        
        <div class="summary-panel">
            <div class="summary-section">
                <div class="summary-title">Total Estimated Development Timeline</div>
                <div class="summary-content">Approximately 6 weeks with parallel development tracks</div>
            </div>
            
            <div class="summary-section">
                <div class="summary-title">Development Priority Sequence</div>
                <div class="summary-content">
                    <div class="priority-item">
                        <div class="small-phase-number">1</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">2</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">4</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">5</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">6</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">7</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">3</div>
                        <span class="priority-arrow">→</span>
                        <div class="small-phase-number">8</div>
                    </div>
                </div>
            </div>
            
            <div class="summary-section">
                <div class="summary-title">Team Assignments Strategy</div>
                <div class="summary-content">Core development team members will be assigned to specific modules based on expertise during weekly agile planning sessions</div>
            </div>
            
            <div class="summary-section">
                <div class="summary-title">Key Development Milestones</div>
                <div class="summary-content">
                    <div class="milestone">
                        <div class="milestone-name">PDR (Preliminary Design Review)</div>
                        <span class="milestone-date">1st week of April</span>
                        <div class="milestone-description">Detailed specification review with stakeholders and technical architecture validation</div>
                    </div>
                    <div class="milestone">
                        <div class="milestone-name">CDR (Critical Design Review)</div>
                        <span class="milestone-date">April 28, 11:00</span>
                        <div class="milestone-description">Comprehensive prototype demonstration and feature validation with performance metrics</div>
                    </div>
                </div>
            </div>
        </div>
        """ 