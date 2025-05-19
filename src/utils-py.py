#!/usr/bin/env python
import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Utils:
    """Utility functions for the Steel Surface Inspection Robot"""
    
    @staticmethod
    def load_image(image_path):
        """Load an image and convert to RGB"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        return Image.open(image_path).convert('RGB')
    
    @staticmethod
    def visualize_detection(image, boxes, classes, confidences, class_names, title=None):
        """
        Visualize detection results
        
        Args:
            image: PIL Image or numpy array
            boxes: List of bounding boxes [x1, y1, x2, y2]
            classes: List of class indices
            confidences: List of confidence scores
            class_names: Dictionary mapping class indices to class names
            title: Title for the plot
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        
        colors = {
            'steel_rust': (255, 0, 0),      # Red
            'steel_clean': (0, 255, 0),     # Green
            'nonsteel_rust': (255, 165, 0), # Orange
            'nonsteel_clean': (0, 0, 255)   # Blue
        }
        
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box
            
            cls_name = class_names.get(cls, f"Class_{cls}")
            color = colors.get(cls_name, (200, 200, 200))
            
            # Convert BGR to RGB for matplotlib
            color_rgb = (color[0]/255, color[1]/255, color[2]/255)
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor=color_rgb, linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(x1, y1-10, f"{cls_name} {conf:.2f}", 
                    color=color_rgb, backgroundcolor=(1, 1, 1, 0.5),
                    fontsize=8, weight='bold')
        
        if title:
            plt.title(title)
        plt.axis('off')
        
        return plt.gcf()
    
    @staticmethod
    def visualize_edge_map(image, edge_map, title=None):
        """
        Visualize edge detection results
        
        Args:
            image: PIL Image or numpy array
            edge_map: Binary edge map
            title: Title for the plot
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(edge_map, cmap='gray')
        plt.title("Edge Map")
        plt.axis('off')
        
        if title:
            plt.suptitle(title)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def calculate_rust_percentage(mask):
        """
        Calculate the percentage of rust in a binary mask
        
        Args:
            mask: Binary mask where 1 indicates rust
            
        Returns:
            float: Percentage of rust (0-100)
        """
        if mask.size == 0:
            return 0.0
        
        return (np.sum(mask) / mask.size) * 100
    
    @staticmethod
    def determine_blasting_intensity(rust_level, rust_percentage):
        """
        Determine the appropriate blasting intensity based on rust level and percentage
        
        Args:
            rust_level: Classification result ('SA2.5' or 'SA3.0')
            rust_percentage: Percentage of rust detected
            
        Returns:
            str: Recommended blasting intensity
            float: Recommended pressure (PSI)
            float: Recommended distance (cm)
        """
        if rust_level == 'SA3.0' or rust_percentage > 50:
            return "Heavy", 120.0, 20.0
        elif rust_level == 'SA2.5' or rust_percentage > 20:
            return "Medium", 90.0, 30.0
        else:
            return "Light", 60.0, 40.0
