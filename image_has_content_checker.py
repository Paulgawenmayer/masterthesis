#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 13:05:00 2025

@author: paulmayer


    Checks if an image appears visually empty (white/bright). Important as LGL Server provides 10 layers per decade,
    of which most will return only a blank tile for a given coordinate. This script ensures that only tiles with content
    will be downloaded. 
    
    Args:
        image_path (str): Path to the image file
        brightness_threshold (int): Minimum pixel brightness for white (0-255)
        stddev_threshold (float): Maximum allowed brightness variation
        white_ratio_threshold (float): Minimum ratio of bright pixels
    
    Returns:
        bool: True if image appears empty, False otherwise
"""
from PIL import Image
import numpy as np
import os

def is_image_blank(image_path, brightness_threshold=245, stddev_threshold=5, white_ratio_threshold=0.98):

    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)

        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)
        white_pixel_ratio = np.mean(img_array >= brightness_threshold)

        #for more detailed console-output
        #print(f"→ Mean brightness: {mean_brightness:.2f}")
        #print(f"→ Std. deviation:  {std_brightness:.2f}")
        #print(f"→ White ratio:     {white_pixel_ratio*100:.2f}%")

        # Criteria: high brightness & low contrast & many bright pixels
        if mean_brightness >= brightness_threshold and \
           std_brightness <= stddev_threshold and \
           white_pixel_ratio >= white_ratio_threshold:
            return True
        else:
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip()

    if not os.path.exists(image_path):
        print("Image not found!")
    elif is_image_blank(image_path):
        print("Image is empty")
    else:
        print("Image is not empty")