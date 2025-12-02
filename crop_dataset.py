#!/usr/bin/env python3
"""
Recursively process all images in directory: crop transparent background
and save multiple regions as separate files
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage
import argparse
from datetime import datetime
import json

class TransparentImageSplitter:
    def __init__(self, threshold=1, padding=2, min_size=10):
        """
        Initialize the splitter
        
        Args:
            threshold: Alpha threshold (0-255) to consider as non-transparent
            padding: Padding around each region in pixels
            min_size: Minimum region size in pixels to process
        """
        self.threshold = threshold
        self.padding = padding
        self.min_size = min_size
        self.supported_formats = {'.png', '.webp', '.gif', '.tiff', '.tif', '.bmp', '.jpg', '.jpeg'}
        
    def process_directory(self, input_dir, output_base_dir="processed_images"):
        """
        Process all images in directory recursively
        
        Args:
            input_dir: Input directory path
            output_base_dir: Base output directory
        """
        input_path = Path(input_dir)
        output_base_path = Path(output_base_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory '{input_dir}' does not exist")
            return
        
        # Create output base directory
        output_base_path.mkdir(exist_ok=True)
        
        # Statistics
        stats = {
            'total_images': 0,
            'processed_images': 0,
            'total_regions': 0,
            'failed_images': [],
            'processing_time': None
        }
        
        start_time = datetime.now()
        
        # Walk through all directories
        for root, dirs, files in os.walk(input_dir):
            root_path = Path(root)
            relative_path = root_path.relative_to(input_path)
            
            for file in files:
                file_path = root_path / file
                
                # Check if file is an image
                if self._is_image_file(file_path):
                    stats['total_images'] += 1
                    print(f"\nProcessing: {file_path}")
                    
                    try:
                        regions_count = self.process_image(file_path, output_base_path, relative_path)
                        stats['processed_images'] += 1
                        stats['total_regions'] += regions_count
                        print(f"  ✓ Extracted {regions_count} regions")
                    except Exception as e:
                        error_msg = f"Failed to process {file_path}: {str(e)}"
                        print(f"  ✗ {error_msg}")
                        stats['failed_images'].append(str(file_path))
        
        # Calculate processing time
        stats['processing_time'] = str(datetime.now() - start_time)
        
        # Save statistics
        self._save_statistics(stats, output_base_path)
        
        return stats
    
    def _is_image_file(self, file_path):
        """Check if file is a supported image format"""
        suffix = file_path.suffix.lower()
        return suffix in self.supported_formats and file_path.is_file()
    
    def process_image(self, input_path, output_base_dir, relative_path):
        """
        Process a single image file
        
        Args:
            input_path: Path to input image
            output_base_dir: Base output directory
            relative_path: Relative path from input base directory
            
        Returns:
            Number of regions extracted
        """
        # Create output directory structure
        output_dir = output_base_dir / relative_path / input_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and process image
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)
        
        # Create mask from alpha channel
        mask = data[:, :, 3] > self.threshold
        
        # Find connected components
        labeled_array, num_features = ndimage.label(mask)
        
        # Process each region
        regions_count = 0
        for i in range(1, num_features + 1):
            rows, cols = np.where(labeled_array == i)
            
            if len(rows) < self.min_size:
                continue  # Skip small regions
            
            # Calculate bounding box
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Skip if region is too small
            region_width = max_col - min_col + 1
            region_height = max_row - min_row + 1
            if region_width < self.min_size or region_height < self.min_size:
                continue
            
            # Apply padding
            crop_box = (
                max(0, min_col - self.padding),
                max(0, min_row - self.padding),
                min(img.width, max_col + self.padding + 1),
                min(img.height, max_row + self.padding + 1)
            )
            
            # Crop the region
            cropped = img.crop(crop_box)
            
            # Save region
            output_filename = f"{input_path.stem}_region_{regions_count + 1:03d}.png"
            output_path = output_dir / output_filename
            cropped.save(output_path, "PNG", optimize=True)
            
            regions_count += 1
        
        return regions_count
    
    def _save_statistics(self, stats, output_dir):
        """Save processing statistics to JSON file"""
        stats_path = output_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Also print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images found: {stats['total_images']}")
        print(f"Successfully processed: {stats['processed_images']}")
        print(f"Total regions extracted: {stats['total_regions']}")
        print(f"Failed images: {len(stats['failed_images'])}")
        print(f"Processing time: {stats['processing_time']}")
        print(f"Statistics saved to: {stats_path}")
        
        if stats['failed_images']:
            print("\nFailed images:")
            for failed in stats['failed_images']:
                print(f"  - {failed}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively process images: crop transparent background and split regions"
    )
    parser.add_argument("input_dir", help="Input directory")
    parser.add_argument("-o", "--output", default="processed_images", 
                       help="Output directory (default: processed_images)")
    parser.add_argument("-t", "--threshold", type=int, default=1,
                       help="Alpha threshold 0-255 (default: 1)")
    parser.add_argument("-p", "--padding", type=int, default=2,
                       help="Padding around regions in pixels (default: 2)")
    parser.add_argument("-m", "--min-size", type=int, default=10,
                       help="Minimum region size in pixels (default: 10)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually processing")
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - No files will be processed")
        print(f"Input directory: {args.input_dir}")
        
        # Count files
        image_count = 0
        for root, dirs, files in os.walk(args.input_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in {'.png', '.webp', '.gif', '.tiff', '.tif', '.bmp', '.jpg', '.jpeg'}:
                    image_count += 1
        
        print(f"Found {image_count} image files to process")
        return
    
    # Create and run splitter
    splitter = TransparentImageSplitter(
        threshold=args.threshold,
        padding=args.padding,
        min_size=args.min_size
    )
    
    stats = splitter.process_directory(args.input_dir, args.output)
    
    if stats['total_images'] == 0:
        print(f"No image files found in {args.input_dir}")
        print(f"Supported formats: .png, .webp, .gif, .tiff, .tif, .bmp, .jpg, .jpeg")

if __name__ == "__main__":
    main()