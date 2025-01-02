import cv2
import numpy as np

class RoadCrackSegmenter:
    def __init__(self, image_path):
        # Read the input image
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Create a window for trackbars and display
        cv2.namedWindow('Crack Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Crack Segmentation', 1200, 800)
        
        # Create trackbars
        cv2.createTrackbar('Blur Kernel', 'Crack Segmentation', 5, 15, self.update)
        cv2.createTrackbar('Custom Thresh', 'Crack Segmentation', 127, 255, self.update)
        cv2.createTrackbar('Morph Kernel', 'Crack Segmentation', 3, 7, self.update)
        
        # Initial update
        self.update(0)
    
    def segment_road_cracks(self, 
        blur_kernel_size, 
        custom_thresh, 
        morph_kernel_size
    ):
        """
        Segment road cracks with simplified parameters
        """
        # Ensure blur kernel is odd
        blur_kernel = (blur_kernel_size * 2 + 1, blur_kernel_size * 2 + 1)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
        
        # Apply binary thresholding
        _, thresh = cv2.threshold(
            blurred, 
            custom_thresh, 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Create morphological kernel
        kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
        
        # Morphological opening
        opening = cv2.morphologyEx(
            thresh, 
            cv2.MORPH_OPEN, 
            kernel, 
            iterations=1
        )
        
        # Find sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Find sure foreground area using distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
            dist_transform, 
            0.5 * dist_transform.max(), 
            255, 
            0
        )
        sure_fg = sure_fg.astype(np.uint8)
        
        # Find unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark unknown region with zero
        markers[unknown == 255] = 0
        
        # Copy of original image for result
        result = self.original_img.copy()
        
        # Apply watershed
        markers_copy = markers.copy()
        markers_watershed = cv2.watershed(result, markers_copy)
        
        # Highlight crack boundaries
        result[markers_watershed == -1] = [0, 255, 0]  # Green for cracks
        
        # Create composite image for display
        composite = np.hstack([
            cv2.resize(self.original_img, (600, 400)),
            cv2.resize(result, (600, 400))
        ])
        
        return composite
    
    def update(self, _):
        """
        Update segmentation based on trackbar values
        """
        # Get current trackbar positions
        blur_kernel_size = cv2.getTrackbarPos('Blur Kernel', 'Crack Segmentation')
        custom_thresh = cv2.getTrackbarPos('Custom Thresh', 'Crack Segmentation')
        morph_kernel_size = cv2.getTrackbarPos('Morph Kernel', 'Crack Segmentation')
        
        # Perform segmentation
        result = self.segment_road_cracks(
            blur_kernel_size,
            custom_thresh,
            morph_kernel_size
        )
        
        # Display result
        cv2.imshow('Crack Segmentation', result)
    
    def run(self):
        """
        Run the interactive segmentation tool
        """
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    # Path to your road crack image
    image_path = 'images\CFD_001.jpg'
    
    # Create and run segmenter
    segmenter = RoadCrackSegmenter(image_path)
    segmenter.run()

if __name__ == '__main__':
    main()