import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np

# Initialize YOLO model
model = YOLO('small.pt')

def show_preds_image(image):
    """
    Process image and show predictions with segmentation masks (if available)
    """
    if image is None:
        return None
        
    # Convert to numpy array if needed
    if isinstance(image, str):
        image = cv2.imread(image)
    
    # Make predictions
    outputs = model.predict(source=image)
    results = outputs[0]  # Extract results from the prediction output
    
    # Initialize image copy
    image_copy = image.copy()
    
    # Check if segmentation masks are available
    if hasattr(results, "masks") and results.masks is not None:
        # Extract the mask data (xy contains the pixel coordinates of the segments)
        masks = results.masks.xy  # List of segments as numpy arrays (pixel coordinates)
        
        for mask in masks:
            # Render the mask as a filled polygon on a black image
            mask_image = np.zeros_like(image_copy)
            cv2.fillPoly(mask_image, [np.int32(mask)], (0, 255, 0))  # Green color for the mask
            image_copy = cv2.addWeighted(image_copy, 1, mask_image, 0.5, 0)  # Overlay mask
    
    # Draw bounding boxes if present
    for i, det in enumerate(results.boxes.xyxy):
        cv2.rectangle(
            image_copy,
            (int(det[0]), int(det[1])),
            (int(det[2]), int(det[3])),
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
    
    # Convert image to RGB format for display in Gradio interface
    return cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)





def show_preds_video(video):
    """
    Process video and show predictions with segmentation masks (if available)
    """
    if video is None:
        return None
        
    cap = cv2.VideoCapture(video)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_copy = frame.copy()
        outputs = model.predict(source=frame)
        results = outputs[0]
        
        # Check if segmentation masks are available
        if hasattr(results, "masks") and results.masks is not None:
            # Extract the mask data (xy contains the pixel coordinates of the segments)
            masks = results.masks.xy  # List of segments as numpy arrays (pixel coordinates)
            
            for mask in masks:
                # Render the mask as a filled polygon on a black image
                mask_image = np.zeros_like(frame_copy)
                cv2.fillPoly(mask_image, [np.int32(mask)], (0, 255, 0))  # Green color for the mask
                frame_copy = cv2.addWeighted(frame_copy, 1, mask_image, 0.5, 0)  # Overlay mask
        
        # Draw bounding boxes if present
        for i, det in enumerate(results.boxes.xyxy):
            cv2.rectangle(
                frame_copy,
                (int(det[0]), int(det[1])),
                (int(det[2]), int(det[3])),
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        
        # Yield the processed frame
        yield cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    
    cap.release()



# Create image interface
interface_image = gr.Interface(
    fn=show_preds_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Image")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detected Potholes")
    ],
    title="CRACK & POTHOLE Detector- Image",
    description="Upload an image to detect cracks and potholes",
    cache_examples=False
)

# Create video interface
interface_video = gr.Interface(
    fn=show_preds_video,
    inputs=[
        gr.Video(label="Upload Video")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detected Potholes")
    ],
    title="CRACK & POTHOLE Detector - Video",
    description="Upload a video to detect cracks and potholes",
    cache_examples=False
)

# Create tabbed interface
demo = gr.TabbedInterface(
    [interface_image, interface_video],
    tab_names=['Image Detection', 'Video Detection']
).queue()

if __name__ == "__main__":
    demo.launch()