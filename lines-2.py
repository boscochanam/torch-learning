import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, Scale, IntVar
from PIL import Image, ImageTk
import sys # To handle potential division by zero for vertical lines

# --- Configuration ---
MAX_DISPLAY_WIDTH = 400  # Reduced from 500
MAX_DISPLAY_HEIGHT = 300 # Reduced from 400
LINE_COLOR = (0, 255, 0)

# Global variable to store current image path
current_image_path = None

def process_and_update():
    """Process current image with updated parameters and refresh display"""
    global photo_orig, photo_gray, photo_thresh, photo_final
    
    if current_image_path is None:
        return
        
    # Process the image with current parameter values
    original_img, gray_img, thresh_img, output_img = detect_lines(current_image_path)
    
    # Update display
    photo_orig = cv_to_photoimage(original_img)
    photo_gray = cv_to_photoimage(gray_img)
    photo_thresh = cv_to_photoimage(thresh_img)
    photo_final = cv_to_photoimage(output_img)
    
    # Update GUI labels
    label_orig_img.config(image=photo_orig)
    label_gray_img.config(image=photo_gray)
    label_thresh_img.config(image=photo_thresh)
    label_final_img.config(image=photo_final)

def on_slider_change(*args):
    """Callback for slider value changes"""
    process_and_update()

# --- Core Image Processing Logic ---
def detect_lines(image_path):
    """
    Loads an image, detects linear lines, and returns intermediate and final images.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        tuple: (original_img, gray_img, thresh_img, output_img)
               Returns None for all if loading fails.
               Intermediate images might be None if processing fails at a step.
    """
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Error: Could not load image from {image_path}")
            return None, None, None, None
        
        output_img = original_img.copy() # Image to draw results on
        rows, cols = original_img.shape[:2]

        # 1. Convert to Grayscale
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

        # Optional: Apply Gaussian Blur to reduce noise before thresholding
        # blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0) # Kernel size (5,5) might need adjustment

        # 2. Apply Binary Inverse Thresholding
        # Lines become white (255), background becomes black (0)
        THRESHOLD_VALUE = threshold_var.get()
        ret, thresh_inv = cv2.threshold(gray_img, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
        
        # Optional: Apply Morphological Closing to fill gaps in lines and remove small black holes
        kernel_size = morph_kernel_var.get()
        MORPH_KERNEL_SIZE = (kernel_size, kernel_size)
        MORPH_ITERATIONS = morph_iter_var.get()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
        morphed_thresh = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
        # You could also use MORPH_OPEN to remove small white noise specks if needed:
        # morphed_thresh = cv2.morphologyEx(thresh_inv, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
        
        processed_thresh_img = morphed_thresh # Use the morphed image for component analysis


        # 3. Connected Components Labeling
        # Find contours works well here too, but connected components gives stats easily
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_thresh_img, connectivity=8, ltype=cv2.CV_32S)

        print(f"Found {num_labels - 1} potential line components (before filtering).")

        # 4. Process Each Component (Blob)
        MIN_LINE_AREA = min_area_var.get()
        for i in range(1, num_labels): # Skip label 0 (background)
            area = stats[i, cv2.CC_STAT_AREA]
            x_stat, y_stat, w_stat, h_stat = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            # Filter out small blobs (noise)
            if area < MIN_LINE_AREA:
                continue

            # Extract pixel coordinates for the current blob
            # points format needs to be (N, 1, 2) or (N, 2) of type float32 for fitLine
            # np.where returns (row, col), need (x, y) which is (col, row)
            points = np.column_stack(np.where(labels == i)[::-1]).astype(np.float32)

            if len(points) < 5: # fitLine needs at least 5 points
                 print(f"Skipping component {i}: Too few points ({len(points)}) for robust line fitting.")
                 continue

            # 5. Fit a Straight Line using cv2.fitLine()
            # Returns a normalized vector (vx, vy) and a point on the line (x0, y0)
            try:
                [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            except cv2.error as e:
                print(f"Error fitting line for component {i}: {e}")
                continue

            # 6. Determine Endpoints for Drawing within component bounds
            # Get the bounding box coordinates
            x_min, x_max = x_stat, x_stat + w_stat
            y_min, y_max = y_stat, y_stat + h_stat

            # Handle near-vertical and near-horizontal lines differently
            if abs(vx) < 0.01:  # Near vertical line
                pt1 = (int(x0), y_min)
                pt2 = (int(x0), y_max)
            elif abs(vy) < 0.01:  # Near horizontal line
                pt1 = (x_min, int(y0))
                pt2 = (x_max, int(y0))
            else:
                # Calculate intersection points with bounding box
                # Line equation: y = (vy/vx)(x - x0) + y0
                slope = vy[0] / vx[0]
                
                # Intersections with vertical boundaries
                y_at_xmin = slope * (x_min - x0[0]) + y0[0]
                y_at_xmax = slope * (x_max - x0[0]) + y0[0]
                
                # Intersections with horizontal boundaries
                x_at_ymin = (y_min - y0[0]) / slope + x0[0]
                x_at_ymax = (y_max - y0[0]) / slope + x0[0]
                
                # Collect all valid intersection points
                points = []
                if y_min <= y_at_xmin <= y_max:
                    points.append((x_min, y_at_xmin))
                if y_min <= y_at_xmax <= y_max:
                    points.append((x_max, y_at_xmax))
                if x_min <= x_at_ymin <= x_max:
                    points.append((x_at_ymin, y_min))
                if x_min <= x_at_ymax <= x_max:
                    points.append((x_at_ymax, y_max))
                
                # Use the first and last intersection points
                if len(points) >= 2:
                    pt1 = (int(points[0][0]), int(points[0][1]))
                    pt2 = (int(points[-1][0]), int(points[-1][1]))
                else:
                    continue  # Skip if we can't find valid intersection points

            # 7. Visualize the Fitted Line
            LINE_THICKNESS = line_thickness_var.get()
            cv2.line(output_img, pt1, pt2, (0, 255, 0), LINE_THICKNESS, cv2.LINE_AA)
            # Optional: Draw bounding box or centroid for debugging
            # cv2.rectangle(output_img, (x_stat, y_stat), (x_stat + w_stat, y_stat + h_stat), (255, 0, 0), 1)
            # cv2.circle(output_img, (int(centroids[i][0]), int(centroids[i][1])), 3, (0, 0, 255), -1)


        # Prepare threshold image for display (convert back to BGR)
        thresh_display = cv2.cvtColor(processed_thresh_img, cv2.COLOR_GRAY2BGR)

        return original_img, gray_img, thresh_display, output_img

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        # Return None for images that couldn't be generated
        return (original_img if 'original_img' in locals() else None,
                gray_img if 'gray_img' in locals() else None,
                None, # Threshold might have failed
                None) # Output might have failed


# --- Tkinter GUI ---

# Global variables to hold PhotoImage objects to prevent garbage collection
photo_orig = None
photo_gray = None
photo_thresh = None
photo_final = None

def resize_for_display(image):
    """Resizes an OpenCV image to fit within display limits."""
    if image is None:
        return None
    h, w = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / w, MAX_DISPLAY_HEIGHT / h, 1.0) # Don't scale up
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def cv_to_photoimage(cv_image):
    """Converts an OpenCV image (BGR or Grayscale) to a Tkinter PhotoImage."""
    if cv_image is None:
        # Return a blank placeholder image if input is None
        img = Image.new('RGB', (MAX_DISPLAY_WIDTH // 2, MAX_DISPLAY_HEIGHT // 2), color = 'lightgrey')
        return ImageTk.PhotoImage(img)
        
    # Resize before conversion
    cv_image_resized = resize_for_display(cv_image)
    
    if len(cv_image_resized.shape) == 2: # Grayscale
        img = Image.fromarray(cv_image_resized)
    else: # BGR
        img = Image.fromarray(cv2.cvtColor(cv_image_resized, cv2.COLOR_BGR2RGB))
    
    return ImageTk.PhotoImage(image=img)

def open_file_and_process():
    """Opens a file dialog, loads image, processes it, and updates GUI."""
    global current_image_path
    
    filepath = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All Files", "*.*")]
    )
    if not filepath:
        return # User cancelled

    current_image_path = filepath
    process_and_update()

# --- Initialize Main Window ---
def create_gui():
    global root, threshold_var, min_area_var, morph_kernel_var, morph_iter_var, line_thickness_var
    global label_orig_img, label_gray_img, label_thresh_img, label_final_img, photo_orig, photo_gray, photo_thresh, photo_final
    
    root = tk.Tk()
    root.title("Hand-Drawn Line Detector")
    
    # Set minimum window size
    root.minsize(850, 700)
    
    # Initialize Tkinter variables
    threshold_var = IntVar(value=180)
    min_area_var = IntVar(value=50)
    morph_kernel_var = IntVar(value=5)
    morph_iter_var = IntVar(value=2)
    line_thickness_var = IntVar(value=2)
    
    # Create frames
    control_frame = Frame(root, padx=10, pady=10)
    control_frame.pack(side=tk.TOP, fill=tk.X)
    
    btn_load = Button(control_frame, text="Load Image", command=open_file_and_process, width=20)
    btn_load.pack()
    
    # Slider frame
    slider_frame = Frame(root, padx=10, pady=5)
    slider_frame.pack(side=tk.TOP, fill=tk.X)
    
    # Create sliders
    Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL, 
          label="Threshold", variable=threshold_var, command=on_slider_change).pack(fill=tk.X)
    Scale(slider_frame, from_=10, to=200, orient=tk.HORIZONTAL, 
          label="Min Area", variable=min_area_var, command=on_slider_change).pack(fill=tk.X)
    Scale(slider_frame, from_=3, to=11, orient=tk.HORIZONTAL, 
          label="Morph Kernel Size", variable=morph_kernel_var, command=on_slider_change).pack(fill=tk.X)
    Scale(slider_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
          label="Morph Iterations", variable=morph_iter_var, command=on_slider_change).pack(fill=tk.X)
    Scale(slider_frame, from_=1, to=5, orient=tk.HORIZONTAL, 
          label="Line Thickness", variable=line_thickness_var, command=on_slider_change).pack(fill=tk.X)
    
    # Image frame with better grid configuration
    image_frame = Frame(root, padx=10, pady=10)
    image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    # Configure grid rows and columns to expand properly
    image_frame.grid_columnconfigure(0, weight=1, uniform="group1")
    image_frame.grid_columnconfigure(1, weight=1, uniform="group1")
    image_frame.grid_rowconfigure(1, weight=1, uniform="group2")
    image_frame.grid_rowconfigure(3, weight=1, uniform="group2")
    
    # Create placeholder images initially
    placeholder = cv_to_photoimage(None)
    
    # Update image displays with consistent frame sizing
    Label(image_frame, text="Original Image").grid(row=0, column=0, pady=(0, 5))
    label_orig_img = Label(image_frame, image=placeholder, borderwidth=1, relief="solid")
    label_orig_img.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
    label_orig_img.image = placeholder
    
    Label(image_frame, text="Grayscale").grid(row=0, column=1, pady=(0, 5))
    label_gray_img = Label(image_frame, image=placeholder, borderwidth=1, relief="solid")
    label_gray_img.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
    label_gray_img.image = placeholder
    
    Label(image_frame, text="Thresholded & Morphed").grid(row=2, column=0, pady=(10, 5))
    label_thresh_img = Label(image_frame, image=placeholder, borderwidth=1, relief="solid")
    label_thresh_img.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")
    label_thresh_img.image = placeholder
    
    Label(image_frame, text="Detected Lines").grid(row=2, column=1, pady=(10, 5))
    label_final_img = Label(image_frame, image=placeholder, borderwidth=1, relief="solid")
    label_final_img.grid(row=3, column=1, padx=5, pady=5, sticky="nsew")
    label_final_img.image = placeholder

# Main execution
if __name__ == "__main__":
    create_gui()
    root.mainloop()