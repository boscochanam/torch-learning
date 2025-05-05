import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, filters, morphology, exposure, transform
from scipy import ndimage
import argparse
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
import queue
import networkx as nx
from sklearn.linear_model import RANSACRegressor
import time
import os


class LineDetectionApp:
    """Main application class for the hand-drawn line detection system."""
    
    def __init__(self, root):
        """Initialize the application with a tkinter root window."""
        self.root = root
        self.root.title("Hand-Drawn Straight Line Detection System")
        self.root.geometry("1280x800")
        
        # Set up the main frames
        self.setup_frames()
        
        # Set up the control panel with parameters
        self.setup_control_panel()
        
        # Set up the image display area
        self.setup_display_area()
        
        # Initialize pipeline components
        self.init_pipeline()
        
        # Queue for thread-safe communication
        self.queue = queue.Queue()
        
        # Status variables
        self.processing = False
        self.current_image_path = None
        self.original_image = None
        self.result_images = {}
        
    def setup_frames(self):
        """Set up the main frames for the UI."""
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
        
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
    def setup_control_panel(self):
        """Set up the control panel with parameters."""
        # Create a LabelFrame for controls
        self.control_frame = ttk.LabelFrame(self.left_frame, text="Controls", padding=10)
        self.control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input/Output controls
        ttk.Button(self.control_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Process Image", command=self.process_image).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=5)
        
        # Parameter frames
        self.create_parameter_section("Pre-processing Parameters", [
            ("CLAHE Clip Limit", 2.0, 0.1, 10.0, 0.1),
            ("CLAHE Tile Size", 8, 2, 32, 1),
            ("Denoise Strength", 10, 0, 50, 1)
        ])
        
        self.create_parameter_section("Edge Detection Parameters", [
            ("Canny Low Threshold", 50, 0, 255, 1),
            ("Canny High Threshold", 150, 0, 255, 1),
            ("Canny Aperture", 3, 3, 7, 2, [3, 5, 7])
        ])
        
        self.create_parameter_section("Hough Transform Parameters", [
            ("Hough Threshold", 50, 10, 200, 1),
            ("Min Line Length", 30, 5, 200, 5),
            ("Max Line Gap", 10, 1, 100, 1)
        ])
        
        self.create_parameter_section("Junction Parameters", [
            ("Junction Threshold", 10, 1, 50, 1)
        ])
        
        # Create a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.left_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=5)
        
    def create_parameter_section(self, title, param_configs):
        """Create a section for a group of related parameters.
        
        Args:
            title: The title of the section
            param_configs: List of tuples (name, default, min, max, step, [optional values list])
        """
        section_frame = ttk.LabelFrame(self.control_frame, text=title, padding=5)
        section_frame.pack(fill=tk.X, pady=5)
        
        for config in param_configs:
            name = config[0]
            default = config[1]
            min_val = config[2]
            max_val = config[3]
            step = config[4]
            
            # Add variable to store parameter value
            var_name = name.lower().replace(" ", "_")
            setattr(self, var_name, tk.DoubleVar(value=default))
            
            # Create parameter frame
            param_frame = ttk.Frame(section_frame)
            param_frame.pack(fill=tk.X, pady=2)
            
            # Add label
            ttk.Label(param_frame, text=f"{name}:").pack(side=tk.LEFT)
            
            # Add scale or combobox based on config
            if len(config) > 5:  # Has discrete values
                values = config[5]
                combo = ttk.Combobox(param_frame, values=values, width=5)
                combo.set(default)
                combo.pack(side=tk.RIGHT)
                
                def on_combo_select(event, var_name=var_name):
                    getattr(self, var_name).set(float(event.widget.get()))
                
                combo.bind("<<ComboboxSelected>>", on_combo_select)
            else:  # Continuous scale
                scale = ttk.Scale(
                    param_frame, 
                    from_=min_val, 
                    to=max_val, 
                    variable=getattr(self, var_name),
                    orient=tk.HORIZONTAL
                )
                scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
                
                # Add value label
                value_label = ttk.Label(param_frame, width=5)
                value_label.pack(side=tk.RIGHT)
                
                # Update value label when scale changes
                def update_label(var_name=var_name, label=value_label):
                    label.config(text=f"{getattr(self, var_name).get():.1f}")
                
                getattr(self, var_name).trace_add("write", lambda *args: update_label())
                update_label()  # Initial update
        
    def setup_display_area(self):
        """Set up the area for displaying images."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for each pipeline stage
        self.tabs = {}
        self.canvas = {}
        
        tab_names = [
            "Original", "CLAHE Enhanced", "Denoised", 
            "Edge Detection", "Hough Lines", "Final Result"
        ]
        
        for name in tab_names:
            # Create frame for tab
            tab = ttk.Frame(self.notebook)
            self.tabs[name] = tab
            self.notebook.add(tab, text=name)
            
            # Create canvas for image display
            canvas = tk.Canvas(tab, bg="white")
            canvas.pack(fill=tk.BOTH, expand=True)
            self.canvas[name] = canvas
            
            # Add scrollbars
            h_scrollbar = ttk.Scrollbar(tab, orient=tk.HORIZONTAL, command=canvas.xview)
            v_scrollbar = ttk.Scrollbar(tab, orient=tk.VERTICAL, command=canvas.yview)
            canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
            
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def init_pipeline(self):
        """Initialize the pipeline components."""
        self.preprocessor = Preprocessor()
        self.edge_detector = EdgeDetector()
        self.line_detector = LineDetector()
        self.junction_analyzer = JunctionAnalyzer()
        
    def open_image(self):
        """Open an image file dialog and load the selected image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        
        if file_path:
            try:
                # Load the image
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not read the image file.")
                
                # Convert BGR to RGB for display
                image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                
                # Display the original image
                self.display_image("Original", image_rgb)
                
                # Update status
                self.current_image_path = file_path
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                
                # Clear previous results
                self.result_images = {}
                
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
    
    def process_image(self):
        """Process the loaded image through the pipeline."""
        if self.original_image is None:
            self.status_var.set("Please open an image first")
            return
        
        if self.processing:
            self.status_var.set("Processing already in progress")
            return
        
        # Set processing flag
        self.processing = True
        self.status_var.set("Processing image...")
        
        # Get parameters
        params = {
            'clahe_clip_limit': self.clahe_clip_limit.get(),
            'clahe_tile_size': int(self.clahe_tile_size.get()),
            'denoise_strength': int(self.denoise_strength.get()),
            'canny_low_threshold': int(self.canny_low_threshold.get()),
            'canny_high_threshold': int(self.canny_high_threshold.get()),
            'canny_aperture': int(self.canny_aperture.get()),
            'hough_threshold': int(self.hough_threshold.get()),
            'min_line_length': int(self.min_line_length.get()),
            'max_line_gap': int(self.max_line_gap.get()),
            'junction_threshold': int(self.junction_threshold.get())
        }
        
        # Start processing in a separate thread
        threading.Thread(
            target=self._process_thread, 
            args=(self.original_image.copy(), params),
            daemon=True
        ).start()
    
    def _process_thread(self, image, params):
        """Thread function for image processing."""
        try:
            # Step 1: Pre-processing
            clahe_result = self.preprocessor.apply_clahe(
                image, 
                clip_limit=params['clahe_clip_limit'], 
                tile_size=params['clahe_tile_size']
            )
            self.queue.put(('display', "CLAHE Enhanced", clahe_result))
            
            denoised_result = self.preprocessor.denoise_image(
                clahe_result, 
                strength=params['denoise_strength']
            )
            self.queue.put(('display', "Denoised", denoised_result))
            
            # Step 2: Edge Detection
            edge_result = self.edge_detector.detect_edges(
                denoised_result,
                low_threshold=params['canny_low_threshold'],
                high_threshold=params['canny_high_threshold'],
                aperture_size=params['canny_aperture']
            )
            self.queue.put(('display', "Edge Detection", cv2.cvtColor(edge_result, cv2.COLOR_GRAY2RGB)))
            
            # Step 3: Line Detection
            lines, line_image = self.line_detector.detect_lines(
                edge_result,
                image.copy(),
                threshold=params['hough_threshold'],
                min_line_length=params['min_line_length'],
                max_line_gap=params['max_line_gap']
            )
            self.queue.put(('display', "Hough Lines", line_image))
            
            # Step 4: Junction Analysis and Final Result
            junctions, graph, final_image = self.junction_analyzer.analyze_junctions(
                lines,
                image.copy(),
                threshold=params['junction_threshold']
            )
            self.queue.put(('display', "Final Result", final_image))
            
            # Store results for later use
            self.queue.put(('results', {
                'clahe': clahe_result,
                'denoised': denoised_result,
                'edges': edge_result,
                'lines': lines,
                'line_image': line_image,
                'junctions': junctions,
                'graph': graph,
                'final_image': final_image
            }))
            
            self.queue.put(('status', "Processing complete"))
            
        except Exception as e:
            self.queue.put(('status', f"Error during processing: {str(e)}"))
        
        finally:
            self.queue.put(('processing', False))
        
        # Process all queued GUI updates
        self.root.after(100, self.process_queue)
    
    def process_queue(self):
        """Process queued GUI updates."""
        try:
            while True:
                message = self.queue.get_nowait()
                
                cmd = message[0]
                if cmd == 'display':
                    _, tab_name, image = message
                    self.display_image(tab_name, image)
                elif cmd == 'status':
                    self.status_var.set(message[1])
                elif cmd == 'processing':
                    self.processing = message[1]
                elif cmd == 'results':
                    self.result_images = message[1]
                
                self.queue.task_done()
                
        except queue.Empty:
            pass
        
        # If still processing, check queue again soon
        if self.processing:
            self.root.after(100, self.process_queue)
    
    def display_image(self, tab_name, image):
        """Display an image in the specified tab.
        
        Args:
            tab_name: Name of the tab to display the image in
            image: OpenCV image (RGB)
        """
        # Convert to PIL Image
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image)
        else:  # Color
            pil_image = Image.fromarray(image)
        
        # Convert PIL Image to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        canvas = self.canvas[tab_name]
        canvas.delete("all")
        canvas.image = photo  # Keep a reference to prevent garbage collection
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        
        # Switch to the tab
        self.notebook.select(self.tabs[tab_name])
    
    def save_results(self):
        """Save all result images to files."""
        if not self.result_images:
            self.status_var.set("No results to save")
            return
        
        # Ask for directory
        directory = filedialog.askdirectory(title="Select Directory to Save Results")
        if not directory:
            return
        
        try:
            # Save each result image
            base_filename = os.path.splitext(os.path.basename(self.current_image_path))[0]
            
            # Save images
            cv2.imwrite(os.path.join(directory, f"{base_filename}_original.png"), self.original_image)
            cv2.imwrite(os.path.join(directory, f"{base_filename}_clahe.png"), self.result_images['clahe'])
            cv2.imwrite(os.path.join(directory, f"{base_filename}_denoised.png"), self.result_images['denoised'])
            cv2.imwrite(os.path.join(directory, f"{base_filename}_edges.png"), self.result_images['edges'])
            cv2.imwrite(os.path.join(directory, f"{base_filename}_lines.png"), self.result_images['line_image'])
            cv2.imwrite(os.path.join(directory, f"{base_filename}_final.png"), self.result_images['final_image'])
            
            # Save line data as CSV
            lines = self.result_images['lines']
            if lines is not None:
                with open(os.path.join(directory, f"{base_filename}_lines.csv"), 'w') as f:
                    f.write("x1,y1,x2,y2\n")
                    for line in lines:
                        x1, y1, x2, y2 = line
                        f.write(f"{x1},{y1},{x2},{y2}\n")
            
            # Save junction data as CSV
            junctions = self.result_images['junctions']
            if junctions:
                with open(os.path.join(directory, f"{base_filename}_junctions.csv"), 'w') as f:
                    f.write("x,y,degree\n")
                    for j in junctions:
                        x, y, degree = j
                        f.write(f"{x},{y},{degree}\n")
            
            self.status_var.set(f"Results saved to {directory}")
            
        except Exception as e:
            self.status_var.set(f"Error saving results: {str(e)}")


class Preprocessor:
    """Class for image preprocessing operations."""
    
    def apply_clahe(self, image, clip_limit=2.0, tile_size=8):
        """Apply Contrast Limited Adaptive Histogram Equalization.
        
        Args:
            image: Input BGR image
            clip_limit: Threshold for contrast limiting
            tile_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE enhanced image
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # Apply CLAHE
        enhanced = clahe.apply(gray)
        
        # Convert back to color if input was color
        if len(image.shape) == 3:
            enhanced_color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return enhanced_color
        
        return enhanced
    
    def denoise_image(self, image, strength=10):
        """Apply denoising to the image.
        
        Args:
            image: Input image
            strength: Denoising strength
            
        Returns:
            Denoised image
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply non-local means denoising
        denoised = cv2.fastNlMeansDenoising(
            gray, 
            None, 
            strength, 
            7, 
            21
        )
        
        # Convert back to color if input was color
        if len(image.shape) == 3:
            denoised_color = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            return denoised_color
        
        return denoised


class EdgeDetector:
    """Class for edge detection operations."""
    
    def detect_edges(self, image, low_threshold=50, high_threshold=150, aperture_size=3):
        """Detect edges using Canny edge detector.
        
        Args:
            image: Input image
            low_threshold: Lower threshold for Canny
            high_threshold: Higher threshold for Canny
            aperture_size: Aperture size for Sobel operator
            
        Returns:
            Binary edge image
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            gray, 
            low_threshold, 
            high_threshold, 
            apertureSize=aperture_size
        )
        
        return edges


class LineDetector:
    """Class for line detection operations."""
    
    def detect_lines(self, edge_image, original_image, threshold=50, min_line_length=30, max_line_gap=10):
        """Detect lines using Probabilistic Hough Transform.
        
        Args:
            edge_image: Binary edge image
            original_image: Original image for visualization
            threshold: Accumulator threshold in Hough transform
            min_line_length: Minimum line length
            max_line_gap: Maximum allowed gap between points on the same line
            
        Returns:
            Tuple of (lines, line_image) where lines is a numpy array of line segments
            and line_image is the original image with detected lines drawn on it
        """
        # Apply Probabilistic Hough Transform
        lines = cv2.HoughLinesP(
            edge_image, 
            rho=1, 
            theta=np.pi/180, 
            threshold=threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )
        
        # Create a copy of the original image for visualization
        line_image = original_image.copy()
        
        # Convert to RGB if necessary
        if len(line_image.shape) == 2:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
        
        # Convert to standard format for further processing
        processed_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                processed_lines.append((x1, y1, x2, y2))
        
        # Refine lines by merging collinear segments
        line_refiner = LineRefiner()
        refined_lines = line_refiner.refine_lines(processed_lines)
        
        # Draw the refined lines
        refined_line_image = original_image.copy()
        if len(refined_line_image.shape) == 2:
            refined_line_image = cv2.cvtColor(refined_line_image, cv2.COLOR_GRAY2BGR)
            
        for line in refined_lines:
            x1, y1, x2, y2 = line
            cv2.line(refined_line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return refined_lines, refined_line_image


class LineRefiner:
    """Class for refining and merging collinear line segments."""
    
    def refine_lines(self, lines, collinearity_threshold=5, distance_threshold=20):
        """Refine lines by merging collinear segments.
        
        Args:
            lines: List of line segments as (x1, y1, x2, y2) tuples
            collinearity_threshold: Maximum angle difference (in degrees) for lines to be considered collinear
            distance_threshold: Maximum perpendicular distance for a point to be considered on a line
            
        Returns:
            List of refined line segments after merging collinear segments
        """
        if not lines:
            return []
        
        # Convert angle to radians
        collinearity_threshold_rad = np.deg2rad(collinearity_threshold)
        
        # Group collinear line segments
        line_groups = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # Skip very short lines
            if np.sqrt((x2-x1)**2 + (y2-y1)**2) < 10:
                continue
                
            # Calculate line parameters (slope and intercept)
            if abs(x2 - x1) < 1e-6:  # Vertical line
                angle = np.pi/2
                distance = x1
            else:
                slope = (y2 - y1) / (x2 - x1)
                angle = np.arctan(slope)
                # Calculate perpendicular distance from origin to the line
                intercept = y1 - slope * x1
                distance = abs(intercept) / np.sqrt(1 + slope**2)
                if intercept < 0:
                    distance = -distance
            
            # Check if the line belongs to an existing group
            found_group = False
            for group in line_groups:
                group_angle, group_distance = group['params']
                
                # Check if angles are similar (handling the case of angles close to pi/2)
                angle_diff = abs(angle - group_angle)
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                # Check if lines are on the same infinite line
                if (angle_diff < collinearity_threshold_rad and 
                    abs(distance - group_distance) < distance_threshold):
                    group['lines'].append(line)
                    
                    # Update group's angle and distance as weighted average
                    num_lines = len(group['lines'])
                    group['params'] = (
                        (group_angle * (num_lines - 1) + angle) / num_lines,
                        (group_distance * (num_lines - 1) + distance) / num_lines
                    )
                    found_group = True
                    break
            
            # If no matching group, create a new one
            if not found_group:
                line_groups.append({
                    'params': (angle, distance),
                    'lines': [line]
                })
        
        # Merge line segments within each group
        refined_lines = []
        
        for group in line_groups:
            group_lines = group['lines']
            
            # Skip singleton groups (no merging needed)
            if len(group_lines) == 1:
                refined_lines.append(group_lines[0])
                continue
            
            # Project all endpoints onto the line
            angle, distance = group['params']
            
            # For vertical lines
            if abs(abs(angle) - np.pi/2) < 1e-6:
                # Get all endpoints
                points = []
                for line in group_lines:
                    x1, y1, x2, y2 = line
                    points.extend([(x1, y1), (x2, y2)])
                
                # Find the extreme y-coordinates
                points.sort(key=lambda p: p[1])
                x_val = distance  # For vertical lines, distance is the x-coordinate
                y_min = points[0][1]
                y_max = points[-1][1]
                
                refined_lines.append((int(x_val), int(y_min), int(x_val), int(y_max)))
            else:
                # Regular lines
                slope = np.tan(angle)
                intercept = distance * np.sqrt(1 + slope**2) * np.sign(np.cos(angle))
                
                # Project all endpoints onto the line
                projected_points = []
                for line in group_lines:
                    x1, y1, x2, y2 = line
                    # For each endpoint, find its projection on the line
                    for x, y in [(x1, y1), (x2, y2)]:
                        # Find the point on the line that is closest to (x,y)
                        if abs(slope) < 1e-6:  # Nearly horizontal line
                            proj_x = x
                            proj_y = intercept
                        else:
                            perp_slope = -1/slope
                            perp_intercept = y - perp_slope * x
                            # Intersection of the line and its perpendicular through (x,y)
                            proj_x = (perp_intercept - intercept) / (slope - perp_slope)
                            proj_y = slope * proj_x + intercept
                        
                        projected_points.append((proj_x, proj_y))
                
                # Find the extreme projections along the line
                if abs(slope) < 1:  # More horizontal
                    projected_points.sort(key=lambda p: p[0])
                    x_min, y_min = projected_points[0]
                    x_max, y_max = projected_points[-1]
                else:  # More vertical
                    projected_points.sort(key=lambda p: p[1])
                    x_min, y_min = projected_points[0]
                    x_max, y_max = projected_points[-1]
                
                refined_lines.append((int(x_min), int(y_min), int(x_max), int(y_max)))
        
        return refined_lines


class JunctionAnalyzer:
    """Class for analyzing junctions between detected lines."""
    
    def analyze_junctions(self, lines, original_image, threshold=10):
        """Analyze junctions between detected lines.
        
        Args:
            lines: List of line segments as (x1, y1, x2, y2) tuples
            original_image: Original image for visualization
            threshold: Distance threshold for junction merging
            
        Returns:
            Tuple of (junctions, graph, final_image) where junctions is a list of junction points,
            graph is a NetworkX graph representing connections, and final_image is the visualization
        """
        if not lines:
            return [], nx.Graph(), original_image.copy()
        
        # Create a copy of the original image for visualization
        final_image = original_image.copy()
        
        # Convert to RGB if necessary
        if len(final_image.shape) == 2:
            final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
        
        # Extract endpoints from all lines
        endpoints = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            endpoints.append((x1, y1, i, 'start'))
            endpoints.append((x2, y2, i, 'end'))
        
        # Find junctions by clustering endpoints
        junctions = []
        used_endpoints = set()
        
        for i, (x1, y1, line_idx1, pos1) in enumerate(endpoints):
            if i in used_endpoints:
                continue
            
            junction_points = [(x1, y1, line_idx1, pos1)]
            used_endpoints.add(i)
            
            for j, (x2, y2, line_idx2, pos2) in enumerate(endpoints):
                if j in used_endpoints or i == j:
                    continue
                
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if distance <= threshold:
                    junction_points.append((x2, y2, line_idx2, pos2))
                    used_endpoints.add(j)
            
            # If more than one endpoint in this junction
            if len(junction_points) > 1:
                # Calculate average position as the junction point
                jx = int(sum(p[0] for p in junction_points) / len(junction_points))
                jy = int(sum(p[1] for p in junction_points) / len(junction_points))
                
                # Store junction with its degree (number of connected lines)
                connected_lines = set(p[2] for p in junction_points)
                junctions.append((jx, jy, len(connected_lines)))
        
        # Add individual endpoints that aren't part of a junction
        for i, (x, y, line_idx, pos) in enumerate(endpoints):
            if i not in used_endpoints:
                junctions.append((x, y, 1))
        
        # Build graph representation
        graph = nx.Graph()
        
        # Add junctions as nodes
        for i, (x, y, degree) in enumerate(junctions):
            graph.add_node(i, pos=(x, y), degree=degree)
        
        # Add lines as edges
        edge_list = []
        for line_idx, line in enumerate(lines):
            x1, y1, x2, y2 = line
            
            # Find closest junctions to the endpoints
            closest_to_start = None
            closest_to_end = None
            min_dist_start = float('inf')
            min_dist_end = float('inf')
            
            for junction_idx, (jx, jy, _) in enumerate(junctions):
                dist_start = np.sqrt((jx - x1) ** 2 + (jy - y1) ** 2)
                dist_end = np.sqrt((jx - x2) ** 2 + (jy - y2) ** 2)
                
                if dist_start < min_dist_start:
                    min_dist_start = dist_start
                    closest_to_start = junction_idx
                
                if dist_end < min_dist_end:
                    min_dist_end = dist_end
                    closest_to_end = junction_idx
            
            # Add edge if both endpoints are close to junctions
            if closest_to_start is not None and closest_to_end is not None and closest_to_start != closest_to_end:
                graph.add_edge(closest_to_start, closest_to_end, line_idx=line_idx)
                edge_list.append((closest_to_start, closest_to_end))
        
        # Draw final visualization with junctions and connections
        # Draw lines
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw junctions
        for junction_idx, (x, y, degree) in enumerate(junctions):
            # Use different colors and sizes based on junction degree
            if degree > 2:  # Major junction
                color = (0, 255, 0)  # Green
                radius = 8
            elif degree == 2:  # Simple connection
                color = (255, 0, 0)  # Blue
                radius = 6
            else:  # Endpoint
                color = (255, 255, 0)  # Yellow
                radius = 4
            
            cv2.circle(final_image, (x, y), radius, color, -1)
            cv2.putText(final_image, f"{junction_idx}", (x+5, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return junctions, graph, final_image


def main():
    """Main function to run the application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hand-Drawn Straight Line Detection System')
    parser.add_argument('--image', type=str, help='Path to input image')
    args = parser.parse_args()
    
    # Create the main application window
    root = tk.Tk()
    app = LineDetectionApp(root)
    
    # If image is specified, open it
    if args.image:
        # Schedule image loading after the GUI is up
        root.after(100, lambda: open_image_delayed(app, args.image))
    
    # Run the application
    root.mainloop()


def open_image_delayed(app, image_path):
    """Open an image after a delay to ensure GUI is loaded."""
    try:
        # Set the image path
        app.current_image_path = image_path
        
        # Load the image
        app.original_image = cv2.imread(image_path)
        if app.original_image is None:
            raise ValueError("Could not read the image file.")
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(app.original_image, cv2.COLOR_BGR2RGB)
        
        # Display the original image
        app.display_image("Original", image_rgb)
        
        # Update status
        app.status_var.set(f"Loaded image: {os.path.basename(image_path)}")
        
    except Exception as e:
        app.status_var.set(f"Error loading image: {str(e)}")


if __name__ == "__main__":
    main()
