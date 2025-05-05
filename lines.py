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
    # Update the LineDetectionApp class to include the LineProcessor
    # Update the LineDetectionApp class to include the LineProcessor

    def init_pipeline(self):
        """Initialize the pipeline components."""
        self.preprocessor = Preprocessor()
        self.edge_detector = EdgeDetector()
        self.line_detector = LineDetector()
        self.line_processor = LineProcessor()  # Add the new line processor
        self.junction_analyzer = JunctionAnalyzer()

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
            
            # Step 4: Line Processing (new step)
            processed_lines, processed_image = self.line_processor.process_lines(
                lines,
                image.copy(),
                collinearity_threshold=params['collinearity_threshold'],
                distance_threshold=params['distance_threshold'],
                min_angle_diff=params['min_angle_diff']
            )
            self.queue.put(('display', "Processed Lines", processed_image))
            
            # Step 5: Junction Analysis and Final Result
            junctions, graph, final_image = self.junction_analyzer.analyze_junctions(
                processed_lines,  # Use the processed lines instead of raw lines
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
                'processed_lines': processed_lines,
                'processed_image': processed_image,
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

    # Add new parameters to the setup_control_panel method
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
        
        # Add new parameter section for Line Processing
        self.create_parameter_section("Line Processing Parameters", [
            ("Collinearity Threshold", 5, 1, 20, 1),
            ("Distance Threshold", 20, 5, 100, 5),
            ("Min Angle Diff", 10, 1, 45, 1)
        ])
        
        self.create_parameter_section("Junction Parameters", [
            ("Junction Threshold", 10, 1, 50, 1)
        ])
        
        # Create a status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.left_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=5)

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
            "Edge Detection", "Hough Lines", "Processed Lines", "Final Result"
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
            'collinearity_threshold': int(self.collinearity_threshold.get()),
            'distance_threshold': int(self.distance_threshold.get()),
            'min_angle_diff': int(self.min_angle_diff.get()),
            'junction_threshold': int(self.junction_threshold.get())
        }
        
        # Start processing in a separate thread
        threading.Thread(
            target=self._process_thread, 
            args=(self.original_image.copy(), params),
            daemon=True
        ).start()

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
            cv2.imwrite(os.path.join(directory, f"{base_filename}_hough_lines.png"), self.result_images['line_image'])
            cv2.imwrite(os.path.join(directory, f"{base_filename}_processed_lines.png"), self.result_images['processed_image'])
            cv2.imwrite(os.path.join(directory, f"{base_filename}_final.png"), self.result_images['final_image'])
            
            # Save line data as CSV
            with open(os.path.join(directory, f"{base_filename}_original_lines.csv"), 'w') as f:
                f.write("x1,y1,x2,y2\n")
                for line in self.result_images['lines']:
                    x1, y1, x2, y2 = line
                    f.write(f"{x1},{y1},{x2},{y2}\n")
            
            # Save processed line data as CSV
            with open(os.path.join(directory, f"{base_filename}_processed_lines.csv"), 'w') as f:
                f.write("x1,y1,x2,y2\n")
                for line in self.result_images['processed_lines']:
                    x1, y1, x2, y2 = line
                    f.write(f"{x1},{y1},{x2},{y2}\n")
            
            # Save junction data as CSV
            with open(os.path.join(directory, f"{base_filename}_junctions.csv"), 'w') as f:
                f.write("x,y,degree\n")
                for j in self.result_images['junctions']:
                    x, y, degree = j
                    f.write(f"{x},{y},{degree}\n")
            
            self.status_var.set(f"Results saved to {directory}")
            
        except Exception as e:
            self.status_var.set(f"Error saving results: {str(e)}")

    def open_image(self):
        """Open an image file and display it."""
        # Get image file path from dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            # Set the image path
            self.current_image_path = file_path
            
            # Load the image
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("Could not read the image file.")
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            # Display the original image
            self.display_image("Original", image_rgb)
            
            # Reset result images
            self.result_images = {}
            
            # Update status
            self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")
    
    def display_image(self, tab_name, image):
        """Display an image in the specified tab.
        
        Args:
            tab_name: Name of the tab to display the image in
            image: Image to display (in RGB format)
        """
        if tab_name not in self.canvas:
            return
            
        # Convert image to PIL format
        pil_image = Image.fromarray(image)
        
        # Get canvas dimensions
        canvas = self.canvas[tab_name]
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # If canvas size is not yet set, use a default size or parent window size
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = max(800, self.root.winfo_width() - 300)  # Subtract left panel width
            canvas_height = max(600, self.root.winfo_height() - 50)  # Subtract some padding
        
        # Calculate scaling factor to fit image in canvas
        image_width, image_height = pil_image.size
        scale_width = canvas_width / image_width if image_width > 0 else 1
        scale_height = canvas_height / image_height if image_height > 0 else 1
        scale = min(scale_width, scale_height, 1.0)  # Don't scale up
        
        # Calculate new dimensions
        new_width = max(1, int(image_width * scale))
        new_height = max(1, int(image_height * scale))
        
        # Resize image
        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        canvas.delete("all")
        
        # Store reference to avoid garbage collection
        canvas.image = photo
        
        # Calculate position to center image
        x = max(0, (canvas_width - new_width) // 2)
        y = max(0, (canvas_height - new_height) // 2)
        
        # Display image
        canvas.create_image(x, y, anchor=tk.NW, image=photo)
        
        # Configure scroll region
        canvas.configure(scrollregion=canvas.bbox("all"))

    def process_queue(self):
        """Process messages from the queue to update the GUI."""
        try:
            while True:
                message = self.queue.get_nowait()
                
                if message[0] == 'display':
                    # Display an image in a tab
                    _, tab_name, image = message
                    self.display_image(tab_name, image)
                    
                elif message[0] == 'status':
                    # Update status message
                    _, status = message
                    self.status_var.set(status)
                    
                elif message[0] == 'results':
                    # Store results
                    _, results = message
                    self.result_images = results
                    
                elif message[0] == 'processing':
                    # Update processing flag
                    _, is_processing = message
                    self.processing = is_processing
                    
        except queue.Empty:
            # No more messages to process
            pass
        
        # Schedule next queue check if still processing
        if self.processing:
            self.root.after(100, self.process_queue)

class LineProcessor:
    """Class for processing and refining detected line segments."""
    
    def __init__(self):
        """Initialize the line processor."""
        self.debug_images = {}
    
    def process_lines(self, lines, original_image, collinearity_threshold=5, distance_threshold=20, min_angle_diff=10):
        """Process detected lines to merge collinear segments and handle islands.
        
        Args:
            lines: List of line segments as (x1, y1, x2, y2) tuples
            original_image: Original image for visualization
            collinearity_threshold: Maximum distance for a point to be considered collinear with a line
            distance_threshold: Maximum distance between endpoints to consider merging
            min_angle_diff: Minimum angle difference (in degrees) to consider lines as having different directions
            
        Returns:
            Tuple of (processed_lines, visualization_image) where processed_lines is a list of refined line segments
            and visualization_image shows the processing results
        """
        if not lines or len(lines) == 0:
            return [], original_image.copy()
        
        # Create a copy of the original image for visualization
        debug_image = original_image.copy()
        if len(debug_image.shape) == 2:
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
        
        # Draw original lines in light gray
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(debug_image, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        # Step 1: Group lines by similar orientation
        line_groups = self._group_by_orientation(lines, min_angle_diff)
        
        # Draw orientation groups with different colors
        orientation_image = debug_image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                  (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)]
        
        for i, group in enumerate(line_groups):
            color = colors[i % len(colors)]
            for line in group:
                x1, y1, x2, y2 = line
                cv2.line(orientation_image, (x1, y1), (x2, y2), color, 2)
        
        self.debug_images['orientation_groups'] = orientation_image
        
        # Step 2: For each orientation group, find collinear segments
        all_merged_lines = []
        for group in line_groups:
            merged_lines = self._merge_collinear_segments(
                group, 
                collinearity_threshold, 
                distance_threshold
            )
            all_merged_lines.extend(merged_lines)
        
        # Draw the final merged lines
        merged_image = original_image.copy()
        if len(merged_image.shape) == 2:
            merged_image = cv2.cvtColor(merged_image, cv2.COLOR_GRAY2BGR)
        
        for line in all_merged_lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(merged_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        self.debug_images['merged_lines'] = merged_image
        
        return all_merged_lines, merged_image
    
    def _group_by_orientation(self, lines, min_angle_diff):
        """Group lines by similar orientation.
        
        Args:
            lines: List of line segments
            min_angle_diff: Minimum angle difference (in degrees) to consider lines as having different directions
            
        Returns:
            List of line groups, where each group contains lines with similar orientation
        """
        if not lines:
            return []
        
        # Calculate angle for each line (in range [0, 180) degrees)
        line_angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            line_angles.append((line, angle))
        
        # Sort by angle
        line_angles.sort(key=lambda x: x[1])
        
        # Group lines with similar angles
        groups = []
        current_group = [line_angles[0][0]]
        current_angle = line_angles[0][1]
        
        for line, angle in line_angles[1:]:
            # Check if the angle difference is small enough (considering the 180-degree wrap-around)
            angle_diff = min(abs(angle - current_angle), 180 - abs(angle - current_angle))
            
            if angle_diff <= min_angle_diff:
                # Same orientation group
                current_group.append(line)
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [line]
                current_angle = angle
        
        # Add the last group if not empty
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _merge_collinear_segments(self, lines, collinearity_threshold, distance_threshold):
        """Merge collinear line segments within the same orientation group.
        
        Args:
            lines: List of line segments in the same orientation group
            collinearity_threshold: Maximum distance for a point to be considered collinear
            distance_threshold: Maximum distance between endpoints to consider merging
            
        Returns:
            List of merged line segments
        """
        if not lines or len(lines) <= 1:
            return lines
        
        # Convert lines to a more convenient representation for RANSAC
        line_data = []
        for line in lines:
            x1, y1, x2, y2 = line
            # Add both endpoints as data points
            line_data.append((x1, y1, line))
            line_data.append((x2, y2, line))
        
        # Sort line data by x-coordinate to help with sequential processing
        line_data.sort(key=lambda p: p[0])
        
        # Find connected components (islands)
        islands = self._find_connected_components(lines, distance_threshold)
        
        # For each island, fit a line using RANSAC
        merged_lines = []
        for island in islands:
            if len(island) == 1:
                # Single line, no need to merge
                merged_lines.append(island[0])
                continue
            
            # Extract points from all lines in this island
            island_points = []
            for line in island:
                x1, y1, x2, y2 = line
                island_points.append((x1, y1))
                island_points.append((x2, y2))
            
            # Find the best line representation using RANSAC
            best_line = self._ransac_fit_line(island_points, collinearity_threshold)
            if best_line is not None:
                merged_lines.append(best_line)
            else:
                # Fallback: use the longest line in the island
                longest_line = max(island, key=lambda line: 
                    (line[0] - line[2])**2 + (line[1] - line[3])**2)
                merged_lines.append(longest_line)
        
        return merged_lines
    
    def _find_connected_components(self, lines, distance_threshold):
        """Find connected components (islands) of lines.
        
        Args:
            lines: List of line segments
            distance_threshold: Maximum distance between endpoints to consider lines connected
            
        Returns:
            List of islands, where each island is a list of connected line segments
        """
        if not lines:
            return []
        
        # Create a graph where nodes are lines and edges indicate proximity
        graph = nx.Graph()
        
        # Add nodes for each line
        for i, line in enumerate(lines):
            graph.add_node(i, line=line)
        
        # Add edges between nearby lines
        for i, line1 in enumerate(lines):
            x1_1, y1_1, x2_1, y2_1 = line1
            
            for j in range(i+1, len(lines)):
                line2 = lines[j]
                x1_2, y1_2, x2_2, y2_2 = line2
                
                # Check distances between all endpoint combinations
                distances = [
                    np.sqrt((x1_1 - x1_2)**2 + (y1_1 - y1_2)**2),
                    np.sqrt((x1_1 - x2_2)**2 + (y1_1 - y2_2)**2),
                    np.sqrt((x2_1 - x1_2)**2 + (y2_1 - y1_2)**2),
                    np.sqrt((x2_1 - x2_2)**2 + (y2_1 - y2_2)**2)
                ]
                
                min_distance = min(distances)
                if min_distance <= distance_threshold:
                    graph.add_edge(i, j)
        
        # Find connected components (islands)
        components = list(nx.connected_components(graph))
        
        # Convert node indices back to lines
        islands = []
        for component in components:
            island = [lines[i] for i in component]
            islands.append(island)
        
        return islands
    
    def _ransac_fit_line(self, points, threshold):
        """Fit a line to points using RANSAC.
        
        Args:
            points: List of (x, y) points
            threshold: Maximum distance for a point to be considered an inlier
            
        Returns:
            Line segment as (x1, y1, x2, y2) or None if fitting fails
        """
        if len(points) < 2:
            return None
        
        # Convert points to numpy arrays
        points_array = np.array(points)
        X = points_array[:, 0].reshape(-1, 1)  # x-coordinates
        y = points_array[:, 1]  # y-coordinates
        
        try:
            # Try to fit a line using RANSAC
            ransac = RANSACRegressor(
                min_samples=2,
                residual_threshold=threshold,
                max_trials=100
            )
            ransac.fit(X, y)
            
            # Get inlier points
            inlier_mask = ransac.inlier_mask_
            inliers = points_array[inlier_mask]
            
            if len(inliers) < 2:
                return None
            
            # Find the extremities of the inlier points to define the line segment
            x_vals = inliers[:, 0]
            y_vals = inliers[:, 1]
            
            # Sort by x-coordinate
            sorted_indices = np.argsort(x_vals)
            sorted_x = x_vals[sorted_indices]
            sorted_y = y_vals[sorted_indices]
            
            # Get the extremity points
            x1, y1 = sorted_x[0], sorted_y[0]
            x2, y2 = sorted_x[-1], sorted_y[-1]
            
            # Special case for vertical lines
            if abs(x2 - x1) < 1e-6:
                # For vertical lines, sort by y-coordinate instead
                sorted_indices = np.argsort(y_vals)
                sorted_x = x_vals[sorted_indices]
                sorted_y = y_vals[sorted_indices]
                
                x1, y1 = sorted_x[0], sorted_y[0]
                x2, y2 = sorted_x[-1], sorted_y[-1]
            
            return (int(x1), int(y1), int(x2), int(y2))
            
        except Exception as e:
            print(f"RANSAC fitting failed: {str(e)}")
            return None
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
        
        # Draw detected lines
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Convert to standard format for further processing
        processed_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                processed_lines.append((x1, y1, x2, y2))
        
        return processed_lines, line_image


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
