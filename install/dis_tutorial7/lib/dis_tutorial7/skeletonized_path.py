#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, label, find_objects
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
import cv2

class SkeletonizedPath(Node):
    def __init__(self):
        super().__init__('skeletonized_path')

        # Declare parameters
        self.declare_parameter('map_image_path', '/home/beta/RINS-TASK2/dis_tutorial3/maps/map.pgm')
        self.declare_parameter('path_topic', '/global_path')
        self.declare_parameter('dilation_pixels', 7)
        self.declare_parameter('resolution', 0.05)  # Map resolution in meters/pixel
        self.declare_parameter('map_origin_x', -8.43)  # Example value, adjust based on your map
        self.declare_parameter('map_origin_y', -8.09)  # Example value, adjust based on your map

        # Get parameters
        self.map_image_path = self.get_parameter('map_image_path').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.dilation_pixels = self.get_parameter('dilation_pixels').get_parameter_value().integer_value
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value

        # Publisher
        self.path_pub = self.create_publisher(
            Path,
            self.path_topic,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        
        # Process the map and publish the path
        self.process_map_and_publish_path()

    def process_map_and_publish_path(self):
        """Process the map image, generate skeletonized path, and publish it."""
        try:
            # Read and preprocess the map
            self.get_logger().info(f"Reading map from {self.map_image_path}")
            map_image = self.read_pgm(self.map_image_path)
            
            # Generate skeletonized path
            self.get_logger().info("Generating skeletonized path...")
            path_points = self.generate_skeleton_path(map_image)
            # 
            # Publish the path
            self.publish_path(path_points)
            self.get_logger().info(f"Published global path with {len(path_points)} waypoints to {self.path_topic}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing map: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def read_pgm(self, filename):
        """Read a PGM file."""
        with open(filename, 'rb') as f:
            line = f.readline().decode('ascii').strip()
            if line != 'P5':
                raise ValueError("Not a PGM image (P5 format)")
            
            # Skip comments
            while True:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    break
            
            width, height = map(int, line.split())
            maxval = int(f.readline().decode('ascii').strip())
            
            # Read image data
            image = np.frombuffer(f.read(), dtype=np.uint8 if maxval < 256 else np.uint16)
            image = image.reshape((height, width))
            
        return image


    def clean_image(self, image):
        """Clean the image by removing small objects and dilating obstacles."""
        # Thresholding to create a binary image
        binary_map = (image > 245).astype(np.uint8)
        
        # Label connected components
        labeled_map, num_features = label(binary_map)
        
        # Find objects and remove small ones
        objects = find_objects(labeled_map)
        for obj in objects:
            if obj is not None:
                if obj[0].stop - obj[0].start < 10 or obj[1].stop - obj[1].start < 10:
                    labeled_map[obj] = 0
        
        # Create a binary image of obstacles
        cleaned_image = labeled_map > 0
        
        # Dilate the obstacles to make them appear larger using an elliptical structuring element
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation_pixels * 2 + 1, self.dilation_pixels * 2 + 1))
        dilated_image = binary_dilation(~cleaned_image, structure=structuring_element)
        
        # Invert the dilated image to get the cleaned free space
        cleaned_image = ~dilated_image
        
        return cleaned_image
    
    def find_junction_points(self, skeleton):
        """Find junction points (branch points) in the skeleton, ensuring only one point per junction area."""
        h, w = skeleton.shape
        junction_points = []
        # Keep track of areas that have been marked as junctions
        processed_areas = np.zeros_like(skeleton, dtype=bool)
        
        # Define neighborhood distance (how close can two junction points be)
        neighborhood_distance = 5  # pixels
        
        # First pass: find all potential junction points
        potential_junctions = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x]:
                    # Count neighbors in the 3x3 neighborhood
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - 1  # -1 to exclude the center pixel
                    if neighbors > 2:  # More than 2 neighbors means it's a junction
                        potential_junctions.append((x, y))
        
        # Second pass: filter to keep only one point per junction area
        for x, y in potential_junctions:
            # Skip if this area has already been processed
            if processed_areas[y, x]:
                continue
                
            # Mark this junction point
            junction_points.append((x, y))
            
            # Mark the neighborhood as processed to avoid placing more waypoints nearby
            y_min = max(0, y - neighborhood_distance)
            y_max = min(h, y + neighborhood_distance + 1)
            x_min = max(0, x - neighborhood_distance)
            x_max = min(w, x + neighborhood_distance + 1)
            processed_areas[y_min:y_max, x_min:x_max] = True
        
        self.get_logger().info(f"Found {len(junction_points)} unique junction points in the skeleton")
        return np.array(junction_points) if junction_points else np.empty((0, 2), dtype=int)

    def generate_skeleton_path(self, image):
        """Generate a skeletonized path from the map image."""
        # Apply skeletonization
        new_space = self.clean_image(image)
        # Save the cleaned image for visualization
        cleaned_image_path = self.map_image_path.replace('.pgm', '_cleaned.pgm')
        with open(cleaned_image_path, 'wb') as f:
            f.write(b'P5\n')
            f.write(f'# Cleaned image generated by SkeletonizedPath\n'.encode('ascii'))
            f.write(f'{new_space.shape[1]} {new_space.shape[0]}\n255\n'.encode('ascii'))
            f.write((new_space * 255).astype(np.uint8).tobytes())

        self.get_logger().info(f"Cleaned image saved to {cleaned_image_path}")

        skeleton = skeletonize(new_space)
        self.get_logger().info(f"Skeleton shape: {skeleton.shape}, unique values: {np.unique(skeleton)}")

        # Extract path points from skeleton
        y_indices, x_indices = np.where(skeleton)
        path_points = np.column_stack((x_indices, y_indices))
        
        # Find junction points in the skeleton
        junction_points = self.find_junction_points(skeleton)
        
        # Save the skeleton on the map image for visualization
        output_image = (new_space).astype(np.uint8) * 255  # Convert dilated space to grayscale
        output_image[skeleton] = 0  # Mark skeleton with black
        
        # Save the modified image to a new file
        output_path = self.map_image_path.replace('.pgm', '_skeleton_proba.pgm')
        with open(output_path, 'wb') as f:
            f.write(b'P5\n')
            f.write(f'# Generated by SkeletonizedPath\n'.encode('ascii'))
            f.write(f'{image.shape[1]} {image.shape[0]}\n255\n'.encode('ascii'))
            f.write(output_image.astype(np.uint8).tobytes())
        
        self.get_logger().info(f"Skeleton path saved to {output_path}")
        
        # Use only junction points as waypoints - no fallbacks
        waypoints = junction_points
        
        if len(waypoints) == 0:
            self.get_logger().warn("No junction points found in the skeleton. Path will be empty.")
        else:
            self.get_logger().info(f"Using {len(waypoints)} junction points as waypoints")
            
        # Save the published path points on the map image for visualization
        output_image_with_path = (image > 245).astype(np.uint8) * 255  # Convert map to grayscale
        output_image_with_path = cv2.cvtColor(output_image_with_path, cv2.COLOR_GRAY2RGB)
        
        # Draw skeleton in gray
        for y, x in zip(y_indices, x_indices):
            output_image_with_path[y, x] = [150, 150, 150]  # Gray color
        
        # Draw green dots for junction points (which are our waypoints) - exactly 1 pixel each
        for point in junction_points:
            x_pixel = int(point[0])
            y_pixel = int(point[1])
            # Set exactly one pixel to green for each junction point
            output_image_with_path[y_pixel, x_pixel] = [0, 255, 0]  # Green color, single pixel
        
        # Save the modified image to a new file (using PNG for color support)
        output_path_with_points = self.map_image_path.replace('.pgm', '_path_points.png')
        cv2.imwrite(output_path_with_points, output_image_with_path)
        
        # Create a high-contrast visualization
        high_contrast_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        # Fill background with white
        high_contrast_img.fill(255)
        
        # Draw skeleton in blue
        for y, x in zip(y_indices, x_indices):
            high_contrast_img[y, x] = [255, 0, 0]  # Blue for skeleton
        
        # Draw junction points in green with larger circles for better visibility
        for point in junction_points:
            x_pixel = int(point[0])
            y_pixel = int(point[1])
            cv2.circle(high_contrast_img, (x_pixel, y_pixel), 3, (0, 255, 0), -1)  # Green circles
        
        # Save the high contrast visualization
        high_contrast_path = self.map_image_path.replace('.pgm', '_high_contrast.png')
        cv2.imwrite(high_contrast_path, high_contrast_img)
        
        self.get_logger().info(f"Saved high contrast visualization to {high_contrast_path}")
        self.get_logger().info(f"Path points saved to {output_path_with_points}")

        return waypoints

    def order_path_points(self, points):
        """Order path points to form a continuous path and visualize the ordering."""
        if len(points) < 2:
            return points
        
        # Convert points to a list of tuples for easier handling
        points_list = [tuple(map(int, p)) for p in points]
                
        # Find the lowest point (highest y-coordinate in image coordinates)
        lowest_idx = np.argmax([p[1] for p in points_list])
        start_point = points[lowest_idx]
        self.get_logger().info(f"Starting path from lowest point at ({points_list[lowest_idx][0]}, {points_list[lowest_idx][1]})")
        
        # Start with the lowest point
        ordered = [start_point]
        remaining = np.delete(points, lowest_idx, axis=0)  # Remove by index instead of value
        
        # For the second point, prioritize going left (points with lower x-coordinate)
        if len(remaining) > 0:
            # Get current x-coordinate
            current_x = ordered[0][0]
            
            # Find points to the left of current point
            left_points_mask = remaining[:, 0] < current_x
            
            if np.any(left_points_mask):
                # Filter points to the left
                left_points = remaining[left_points_mask]
                
                # Among left points, find the closest one
                current = ordered[0]
                distances = np.sqrt(np.sum((left_points - current)**2, axis=1))
                idx = np.argmin(distances)
                
                # Add closest left point to ordered list
                ordered.append(left_points[idx])
                
                # Remove the selected point from remaining
                selected_point = left_points[idx]
                remaining = np.array([p for p in remaining if not np.array_equal(p, selected_point)])
                self.get_logger().info(f"Selected left point at ({selected_point[0]}, {selected_point[1]}) as second waypoint")
            else:
                self.get_logger().warn("No points to the left of start point. Using standard nearest neighbor.")
        
        # Continue with greedy nearest neighbor for remaining points
        while len(remaining) > 0:
            current = ordered[-1]
            # Find index of closest point
            distances = np.sqrt(np.sum((remaining - current)**2, axis=1))
            idx = np.argmin(distances)
            # Add closest point to ordered list
            ordered.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)  # Remove by index
        
        # Add the starting point at the end to create a loop
        ordered.append(start_point)
        self.get_logger().info("Added starting point at the end to create a closed loop")
        
        ordered_points = np.array(ordered)
        
        # Create visualization of the ordered path
        try:
            # Try to load the high contrast image if it exists
            path_vis = cv2.imread(self.map_image_path.replace('.pgm', '_high_contrast.png'))
            if path_vis is None:
                # If it doesn't exist, create a new image
                path_vis = np.zeros((points[0].shape[0] * 2, points[0].shape[1] * 2, 3), dtype=np.uint8)
                path_vis.fill(255)  # White background
                
                # If we have the skeleton data, draw it
                if hasattr(self, 'skeleton_indices'):
                    y_indices, x_indices = self.skeleton_indices
                    for y, x in zip(y_indices, x_indices):
                        if 0 <= y < path_vis.shape[0] and 0 <= x < path_vis.shape[1]:
                            path_vis[y, x] = [200, 200, 200]  # Light gray for skeleton
        except Exception as e:
            self.get_logger().warn(f"Could not load high contrast image: {e}. Creating new visualization.")
            # Create a blank visualization
            path_vis = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
            
            # Draw obstacles in black if we have the original image
            try:
                orig_img = self.read_pgm(self.map_image_path)
                obstacles = orig_img < 245
                for y in range(orig_img.shape[0]):
                    for x in range(orig_img.shape[1]):
                        if obstacles[y, x] and y < path_vis.shape[0] and x < path_vis.shape[1]:
                            path_vis[y, x] = [0, 0, 0]  # Black for obstacles
            except:
                pass
        
        # Draw the ordered path
        for i in range(len(ordered_points)-1):
            p1 = (int(ordered_points[i][0]), int(ordered_points[i][1]))
            p2 = (int(ordered_points[i+1][0]), int(ordered_points[i+1][1]))
            
            # Draw an arrow from p1 to p2 to show direction
            cv2.arrowedLine(path_vis, p1, p2, (0, 0, 255), 2, tipLength=0.3)  # Red arrows
            
            # Add numbers to show order
            cv2.putText(path_vis, str(i), (p1[0]+5, p1[1]+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Highlight both the start/end point with a special marker
        if len(ordered_points) > 0:
            start_p = (int(ordered_points[0][0]), int(ordered_points[0][1]))
            cv2.circle(path_vis, start_p, 8, (0, 255, 255), -1)  # Larger yellow circle for start
            
            # Add a special marker to indicate it's both start and end
            cv2.putText(path_vis, "START/END", (start_p[0]+10, start_p[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Save the visualization
        vis_path = self.map_image_path.replace('.pgm', '_path_order.png')
        cv2.imwrite(vis_path, path_vis)
        self.get_logger().info(f"Saved path ordering visualization to {vis_path}")
        
        return ordered_points

    def publish_path(self, path_points):
        """Publish the path as a nav_msgs/Path message with correct coordinate transformation."""
        # Order points to create a coherent path
        ordered_points = self.order_path_points(path_points)
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Get the image dimensions for Y-flipping
        height = 0
        try:
            with open(self.map_image_path, 'rb') as f:
                line = f.readline()  # P5
                line = f.readline()  # Skip comment line
                while line.startswith(b'#'):
                    line = f.readline()
                width, height = map(int, line.decode().strip().split())
        except Exception as e:
            self.get_logger().warn(f"Could not read map dimensions: {e}. Using default Y-flip.")
        
        # Define map origin and orientation 
        # These should be read from your map.yaml file
        # Adding parameters for map origin
                
        map_origin_x = self.get_parameter('map_origin_x').get_parameter_value().double_value
        map_origin_y = self.get_parameter('map_origin_y').get_parameter_value().double_value
        
        self.get_logger().info(f"Using map origin: ({map_origin_x}, {map_origin_y})")
        
        # Apply transformations to each point
        for point in ordered_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            
            # Convert from pixels to meters with correct orientation and origin
            pixel_x = point[0]
            pixel_y = point[1]
            
            # Apply resolution scaling and origin offset
            # First convert from pixel to meters
            if height > 0:
                # Flip Y coordinate (image coordinates have y=0 at top, map has y=0 at bottom)
                world_y = (height - pixel_y) * self.resolution
            else:
                world_y = pixel_y * self.resolution
                
            world_x = pixel_x * self.resolution
            
            # Then add origin offset
            pose.pose.position.x = world_x + map_origin_x
            pose.pose.position.y = world_y + map_origin_y
            
            # Set default orientation
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} poses")
        
        # Log first and last few points with corrected coordinates
        for i in range(min(5, len(path_msg.poses))):
            self.get_logger().info(f"Transformed path point {i}: "
                                f"({path_msg.poses[i].pose.position.x:.2f}, "
                                f"{path_msg.poses[i].pose.position.y:.2f})")
        if len(path_msg.poses) > 10:
            self.get_logger().info("...")
            for i in range(len(path_msg.poses) - 5, len(path_msg.poses)):
                self.get_logger().info(f"Transformed path point {i}: "
                                    f"({path_msg.poses[i].pose.position.x:.2f}, "
                                    f"{path_msg.poses[i].pose.position.y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    
    # Import scikit-image at runtime to avoid startup delays
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        import sys
        print("Error: This node requires scikit-image. Please install it with:")
        print("pip install scikit-image")
        sys.exit(1)
        
    node = SkeletonizedPath()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
