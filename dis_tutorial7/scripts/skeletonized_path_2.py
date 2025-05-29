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
import heapq  # Add heapq for A* algorithm

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

    def find_skeleton_path_between_points(self, skeleton, start_point, end_point):
        """Find the path along the skeleton between two points using BFS."""
        # Convert points to integer tuples
        start = (int(start_point[0]), int(start_point[1]))
        end = (int(end_point[0]), int(end_point[1]))
        
        # If points are the same, return just that point
        if start == end:
            return [start]
        
        # BFS to find path
        queue = [(start, [start])]
        visited = set([start])
        
        # Directions for 8-connected grid
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        while queue:
            (x, y), path = queue.pop(0)
            
            # Check if we've reached the end
            if (x, y) == end:
                return path
                
            # Check all neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < 0 or ny < 0 or nx >= skeleton.shape[1] or ny >= skeleton.shape[0]:
                    continue
                    
                # Only follow the skeleton
                if not skeleton[ny, nx]:
                    continue
                    
                # Avoid cycles
                if (nx, ny) in visited:
                    continue
                    
                # Add to queue and mark as visited
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
        
        # If no path found
        self.get_logger().warn(f"No skeleton path found between {start} and {end}")
        return []

    def add_midpoints_to_waypoints(self, skeleton, junction_points):
        """Add midpoints along the skeleton between junction points, but only if they are far enough from junctions."""
        if len(junction_points) < 2:
            return junction_points
            
        enhanced_waypoints = []
        # Minimum distance (in pixels) a midpoint should be from any junction
        # Lowered from 20 to 10 to add more midpoints
        min_distance_threshold = 12
        
        # Find connected junction points
        for i in range(len(junction_points)):
            # Add the original junction point
            enhanced_waypoints.append(junction_points[i])
            
            # Find path to next junction point (with wraparound)
            next_idx = (i + 1) % len(junction_points)
            path = self.find_skeleton_path_between_points(
                skeleton, 
                junction_points[i], 
                junction_points[next_idx]
            )
            
            # If path found and it's long enough, consider adding a midpoint
            if len(path) > 2:  # Need at least 3 points to have a meaningful midpoint
                midpoint_idx = len(path) // 2
                midpoint = path[midpoint_idx]
                
                # Check if midpoint is far enough from all junction points
                too_close = False
                for jp in junction_points:
                    dist = np.sqrt((jp[0] - midpoint[0])**2 + (jp[1] - midpoint[1])**2)
                    if dist < min_distance_threshold:
                        too_close = True
                        break
                
                if not too_close:
                    self.get_logger().info(f"Adding midpoint {midpoint} between junctions {i} and {next_idx}")
                    enhanced_waypoints.append(np.array(midpoint))
                else:
                    self.get_logger().info(f"Skipping midpoint {midpoint} as it's too close to a junction")
        
        return np.array(enhanced_waypoints)

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
        self.skeleton_indices = (y_indices, x_indices)  # Store for later use
        path_points = np.column_stack((x_indices, y_indices))
        
        # Find junction points in the skeleton
        junction_points = self.find_junction_points(skeleton)
        
        # Add midpoints between junction points
        waypoints = self.add_midpoints_to_waypoints(skeleton, junction_points)
        
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
        
        # Use enhanced waypoints with both junctions and midpoints
        if len(waypoints) == 0:
            self.get_logger().warn("No waypoints found in the skeleton. Path will be empty.")
        else:
            self.get_logger().info(f"Using {len(waypoints)} waypoints (junction points + midpoints)")
            
        # Save the published path points on the map image for visualization
        output_image_with_path = (image > 245).astype(np.uint8) * 255  # Convert map to grayscale
        output_image_with_path = cv2.cvtColor(output_image_with_path, cv2.COLOR_GRAY2RGB)
        
        # Draw skeleton in gray
        for y, x in zip(y_indices, x_indices):
            output_image_with_path[y, x] = [150, 150, 150]  # Gray color
        
        # Draw green dots for junction points
        for point in junction_points:
            x_pixel = int(point[0])
            y_pixel = int(point[1])
            # Set exactly one pixel to green for each junction point
            output_image_with_path[y_pixel, x_pixel] = [0, 255, 0]  # Green color for junctions
        
        # Draw blue dots for midpoints
        for point in waypoints:
            if not any(np.array_equal(point, jp) for jp in junction_points):
                x_pixel = int(point[0])
                y_pixel = int(point[1])
                # Set exactly one pixel to blue for each midpoint
                output_image_with_path[y_pixel, x_pixel] = [255, 0, 0]  # Blue color for midpoints
        
        # Save the modified image to a new file (using PNG for color support)
        output_path_with_points = self.map_image_path.replace('.pgm', '_path_points.png')
        cv2.imwrite(output_path_with_points, output_image_with_path)
        
        # Create a high-contrast visualization
        high_contrast_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        
        # Fill background with white
        high_contrast_img.fill(255)
        
        # Draw skeleton in light gray
        for y, x in zip(y_indices, x_indices):
            high_contrast_img[y, x] = [200, 200, 200]  # Light gray for skeleton
        
        # Draw junction points in green with larger circles
        for point in junction_points:
            x_pixel = int(point[0])
            y_pixel = int(point[1])
            cv2.circle(high_contrast_img, (x_pixel, y_pixel), 3, (0, 255, 0), -1)  # Green circles
            
        # Draw midpoints in blue with medium circles
        for point in waypoints:
            if not any(np.array_equal(point, jp) for jp in junction_points):
                x_pixel = int(point[0])
                y_pixel = int(point[1])
                cv2.circle(high_contrast_img, (x_pixel, y_pixel), 2, (255, 0, 0), -1)  # Blue circles
        
        # Save the high contrast visualization
        high_contrast_path = self.map_image_path.replace('.pgm', '_high_contrast.png')
        cv2.imwrite(high_contrast_path, high_contrast_img)
        
        self.get_logger().info(f"Saved high contrast visualization to {high_contrast_path}")
        self.get_logger().info(f"Path points saved to {output_path_with_points}")

        return waypoints

    def order_path_points(self, points):
        """Order path points to form a continuous path using A* pathfinding and 2-opt optimization."""
        if len(points) < 2:
            return points
        
        # Convert points to a list of tuples for easier handling
        points_list = [tuple(map(int, p)) for p in points]
                
        # Find the lowest point (highest y-coordinate in image coordinates)
        lowest_idx = np.argmax([p[1] for p in points_list])
        start_point = points[lowest_idx]
        self.get_logger().info(f"Starting path from lowest point at ({points_list[lowest_idx][0]}, {points_list[lowest_idx][1]})")
        
        # Load the map to check for obstacles
        try:
            map_image = self.read_pgm(self.map_image_path)
            # Create obstacle grid: True for obstacles (walls), False for free space
            obstacle_grid = map_image < 245
        except Exception as e:
            self.get_logger().warn(f"Could not read map for obstacle detection: {e}. Using Euclidean distances.")
            obstacle_grid = None
        
        # Function to compute A* path between two points
        def a_star_distance(start, end):
            if obstacle_grid is None:
                # Fallback to Euclidean distance if no obstacle information
                return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            
            # A* implementation
            open_set = []
            closed_set = set()
            
            # Heuristic function (Euclidean distance)
            def h(node):
                return np.sqrt((node[0] - end[0])**2 + (node[1] - end[1])**2)
            
            # Start with the starting node
            heapq.heappush(open_set, (h(start), 0, start))  # (f_score, g_score, node)
            g_scores = {start: 0}
            f_scores = {start: h(start)}
            came_from = {}
            
            # Directions for movement (8-connected grid)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            
            while open_set:
                _, current_g, current = heapq.heappop(open_set)
                
                if current == end:
                    # Path found, return the length
                    return current_g
                
                if current in closed_set:
                    continue
                    
                closed_set.add(current)
                
                # Check all neighbors
                for dx, dy in directions:
                    nx, ny = current[0] + dx, current[1] + dy
                    
                    # Check bounds
                    if nx < 0 or ny < 0 or nx >= obstacle_grid.shape[1] or ny >= obstacle_grid.shape[0]:
                        continue
                        
                    # Skip obstacles
                    if obstacle_grid[ny, nx]:
                        continue
                    
                    neighbor = (nx, ny)
                    
                    # Distance to move (sqrt(2) for diagonals, 1 for orthogonals)
                    move_cost = 1.414 if abs(dx) == abs(dy) else 1
                    tentative_g = g_scores[current] + move_cost
                    
                    if neighbor in closed_set and tentative_g >= g_scores.get(neighbor, float('inf')):
                        continue
                        
                    if tentative_g < g_scores.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_scores[neighbor] = tentative_g
                        f_scores[neighbor] = tentative_g + h(neighbor)
                        heapq.heappush(open_set, (f_scores[neighbor], tentative_g, neighbor))
            
            # No path found, fallback to Euclidean distance multiplied by a penalty factor
            self.get_logger().warn(f"No path found between {start} and {end}, using penalized Euclidean distance.")
            return np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) * 10
        
        # Create distance matrix between all waypoints
        n = len(points)
        distance_matrix = np.zeros((n, n))
        
        # Compute distances between all pairs of points
        self.get_logger().info("Computing distance matrix between waypoints...")
        for i in range(n):
            for j in range(i+1, n):
                dist = a_star_distance(points_list[i], points_list[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Nearest Neighbor algorithm with A* distances
        ordered_indices = [lowest_idx]
        unvisited = set(range(n))
        unvisited.remove(lowest_idx)
        
        while unvisited:
            current = ordered_indices[-1]
            # Find nearest unvisited neighbor
            nearest = min(unvisited, key=lambda i: distance_matrix[current, i])
            ordered_indices.append(nearest)
            unvisited.remove(nearest)
        
        # 2-opt optimization to improve the tour
        self.get_logger().info("Applying 2-opt optimization to improve path...")
        
        def two_opt_swap(route, i, k):
            new_route = route[:i]
            new_route.extend(reversed(route[i:k+1]))
            new_route.extend(route[k+1:])
            return new_route
        
        def calculate_route_distance(route):
            return sum(distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
        
        # Add start point at the end to make it a loop for 2-opt
        ordered_indices.append(lowest_idx)
        
        # 2-opt improvement
        improvement = True
        iterations = 0
        max_iterations = 100  # Limit the number of iterations
        
        while improvement and iterations < max_iterations:
            improvement = False
            best_distance = calculate_route_distance(ordered_indices)
            
            for i in range(1, len(ordered_indices) - 2):
                for k in range(i + 1, len(ordered_indices) - 1):
                    new_route = two_opt_swap(ordered_indices, i, k)
                    new_distance = calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        ordered_indices = new_route
                        best_distance = new_distance
                        improvement = True
                        break
                
                if improvement:
                    break
            
            iterations += 1
        
        self.get_logger().info(f"2-opt completed after {iterations} iterations")
        
        # Remove the duplicated end point
        ordered_indices.pop()
        
        # Convert indices back to points
        ordered = [points[i] for i in ordered_indices]
        
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
        import heapq  # Add heapq import here too for clarity
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
