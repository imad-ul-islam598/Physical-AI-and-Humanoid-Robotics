---
sidebar_position: 15
---

# Chapter 15: Perception and Object Interaction

## Introduction

Perception and object interaction form the foundation of Physical AI systems' ability to understand and manipulate their environment. This chapter explores the integration of computer vision, sensor processing, and manipulation planning to enable robots to detect, recognize, and interact with objects in their environment. We'll examine perception pipelines, object detection algorithms, and the coordination between perception and manipulation systems that allow humanoid robots to perform complex object interaction tasks.

## Learning Objectives

By the end of this chapter, you will be able to:
- Implement object detection and recognition pipelines for robotic systems
- Integrate perception data with manipulation planning systems
- Design robust object interaction strategies for humanoid robots
- Handle uncertainty and noise in perception systems
- Coordinate multi-sensor data for improved object understanding

## Key Concepts

- **Object Detection**: Identifying and localizing objects in sensor data
- **Object Recognition**: Classifying objects and understanding their properties
- **Pose Estimation**: Determining the 6D pose (position and orientation) of objects
- **Manipulation Planning**: Planning robot movements to interact with objects
- **Sensor Fusion**: Combining data from multiple sensors for robust perception
- **Grasp Planning**: Determining optimal ways to grasp objects
- **Visual Servoing**: Using visual feedback to control robot movements

## Technical Explanation

Perception and object interaction in Physical AI systems involve multiple interconnected components that work together to enable robots to understand and manipulate objects in their environment. The process typically follows this pipeline:

1. **Sensor Data Acquisition**: Collecting data from various sensors including cameras, LiDAR, depth sensors, and tactile sensors.

2. **Preprocessing**: Cleaning and conditioning sensor data to improve quality and reduce noise.

3. **Object Detection**: Identifying the presence and location of objects in the sensor data.

4. **Object Recognition**: Classifying detected objects and understanding their properties such as size, shape, material, and function.

5. **Pose Estimation**: Determining the 6D pose (position and orientation) of objects relative to the robot or environment.

6. **Grasp Planning**: Determining optimal grasp points and configurations for manipulation.

7. **Manipulation Execution**: Executing planned movements to interact with objects.

**Object Detection** systems in robotics typically use deep learning approaches such as YOLO (You Only Look Once), SSD (Single Shot Detector), or R-CNN (Region-based Convolutional Neural Networks). These systems can detect multiple objects in a single pass through the network, making them suitable for real-time applications.

**Object Recognition** goes beyond detection to classify objects into specific categories and understand their properties. This might include recognizing that an object is a "coffee mug" rather than just "a cylindrical object."

**Pose Estimation** is critical for manipulation tasks as it provides the precise location and orientation needed for successful grasping. This can be achieved through various methods:
- Template matching with known 3D models
- Keypoint detection and correspondence
- Direct regression of pose parameters
- Multi-view geometry approaches

**Sensor Fusion** combines data from multiple sensors to improve perception robustness:
- RGB cameras for color and texture information
- Depth cameras for 3D structure
- LiDAR for accurate distance measurements
- Tactile sensors for contact feedback

**Grasp Planning** algorithms determine where and how to grasp objects based on:
- Object shape and size
- Material properties
- Task requirements
- Robot hand configuration
- Stability considerations

The perception pipeline must handle significant challenges including:
- **Variability**: Objects appear differently under various lighting conditions, viewpoints, and occlusions
- **Real-time Requirements**: Robots need to perceive and react quickly to environmental changes
- **Uncertainty**: Sensor data contains noise and uncertainty that must be properly handled
- **Scale**: Systems must work with objects of various sizes and at various distances

## Diagrams written as text descriptions

**Diagram 1: Perception and Object Interaction Pipeline**
```
RGB-D Camera → Preprocessing → Object Detection → Object Recognition → Pose Estimation
     │              │                │                  │                  │
     ▼              ▼                ▼                  ▼                  ▼
Depth Sensor → Sensor Fusion → Object List → Object Properties → Object Poses
     │              │                │                  │                  │
     ▼              ▼                ▼                  ▼                  ▼
Tactile Data → Multi-Sensor → Object Map → Classification → 6D Poses → Grasp Planning
     │              │                │                  │                  │
     ▼              ▼                ▼                  ▼                  ▼
Environment → Perceptual → Detected → Recognized → Located → Grasp → Manipulation
  State       Context     Objects    Objects     Objects   Points    Planning
```

**Diagram 2: Grasp Planning Process**
```
Object: Coffee Mug
     │
     ▼
Shape Analysis (Cylinder with handle)
     │
     ▼
Stability Analysis (Center of mass, orientation)
     │
     ▼
Grasp Point Generation (Multiple candidates)
     │
     ▼
Quality Evaluation (Force closure, accessibility)
     │
     ▼
Optimal Grasp Selection (Best candidate)
     │
     ▼
Robot Hand Configuration (Joint angles)
     │
     ▼
Approach Path Planning (Collision-free trajectory)
     │
     ▼
Execution (Grasp execution with feedback)
```

**Diagram 3: Visual Servoing Control Loop**
```
Desired Object Pose → Image Error → Velocity Commands → Robot Motion → Actual Object Pose
         │                   │              │              │                   │
         └───────────────────┼──────────────┼──────────────┼───────────────────┘
                             ▼              ▼              ▼
                         Feature          Control        Forward
                         Detection        Law            Kinematics
```

## Code Examples

Here's an example of an object detection and manipulation system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose, PoseStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO
import tf2_ros
from tf2_ros import TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import math

class PerceptionAndManipulationNode(Node):
    def __init__(self):
        super().__init__('perception_manipulation')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize YOLO model (using a pre-trained model)
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # You can use other model sizes too
        except Exception as e:
            self.get_logger().warn(f'Could not load YOLO model: {str(e)}')
            self.yolo_model = None

        # TF2 setup for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/rgb/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/rgb/camera_info', self.camera_info_callback, 10)

        # Publishers
        self.object_pub = self.create_publisher(String, '/detected_objects', 10)
        self.manipulation_goal_pub = self.create_publisher(
            PoseStamped, '/manipulation_goal', 10)

        # Camera parameters
        self.camera_matrix = None
        self.depth_image = None
        self.latest_image = None

        # Object database
        self.object_database = {
            'bottle': {'grasp_type': 'top_grasp', 'approach_distance': 0.15},
            'cup': {'grasp_type': 'side_grasp', 'approach_distance': 0.12},
            'box': {'grasp_type': 'top_grasp', 'approach_distance': 0.20}
        }

        self.get_logger().info('Perception and Manipulation node initialized')

    def camera_info_callback(self, msg):
        """Process camera calibration information"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)

    def depth_callback(self, msg):
        """Process depth image"""
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
            self.depth_image = depth_image
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def image_callback(self, msg):
        """Process RGB image for object detection"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image = cv_image

            # Perform object detection
            if self.yolo_model is not None:
                results = self.yolo_model(cv_image)

                # Process detection results
                detected_objects = []
                for result in results:
                    for box in result.boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # Get class name
                        class_name = result.names[cls]

                        # Filter by confidence
                        if conf > 0.5:  # Confidence threshold
                            obj_info = {
                                'class': class_name,
                                'confidence': float(conf),
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
                            }

                            # Estimate 3D position if depth is available
                            if self.depth_image is not None:
                                center_x, center_y = int(obj_info['center'][0]), int(obj_info['center'][1])
                                if (0 <= center_x < self.depth_image.shape[1] and
                                    0 <= center_y < self.depth_image.shape[0]):
                                    depth = self.depth_image[center_y, center_x]
                                    if not np.isnan(depth) and depth > 0:
                                        # Convert pixel coordinates to 3D world coordinates
                                        world_pos = self.pixel_to_world(
                                            center_x, center_y, depth)
                                        obj_info['world_position'] = world_pos

                            detected_objects.append(obj_info)

                # Publish detected objects
                self.publish_detected_objects(detected_objects)

                # Visualize detections on the image
                self.visualize_detections(cv_image, detected_objects)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def pixel_to_world(self, u, v, depth):
        """Convert pixel coordinates to world coordinates"""
        if self.camera_matrix is None:
            return [0.0, 0.0, 0.0]

        # Camera intrinsic parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Convert to world coordinates
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return [x, y, z]

    def publish_detected_objects(self, objects):
        """Publish detected objects information"""
        if not objects:
            return

        # Create and publish object information
        for obj in objects:
            obj_msg = String()
            obj_msg.data = f"Object: {obj['class']}, Confidence: {obj['confidence']:.2f}"
            self.object_pub.publish(obj_msg)

        # If we have a target object, plan manipulation
        target_obj = self.find_target_object(objects)
        if target_obj:
            self.plan_manipulation(target_obj)

    def find_target_object(self, objects):
        """Find the most suitable object for manipulation"""
        # For this example, we'll look for the highest confidence object
        # that's in our object database
        for obj in sorted(objects, key=lambda x: x['confidence'], reverse=True):
            if obj['class'] in self.object_database:
                return obj
        return None

    def plan_manipulation(self, obj):
        """Plan manipulation for the detected object"""
        if 'world_position' not in obj:
            self.get_logger().warn('Cannot plan manipulation without world position')
            return

        world_pos = obj['world_position']
        obj_class = obj['class']

        # Get object-specific parameters
        obj_params = self.object_database.get(obj_class, {})
        approach_distance = obj_params.get('approach_distance', 0.15)

        # Create manipulation goal
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'camera_link'  # or appropriate frame
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        # Set position (with approach distance)
        goal_pose.pose.position.x = world_pos[0] - approach_distance  # Approach from front
        goal_pose.pose.position.y = world_pos[1]
        goal_pose.pose.position.z = world_pos[2]

        # Set orientation (facing the object)
        goal_pose.pose.orientation.w = 1.0  # Simple orientation for now

        # Transform to base frame if needed
        try:
            goal_pose = self.transform_pose_to_base_frame(goal_pose)
        except Exception as e:
            self.get_logger().warn(f'Could not transform pose: {str(e)}')
            return

        # Publish manipulation goal
        self.manipulation_goal_pub.publish(goal_pose)
        self.get_logger().info(f'Planned manipulation for {obj_class} at {world_pos}')

    def transform_pose_to_base_frame(self, pose_stamped):
        """Transform pose from camera frame to robot base frame"""
        try:
            # Wait for transform
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # Target frame
                pose_stamped.header.frame_id,  # Source frame
                rclpy.time.Time(),  # Use latest available
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Apply transform (simplified - in practice, use tf2 geometry_msgs functions)
            # This is a placeholder for actual transformation
            return pose_stamped

        except Exception as e:
            self.get_logger().warn(f'Could not transform pose: {str(e)}')
            return pose_stamped

    def visualize_detections(self, image, objects):
        """Draw bounding boxes and labels on the image"""
        vis_image = image.copy()

        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['class']}: {obj['confidence']:.2f}"

            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            cv2.putText(vis_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            center_x, center_y = int(obj['center'][0]), int(obj['center'][1])
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Optionally publish the visualized image
        # (would need an image publisher for this)

def main(args=None):
    rclpy.init(args=args)
    perception_node = PerceptionAndManipulationNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down perception node...')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Example of grasp planning for different object types:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class GraspPlanner:
    def __init__(self):
        # Define grasp types and their parameters
        self.grasp_types = {
            'top_grasp': {
                'approach_direction': [0, 0, -1],  # Approach from above
                'gripper_width_range': [0.02, 0.15],
                'max_object_height': 0.3
            },
            'side_grasp': {
                'approach_direction': [-1, 0, 0],  # Approach from side
                'gripper_width_range': [0.03, 0.20],
                'max_object_width': 0.15
            },
            'pinch_grasp': {
                'approach_direction': [0, -1, 0],  # Approach from front
                'gripper_width_range': [0.01, 0.08],
                'min_object_thickness': 0.005
            }
        }

    def plan_grasp(self, object_info):
        """
        Plan a grasp for the given object
        object_info: dict with keys like 'class', 'bbox', 'world_position', etc.
        """
        obj_class = object_info['class']
        obj_dims = self.estimate_object_dimensions(object_info)

        # Determine best grasp type based on object properties
        grasp_type = self.select_grasp_type(obj_class, obj_dims)

        if grasp_type is None:
            return None

        # Generate grasp poses
        grasp_poses = self.generate_grasp_poses(object_info, grasp_type, obj_dims)

        # Evaluate and select best grasp
        best_grasp = self.evaluate_grasps(grasp_poses, object_info)

        return best_grasp

    def estimate_object_dimensions(self, object_info):
        """Estimate object dimensions from detection info"""
        if 'bbox' in object_info and 'world_position' in object_info:
            # Use bounding box and depth to estimate dimensions
            bbox = object_info['bbox']
            center_3d = object_info['world_position']

            # This is a simplified estimation
            # In practice, you'd use more sophisticated methods
            width = abs(bbox[2] - bbox[0])  # pixel width
            height = abs(bbox[3] - bbox[1])  # pixel height

            # Convert to real dimensions using depth and camera parameters
            # This is a placeholder - actual implementation would be more complex
            real_width = width * center_3d[2] / 1000  # Simplified conversion
            real_height = height * center_3d[2] / 1000  # Simplified conversion

            return {
                'width': real_width,
                'height': real_height,
                'depth': real_width * 0.5  # Assumed depth
            }

        return {'width': 0.1, 'height': 0.1, 'depth': 0.1}  # Default values

    def select_grasp_type(self, obj_class, obj_dims):
        """Select appropriate grasp type based on object class and dimensions"""
        # Default mapping based on common object types
        class_to_grasp = {
            'bottle': 'top_grasp',
            'cup': 'side_grasp',
            'can': 'top_grasp',
            'box': 'top_grasp',
            'book': 'side_grasp',
            'phone': 'pinch_grasp'
        }

        # Check if object class has a preferred grasp
        preferred_grasp = class_to_grasp.get(obj_class)

        if preferred_grasp and self.is_grasp_feasible(preferred_grasp, obj_dims):
            return preferred_grasp

        # If preferred grasp is not feasible, try alternatives
        for grasp_type in self.grasp_types:
            if self.is_grasp_feasible(grasp_type, obj_dims):
                return grasp_type

        return None

    def is_grasp_feasible(self, grasp_type, obj_dims):
        """Check if a grasp type is feasible for the given object dimensions"""
        params = self.grasp_types[grasp_type]

        # Check gripper width constraints
        if 'width' in obj_dims:
            obj_width = obj_dims['width']
            min_width, max_width = params['gripper_width_range']
            if not (min_width <= obj_width <= max_width):
                return False

        # Check other constraints based on grasp type
        if grasp_type == 'top_grasp':
            if 'height' in obj_dims and obj_dims['height'] > params.get('max_object_height', 1.0):
                return False
        elif grasp_type == 'side_grasp':
            if 'width' in obj_dims and obj_dims['width'] > params.get('max_object_width', 1.0):
                return False
        elif grasp_type == 'pinch_grasp':
            if 'depth' in obj_dims and obj_dims['depth'] < params.get('min_object_thickness', 0.001):
                return False

        return True

    def generate_grasp_poses(self, object_info, grasp_type, obj_dims):
        """Generate candidate grasp poses"""
        world_pos = object_info['world_position']
        approach_dir = np.array(self.grasp_types[grasp_type]['approach_direction'])

        # Generate multiple grasp poses around the object
        grasp_poses = []

        # Main grasp pose - at the object center with appropriate orientation
        main_pose = self.create_grasp_pose(world_pos, approach_dir)
        grasp_poses.append({
            'pose': main_pose,
            'type': grasp_type,
            'quality': 1.0  # Initial quality score
        })

        # Additional poses at different angles around the object
        for angle in [math.pi/4, math.pi/2, 3*math.pi/4]:
            rotated_approach = self.rotate_vector_around_z(approach_dir, angle)
            rotated_pos = world_pos.copy()
            # Adjust position based on rotation
            offset_distance = obj_dims.get('width', 0.1) * 0.6
            rotated_pos[0] += rotated_approach[0] * offset_distance
            rotated_pos[1] += rotated_approach[1] * offset_distance

            pose = self.create_grasp_pose(rotated_pos, rotated_approach)
            grasp_poses.append({
                'pose': pose,
                'type': grasp_type,
                'quality': 0.8  # Slightly lower quality for non-center grasps
            })

        return grasp_poses

    def create_grasp_pose(self, position, approach_direction):
        """Create a grasp pose with appropriate orientation"""
        # The grasp approach direction should be the -z axis of the gripper frame
        # Calculate orientation to achieve this
        z_axis = -np.array(approach_direction) / np.linalg.norm(approach_direction)

        # Choose x-axis perpendicular to z-axis
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [1, 0, 0])

        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Create rotation matrix
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # Convert to quaternion
        r = R.from_matrix(rotation_matrix)
        quat = r.as_quat()  # [x, y, z, w]

        pose = {
            'position': position,
            'orientation': quat
        }

        return pose

    def rotate_vector_around_z(self, vector, angle):
        """Rotate a 3D vector around the Z-axis by the given angle"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        x_rot = vector[0] * cos_a - vector[1] * sin_a
        y_rot = vector[0] * sin_a + vector[1] * cos_a
        z_rot = vector[2]

        return np.array([x_rot, y_rot, z_rot])

    def evaluate_grasps(self, grasp_poses, object_info):
        """Evaluate and select the best grasp from candidates"""
        if not grasp_poses:
            return None

        # For now, return the highest quality grasp
        # In practice, you'd consider collision checking, accessibility, etc.
        best_grasp = max(grasp_poses, key=lambda x: x['quality'])
        return best_grasp

# Example usage
def example_usage():
    planner = GraspPlanner()

    # Example object info (as would come from perception system)
    object_info = {
        'class': 'cup',
        'world_position': [0.5, 0.2, 0.1],
        'bbox': [100, 150, 200, 250]  # pixel coordinates
    }

    best_grasp = planner.plan_grasp(object_info)

    if best_grasp:
        print(f"Best grasp pose: {best_grasp['pose']}")
        print(f"Grasp type: {best_grasp['type']}")
        print(f"Quality: {best_grasp['quality']}")
    else:
        print("No feasible grasp found")
```

Example of sensor fusion for improved object perception:

```python
import numpy as np
from scipy.spatial import distance
import cv2

class SensorFusionPerception:
    def __init__(self):
        self.object_tracks = {}  # Track objects across time
        self.max_track_age = 10  # Maximum frames to keep track
        self.iou_threshold = 0.3  # Minimum IoU for association

    def fuse_sensor_data(self, rgb_detections, depth_data, camera_info):
        """
        Fuse RGB object detections with depth information
        """
        fused_objects = []

        for detection in rgb_detections:
            # Get 3D position from depth
            center_x, center_y = int(detection['center'][0]), int(detection['center'][1])

            if (0 <= center_x < depth_data.shape[1] and
                0 <= center_y < depth_data.shape[0]):

                depth = depth_data[center_y, center_x]

                if not np.isnan(depth) and depth > 0:
                    # Convert to 3D world coordinates
                    world_pos = self.pixel_to_world(
                        center_x, center_y, depth, camera_info)

                    # Add 3D information to detection
                    detection['world_position'] = world_pos
                    detection['distance'] = float(depth)

                    # Estimate size using depth and bounding box
                    bbox_size = detection['bbox'][2] - detection['bbox'][0]  # width in pixels
                    real_size = self.estimate_real_size(bbox_size, depth, camera_info)
                    detection['estimated_size'] = real_size

            fused_objects.append(detection)

        return fused_objects

    def pixel_to_world(self, u, v, depth, camera_info):
        """Convert pixel coordinates to world coordinates"""
        k_matrix = np.array(camera_info.k).reshape(3, 3)
        fx, fy = k_matrix[0, 0], k_matrix[1, 1]
        cx, cy = k_matrix[0, 2], k_matrix[1, 2]

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return [x, y, z]

    def estimate_real_size(self, pixel_size, depth, camera_info):
        """Estimate real size of object from pixel size and depth"""
        k_matrix = np.array(camera_info.k).reshape(3, 3)
        focal_length = (k_matrix[0, 0] + k_matrix[1, 1]) / 2  # Average focal length

        # Simple estimation: real_size = (pixel_size * real_depth) / focal_length
        # This is approximate and depends on the object's orientation
        estimated_size = (pixel_size * depth) / focal_length

        return estimated_size

    def associate_detections_with_tracks(self, detections):
        """
        Associate new detections with existing tracks using IoU
        """
        updated_tracks = {}

        for det in detections:
            best_match = None
            best_iou = 0

            # Find best matching track
            for track_id, track in self.object_tracks.items():
                iou = self.calculate_iou(det['bbox'], track['last_detection']['bbox'])

                if iou > self.iou_threshold and iou > best_iou:
                    best_match = track_id
                    best_iou = iou

            if best_match is not None:
                # Update existing track
                self.object_tracks[best_match]['last_detection'] = det
                self.object_tracks[best_match]['age'] = 0
                self.object_tracks[best_match]['detections'].append(det)
                updated_tracks[best_match] = self.object_tracks[best_match]
            else:
                # Create new track
                new_id = len(self.object_tracks) + 1000  # Avoid conflicts
                self.object_tracks[new_id] = {
                    'id': new_id,
                    'last_detection': det,
                    'age': 0,
                    'detections': [det],
                    'class': det['class']
                }
                updated_tracks[new_id] = self.object_tracks[new_id]

        # Age existing tracks and remove old ones
        for track_id in list(self.object_tracks.keys()):
            if track_id not in updated_tracks:
                self.object_tracks[track_id]['age'] += 1

                if self.object_tracks[track_id]['age'] > self.max_track_age:
                    del self.object_tracks[track_id]
                else:
                    updated_tracks[track_id] = self.object_tracks[track_id]

        self.object_tracks = updated_tracks

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

# Example usage
def main():
    # This would be integrated into the main perception node
    fusion_system = SensorFusionPerception()

    # Example: fuse detections from different sensors
    # rgb_detections = get_rgb_detections()  # From camera
    # depth_data = get_depth_data()          # From depth sensor
    # camera_info = get_camera_info()        # From camera info topic

    # fused_objects = fusion_system.fuse_sensor_data(
    #     rgb_detections, depth_data, camera_info)
    #
    # fusion_system.associate_detections_with_tracks(fused_objects)

    print("Sensor fusion system ready")

if __name__ == '__main__':
    main()
```

## Exercises

1. **Object Detection**: Implement a YOLO-based object detection system integrated with ROS 2 for real-time object recognition.

2. **Grasp Planning**: Create a grasp planning algorithm that can handle different object shapes and sizes for a humanoid robot hand.

3. **Sensor Fusion**: Develop a system that combines RGB, depth, and LiDAR data for robust object perception.

4. **Manipulation Strategy**: Design a manipulation strategy that adapts to object properties (weight, fragility, shape) for safe interaction.

## Summary

Perception and object interaction are fundamental capabilities for Physical AI systems, enabling robots to understand their environment and perform meaningful tasks. The integration of computer vision, sensor processing, and manipulation planning creates the foundation for robots to detect, recognize, and interact with objects in complex environments. Success in this domain requires robust algorithms that can handle uncertainty, real-time constraints, and the complexity of real-world scenarios. As we continue our exploration of Physical AI, perception and manipulation systems will serve as the crucial link between the robot's cognitive planning and its physical interaction with the world.