import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray
import math

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Paramètres pour la position du robot et de la carte
        self.robot_pose = self.declare_parameter('robot_pose', [0.0, 0.00, math.pi]).get_parameter_value().double_array_value
        self.map_bounds = self.declare_parameter('map_bounds', [0.30, 0.30]).get_parameter_value().double_array_value
        self.cluster_distance = self.declare_parameter('cluster_distance', 0.1).get_parameter_value().double_value
        self.detection_distance = self.declare_parameter('detection_distance', 0.3).get_parameter_value().double_value  # 30 cm

        # Publications
        self.evitement_pub = self.create_publisher(Bool, 'evitementFlag', 10)
        self.pose_enemy_pub = self.create_publisher(Float32MultiArray, 'poseEnemy', 10)

        # Souscription au LiDAR
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    def scan_callback(self, msg: LaserScan):
        # Convertir les points LiDAR en coordonnées globales
        points = self.convert_scan_to_global_points(msg)

        # Filtrer les points dans la carte
        filtered_points = self.filter_points_in_map(points)

        # Identifier les objets à partir des clusters
        objects = self.cluster_points(filtered_points)

        # Vérifier la distance et publier les résultats
        self.process_objects(objects)

    def convert_scan_to_global_points(self, msg):
        """Convertir les points LiDAR en coordonnées globales."""
        points = []
        for i, distance in enumerate(msg.ranges):
            if not (msg.range_min <= distance <= msg.range_max):
                continue
            
            angle = msg.angle_min + i * msg.angle_increment
            x_rel = distance * math.cos(angle)
            y_rel = distance * math.sin(angle)
            x_global = (
                x_rel * math.cos(self.robot_pose[2]) - y_rel * math.sin(self.robot_pose[2]) + self.robot_pose[0]
            )
            y_global = (
                x_rel * math.sin(self.robot_pose[2]) + y_rel * math.cos(self.robot_pose[2]) + self.robot_pose[1]
            )
            points.append((x_global, y_global))
        return points

    def filter_points_in_map(self, points):
        """Filtrer les points pour ne garder que ceux à l'intérieur de la carte."""
        x_min, x_max = 0, self.map_bounds[0]
        y_min, y_max = 0, self.map_bounds[1]
        return [(x, y) for x, y in points if x_min <= x <= x_max and y_min <= y <= y_max]

    def cluster_points(self, points):
        """Regrouper les points proches en clusters représentant des objets."""
        clusters = []
        current_cluster = []

        for point in points:
            if not current_cluster:
                current_cluster.append(point)
                continue

            distance = math.sqrt(
                (point[0] - current_cluster[-1][0])**2 + (point[1] - current_cluster[-1][1])**2
            )
            if distance <= self.cluster_distance:
                current_cluster.append(point)
            else:
                clusters.append(current_cluster)
                current_cluster = [point]

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def process_objects(self, clusters):
        """Identifier les objets proches et publier les résultats."""
        for cluster in clusters:
            # Calculer le centre de l'objet
            x_coords = [p[0] for p in cluster]
            y_coords = [p[1] for p in cluster]
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)

            # Calculer la distance relative au robot
            rel_x = center_x - self.robot_pose[0]
            rel_y = center_y - self.robot_pose[1]
            distance_to_robot = math.sqrt(rel_x**2 + rel_y**2)

            # Vérifier si l'objet est proche (à 30 cm ou moins)
            if distance_to_robot <= self.detection_distance:
                # Publier le flag de détection
                evitement_msg = Bool()
                evitement_msg.data = True
                self.evitement_pub.publish(evitement_msg)

                # Publier la position relative et absolue
                pose_msg = Float32MultiArray()
                pose_msg.data = [rel_x, rel_y, center_x, center_y]
                self.pose_enemy_pub.publish(pose_msg)

                # Afficher des logs pour le suivi
               # self.get_logger().info(f"Objet détecté à {distance_to_robot:.2f} m. Position relative : ({rel_x:.2f}, {rel_y:.2f})")
                return

        # Si aucun objet proche, publier un flag négatif
        evitement_msg = Bool()
        evitement_msg.data = False
        self.evitement_pub.publish(evitement_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
