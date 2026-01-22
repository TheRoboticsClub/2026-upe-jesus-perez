import WebGUI
import HAL
import Frequency
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


# Subscriptor ROS2 imagen
class ROS2(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()
        self.image = None

        self.subscription = self.create_subscription(
            Image,
            '/cam_f1_left/image_raw',
            self.image_callback,
            10
        )

        # Publisher de velocidad
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')


# Procesamiento de imagen
def procesar_imagen(img):
    copia = img.copy()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    momentos = cv2.moments(mask)

    if momentos["m00"] > 0:
        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"])
        cv2.circle(copia, (cx, cy), 8, (0, 255, 0), -1)
        return copia, (cx, cy)
    else:
        return copia, None


# Controlador P / PD / PID
def controlador(error, error_anterior, integral, derivada_anterior,
                dt, modo, Kp, Ki, Kd, alpha):
    # Proporcional
    P = Kp * error

    # Derivada (filtrada)
    D = 0.0
    derivada = 0.0
    if modo >= 2:
        derivada = (error - error_anterior) / dt
        derivada = alpha * derivada_anterior + (1 - alpha) * derivada
        D = Kd * derivada

    # Integral
    I = 0.0
    if modo == 3:
        integral += error * dt
        integral = max(min(integral, 1000), -1000) 
        I = Ki * integral

    salida = P + D + I

    return salida, integral, derivada


# Velocidad adaptativa
def velocidad_adaptativa(w, v_max=5.0, v_min=1.0):
    w_abs = abs(w)
    w_norm = min(w_abs / 1.0, 1.0)
    v = v_max * (1.0 - w_norm)
    return max(v, v_min)


# Main
node = ROS2()

# Parámetros de control
modo = 3          # 1=P, 2=PD, 3=PID
Kp = 0.0025
Kd = 0.0015
Ki = 0.00001      
dt = 0.05
alpha = 0.7 

error_anterior = 0.0
integral = 0.0
derivada_anterior = 0.0

ROS = True #Si es True se usa ROS, si es False HAL:

while True:
    Frequency.tick()
    rclpy.spin_once(node, timeout_sec=0)

    if ROS:
       img = node.image 
    else:
        img = HAL.getImage()

    if img is not None:
        processed, center = procesar_imagen(img)
        WebGUI.showImage(processed)

        if center is not None:
            cx, _ = center

            # Error con signo correcto
            error = (img.shape[1] // 2) - cx

            # Control
            w, integral, derivada = controlador(
                error,
                error_anterior,
                integral,
                derivada_anterior,
                dt,
                modo,
                Kp,
                Ki,
                Kd,
                alpha
            )

            # Saturación
            w = max(min(w, 1.0), -1.0)

            # Velocidad adaptativa
            v = velocidad_adaptativa(w)

            if ROS:
                msg = Twist()
                msg.linear.x = float(v)
                msg.angular.z = float(w)
                node.publisher.publish(msg)
            else:
                HAL.setV(v)
                HAL.setW(w)

            error_anterior = error
            derivada_anterior = derivada
