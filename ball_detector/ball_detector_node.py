from rclpy.node import Node
import rclpy
from std_msgs.msg import String
from ultralytics import YOLO
import os
from std_msgs.msg import Bool
import pyrealsense2 as rs
import numpy as np
import cv2
from imrc_messages.msg import BallInfo
from imrc_messages.msg import LedControl


# ===============================
# 調整用パラメータ（ロボット・環境依存なので適宜調整をするお）
# ===============================
IMG_W, IMG_H = 640, 480

# 制御ループ周期　これは変えないほうがいいかも　
FPS = 15

# 画面中心のオフセット（単位はピクセル）
center_paramX = 23
center_paramY = -50

#この２つはoperaterと合致するようにしないと正しくGUIが使えない
#14がマックス
DX_TH = 10
#36がマックス
DY_TH = 20

#目標とするボールまでの距離の範囲（単位はcm）
DEPTH_MIN = 41.0
DEPTH_MAX = 51.0


# YOLO の信頼度しきい値
CONF_TH = 0.15

# 画面上で「目標のボールの中心」とみなす座標を指定   
CENTER_X, CENTER_Y = (IMG_W // 2) + center_paramX, (IMG_H // 2) + center_paramY


# 検出ロスト許容フレーム数
MAX_MISS = 5

CV2_WINDOW_X = 640
CV2_WINDOW_Y = 640

class BallDetector(Node):

    def __init__(self):
        super().__init__('ball_detector')
        
        self.last = None
        self.miss = 0

        self.current_model = None
        self.target_color = None
 
        cv2.namedWindow("ball_detector", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ball_detector", CV2_WINDOW_X, CV2_WINDOW_Y)  

        # ===== RealSense 初期化 =====
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)

        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # ===== YOLO モデル読み込み =====
        base = os.path.dirname(__file__)
        self.model_red    = YOLO(os.path.join(base, 'best_red.pt'))
        self.model_blue   = YOLO(os.path.join(base, 'best_blue.pt'))
        self.model_yellow = YOLO(os.path.join(base, 'best_yellow.pt'))

        # ===== Publisher ====
        self.status_pub = self.create_publisher(Bool,'detect_ball_status',10)
        self.ball_pub = self.create_publisher(BallInfo,'ball_info',10)
        self.led_pub = self.create_publisher(LedControl,'led_cmd',10)
        self.depth_pub = self.create_publisher(String,'depth_status',10)

        # ===== Subscriber =====
        self.create_subscription(String,'detect_ball_color',self.color_cb,10)


        # ===== タイマー（制御ループ）=====
        self.create_timer(1.0 / FPS, self.timer_cb)
        # =================================

        self.msg_led = LedControl()

    # ===============================
    # 色を受け取るコールバック
    # ===============================
    def color_cb(self, msg):
        
        self.target_color = msg.data


        # ====受け取った色に応じてモデルを切り替える====
        if self.target_color == "赤":
            self.current_model = self.model_red
            
            # LED を赤く点灯させる
            self.msg_led.led_brightness = 1.0    #明るさ　0.0～1.0
            self.msg_led.led_index = 5           #私に使うことが許されるのは5番LED
            self.msg_led.led_color = "RED"       #色
            self.msg_led.led_mode = "apply"      #gblinkはじんわりブリンク、applyはに点灯、brinnkは点滅
            self.msg_led.blink_duration = 1000.0 #周期　1000で1秒
            self.led_pub.publish(self.msg_led)
            
        elif self.target_color == "青":
            self.current_model = self.model_blue

            # LED を青く点灯させる
            self.msg_led.led_brightness = 1.0
            self.msg_led.led_index = 5
            self.msg_led.led_color = "BLUE"
            self.msg_led.led_mode = "apply"
            self.msg_led.blink_duration = 1000.0
            self.led_pub.publish(self.msg_led)

        elif self.target_color == "黄":
            self.current_model = self.model_yellow

            # LED を黄色に点灯させる
            self.msg_led.led_brightness = 1.0
            self.msg_led.led_index = 5
            self.msg_led.led_color = "YELLOW"
            self.msg_led.led_mode = "apply"
            self.msg_led.blink_duration = 1000.0
            self.led_pub.publish(self.msg_led)

        else:
            self.current_model = None
            msg = Bool()
            msg.data = False
            self.status_pub.publish(msg)
            self.get_logger().warn(f"不正な色を受信: {self.target_color}")

        self.get_logger().info(f"現在の検出対象: {self.target_color}")
        # ==============================================    


    # ===============================
    # メイン処理
    # ===============================
    def timer_cb(self):

        # 色未指定なら「未検出」を送る
        if self.current_model is None:
            self.publish_ball_info()
            return

        # ---- カメラ取得 ----
        frames = self.pipeline.wait_for_frames()
        cf, df = frames.get_color_frame(), frames.get_depth_frame()
        if not cf or not df:
            return

        color = np.asanyarray(cf.get_data())
        depth = np.asanyarray(df.get_data())

        # ===============================
        # 画面中心点にある物体までの距離 を ぱぶりっしゅ
        # ===============================
        center_depth_raw = depth[(IMG_W // 2), (IMG_H // 2)]
        #　カメラで数値を取得できたとき
        if center_depth_raw > 0:
            center_depth_cm = center_depth_raw * self.depth_scale * 100.0
            msg = String()
            msg.data = f"{center_depth_cm:.1f}"
        else:
            msg = String()
            msg.data = "測定不能"
        self.depth_pub.publish(msg)

        # ---- YOLO 検出 ----
        dets = []
        for r in self.current_model(color, conf=CONF_TH, verbose=False):
            if not r.boxes:
                continue
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf[0])   
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                d = depth[cy, cx]
                if d <= 0:
                    continue

                dets.append({
                    "bbox": (x1, y1, x2, y2),
                    "dx": cx - CENTER_X,
                    "dy": cy - CENTER_Y,
                    "depth": d * self.depth_scale * 100.0,
                    "conf": conf
                })

        # ===============================
        # GUI 描画
        # ===============================
        draw = color.copy()

        # 画面中心マーカー
        cv2.drawMarker(
            draw,
            (CENTER_X, CENTER_Y),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            30,
            2
        )
        draw = color.copy()

        # ===== 目標点 =====
        cv2.drawMarker(
            draw,
            (CENTER_X, CENTER_Y),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            40,
            2
        )

        if dets:
            self.last = min(dets, key=lambda x: x["depth"])
            self.miss = 0

            x1, y1, x2, y2 = self.last["bbox"]
            dx = self.last["dx"]
            dy = self.last["dy"]
            depth_cm = self.last["depth"]
            confidence = self.last["conf"]
            # ===== ボール中心座標 =====
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # バウンディングボックス
            cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #  ボール中心点（赤丸）★
            cv2.circle(draw, (cx, cy), 6, (0, 0, 255), -1)

            #  中点から画面中心への線 
            cv2.line(
                draw,
                (CENTER_X, CENTER_Y),
                (cx, cy),
                (255, 0, 0),
                2
            )

            # 情報テキスト
            # cv2.putText(
            #     draw,
            #     f"cx:{cx} cy:{cy}",
            #     (x1, y2 + 20),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 255),
            #     2
            # )



            # ===== しきい値判定 =====
            dx_ok = abs(dx) <= DX_TH
            dy_ok = abs(dy) <= DY_TH
            depth_ok = DEPTH_MIN <= depth_cm <= DEPTH_MAX

            dx_color = (0, 255, 0) if dx_ok else (0, 0, 255)
            dy_color = (0, 255, 0) if dy_ok else (0, 0, 255)
            depth_color = (0, 255, 0) if depth_ok else (0, 0, 255)

            # ===== 個別表示 =====
            y_text = y1 - 10

            cv2.putText(
                draw,
                f"dx:{dx}",
                (x1, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                dx_color,
                2
            )

            cv2.putText(
                draw,
                f"dy:{dy}",
                (x1 + 80, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                dy_color,
                2
            )

            cv2.putText(
                draw,
                f"depth:{depth_cm:.1f}cm",
                (x1 + 160, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                depth_color,
                2
            )

            cv2.putText(
                draw,
                f"conf:{confidence:.2f}",
                (x1 + 20, y2+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            self.publish_ball_info(dx, dy, depth_cm)
        else:
            self.miss += 1
            if self.miss > MAX_MISS:
                self.last = None
                self.publish_ball_info()

        # ウィンドウ表示
        cv2.imshow("ball_detector", draw)
        cv2.waitKey(1)


    def publish_ball_info(self, dx=None, dy=None, depth=None):
        msg = BallInfo()

        if dx is None:
            msg.detected = False
        else:
            msg.detected = True
            msg.dx = dx
            msg.dy = dy
            msg.depth_cm = depth

        self.ball_pub.publish(msg)


def main():
    rclpy.init()
    node = BallDetector()
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()