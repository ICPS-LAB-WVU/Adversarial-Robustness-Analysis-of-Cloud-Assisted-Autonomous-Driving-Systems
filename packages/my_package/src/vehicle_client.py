#!/usr/bin/env python3
"""
Vehicle camera client for network-adversary experiments with centralized server logging.

Protocol:
  - Client -> Server header line (ASCII): "img_size,vehicle,frame_id,t_gen\n"
    followed by exactly img_size bytes of JPEG payload.
  - Server -> Client reply line (ASCII): "frame_id,v,omega,t_server,aoi_server\n"
  - Client -> Server (after reply): "frame_id,t_gen,t_send,t_recv,t_apply,local_aoi,local_rtt\n"
"""

import os
import socket
import time
import cv2
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge

SERVER_IP = "10.70.111.40"
PORT = 5001


class CameraReaderNode(DTROS):
    def __init__(self, node_name):
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)

        # --- core params ---
        self.frame_rate = float(rospy.get_param("~frame_rate", os.environ.get("GPU_FRAME_RATE", "15")))
        self.sock_timeout = float(rospy.get_param("~sock_timeout", 2.0))
        self.receiver_ip = rospy.get_param("~gpu_ip", os.environ.get("GPU_SERVER_IP", SERVER_IP))
        self.port = int(rospy.get_param("~gpu_port", os.environ.get("GPU_SERVER_PORT", PORT)))
        self.wait_for_ack = rospy.get_param("~wait_for_ack", True)

        # --- env & topics ---
        self._vehicle_name = os.environ.get("VEHICLE_NAME", "duckiebot")
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"

        # --- pubs/subs ---
        self.vel_pub = rospy.Publisher(
            f"/{self._vehicle_name}/car_cmd_switch_node/cmd",
            Twist2DStamped,
            queue_size=1
        )
        self.img_sub = rospy.Subscriber(
            self._camera_topic,
            CompressedImage,
            self._camera_cb,
            queue_size=1,
            buff_size=2 ** 24
        )

        # --- networking state ---
        self._sock = None
        self._connected = False
        self._reply_buffer = b""
        self._reconnect_backoff = 0.5
        self._reconnect_backoff_max = 5.0

        # --- runtime state ---
        self.latest_image = None
        self._bridge = CvBridge()
        self._frame_counter = 0

        rospy.on_shutdown(self._stop_bot)
        rospy.loginfo(
            "CameraReaderNode initialized frame_rate=%.1f wait_for_ack=%s",
            self.frame_rate,
            self.wait_for_ack,
        )

    # -------------------------------------------------
    # camera callback
    # -------------------------------------------------
    def _camera_cb(self, msg: CompressedImage):
        self.latest_image = msg

    # -------------------------------------------------
    # networking: connect / reconnect with backoff
    # -------------------------------------------------
    def _connect(self):
        """Establish TCP connection to server with exponential backoff."""
        # kill previous socket if any
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False
        self._reply_buffer = b""

        while not rospy.is_shutdown():
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # small send latency helps control loop
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(self.sock_timeout)
                s.connect((self.receiver_ip, self.port))
                # from here on, per-transaction we'll override timeout
                self._sock = s
                self._connected = True
                rospy.loginfo("TCP connected to %s:%d", self.receiver_ip, self.port)

                # reset backoff on success
                self._reconnect_backoff = 0.5
                return
            except Exception as e:
                rospy.logwarn_throttle(
                    5.0,
                    "Connection attempt failed: %s — retrying in %.1fs...",
                    e,
                    self._reconnect_backoff,
                )
                time.sleep(self._reconnect_backoff)
                self._reconnect_backoff = min(
                    self._reconnect_backoff_max,
                    self._reconnect_backoff * 1.5,
                )

    # -------------------------------------------------
    # send frame, wait for reply, send stats
    # -------------------------------------------------
    def _send_frame_and_get_cmd(self, jpeg_bytes: bytes, frame_id: int, t_gen: float):
        """
        Send header + jpeg, then receive a single control reply line.
        Returns (v, omega, aoi_server) if successful.
        Raises on any network failure to trigger reconnect in caller.
        """
        if not self._connected or self._sock is None:
            self._connect()

        # build header
        img_size = len(jpeg_bytes)
        header_line = f"{img_size},{self._vehicle_name},{frame_id},{t_gen:.6f}\n".encode()

        # send header + payload
        t_send = time.time()
        try:
            self._sock.sendall(header_line)
            self._sock.sendall(jpeg_bytes)
        except (BrokenPipeError, ConnectionResetError, OSError, socket.timeout) as e:
            raise RuntimeError(f"send failure: {e}")

        # wait for server reply
        if not self.wait_for_ack:
            # "fire and forget" mode; return last known safe command: straight stop
            return 0.0, 0.0, 0.0

        try:
            # short timeout for reply so we don't hang
            self._sock.settimeout(2.0)
            while b"\n" not in self._reply_buffer:
                chunk = self._sock.recv(128)
                if not chunk:
                    raise RuntimeError("empty reply from server")
                self._reply_buffer += chunk
                if len(self._reply_buffer) > 4096:
                    raise RuntimeError("reply line exceeded 4096 bytes")
            raw_reply, self._reply_buffer = self._reply_buffer.split(b"\n", 1)
        except (socket.timeout, OSError) as e:
            raise RuntimeError(f"recv timeout/no data: {e}")

        t_recv = time.time()

        try:
            reply_line = raw_reply.decode("ascii").strip()
            parts = reply_line.split(",", 5)
            if len(parts) != 6:
                raise ValueError(f"malformed reply: {reply_line!r}")

            veh = parts[0]
            fid = int(parts[1])
            v = float(parts[2])
            omega = float(parts[3])
            t_server = float(parts[4])
            aoi_server = float(parts[5])

            # ignore replies NOT intended for THIS bot
            if veh != self._vehicle_name:
                raise RuntimeError(
                    f"reply vehicle mismatch: expected {self._vehicle_name} got {veh}"
                )
        except Exception as e:
            raise RuntimeError(f"bad parse reply: {e}")

        # sanity check: make sure server is responding to THIS frame
        if fid != frame_id:
            # don't apply stale control from a previous frame
            raise RuntimeError(
                f"reply frame mismatch: expected {frame_id} got {fid}"
            )

        # after getting reply, push stats line back to server (best-effort)
        try:
            t_apply = time.time()
            local_aoi = t_apply - t_gen
            local_rtt = t_recv - t_send
            stats_line = (
                f"{frame_id},{t_gen:.6f},{t_send:.6f},{t_recv:.6f},"
                f"{t_apply:.6f},{local_aoi:.6f},{local_rtt:.6f}\n"
            ).encode()
            self._sock.sendall(stats_line)
        except Exception as e:
            rospy.logwarn_throttle(
                5.0,
                "failed to send timing stats to server: %s",
                e,
            )

        # restore to long-ish timeout for next frame send
        self._sock.settimeout(self.sock_timeout)

        return v, omega, aoi_server

    # -------------------------------------------------
    # actuation helpers
    # -------------------------------------------------
    def publish_cmd(self, v, omega):
        cmd = Twist2DStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.v = float(v)
        cmd.omega = float(omega)
        self.vel_pub.publish(cmd)

    def publish_stop(self):
        self.publish_cmd(0.0, 0.0)

    # -------------------------------------------------
    # main loop
    # -------------------------------------------------
    def run(self):
        rate = rospy.Rate(self.frame_rate)

        while not rospy.is_shutdown():
            if self.latest_image is None:
                rospy.sleep(0.01)
                continue

            # convert ROS CompressedImage -> cv2 BGR
            frame_msg = self.latest_image
            try:
                frame = self._bridge.compressed_imgmsg_to_cv2(frame_msg)
            except Exception:
                try:
                    arr = np.frombuffer(frame_msg.data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                except Exception:
                    frame = None

            if frame is None:
                rospy.logwarn_throttle(5.0, "failed to extract cv2 frame from camera msg")
                rospy.sleep(0.01)
                continue

            # compress outgoing frame
            ok_jpg, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok_jpg:
                rospy.logwarn_throttle(5.0, "JPEG encode failed; skipping frame")
                rospy.sleep(0.01)
                continue

            jpeg = buf.tobytes()

            # frame ID + timestamp
            self._frame_counter += 1
            frame_id = self._frame_counter
            try:
                t_gen = float(frame_msg.header.stamp.to_sec())
            except Exception:
                t_gen = time.time()

            # send to server and act on returned control
            try:
                v, omega, aoi_server = self._send_frame_and_get_cmd(jpeg, frame_id, t_gen)

                # publish control to robot
                self.publish_cmd(v, omega)

                rospy.loginfo_throttle(
                    1.0,
                    "fid=%d v=%.3f ω=%.3f server_aoi=%.3f",
                    frame_id, v, omega, aoi_server
                )

                if not self.wait_for_ack:
                    rate.sleep()
                else:
                    rospy.sleep(0.0005)

            except RuntimeError as net_err:
                # Network broke OR protocol desync OR server lagged.
                rospy.logwarn_throttle(
                    3.0,
                    "network/protocol issue: %s — stopping and reconnecting...",
                    net_err,
                )
                # immediate safety stop
                self.publish_stop()

                # tear down socket HARD and back off a little before reconnect
                if self._sock:
                    try:
                        self._sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                self._connected = False

                # give server time to kill its side cleanly
                time.sleep(0.8)

                # reconnect on next loop iteration
                continue

            except Exception as e:
                # any truly unexpected failure
                rospy.logwarn_throttle(
                    3.0,
                    "unhandled client error: %s",
                    e,
                )
                self.publish_stop()
                time.sleep(0.8)
                self._connected = False
                if self._sock:
                    try:
                        self._sock.shutdown(socket.SHUT_RDWR)
                    except Exception:
                        pass
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                continue

        # end while
        # graceful final stop
        self.publish_stop()

    # -------------------------------------------------
    # shutdown hook
    # -------------------------------------------------
    def _stop_bot(self):
        # publish stop a few times so the wheels actually listen
        for _ in range(3):
            self.publish_stop()
            rospy.sleep(0.05)

        # shutdown socket
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._connected = False
        rospy.loginfo("CameraReaderNode shutdown complete (motors stopped)")


if __name__ == "__main__":
    node = CameraReaderNode("camera_reader_node")
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass

