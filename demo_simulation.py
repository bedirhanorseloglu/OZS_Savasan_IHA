"""
Savaşan İHA - Canlı Simülasyon ve Algoritma Oynatıcı (Standalone Simulator & Player)

===============================================================================
🧑‍🏫 BU PROGRAM NEDİR?
===============================================================================
Bu program, Gazebo / ROS / SITL simülasyonuna gerek kalmadan;
Gazebo simülasyonundan kaydedilmiş uçuş verisi üzerinde gerçek zamanlı olarak:
1. YOLOv11 Derin Öğrenme Modelini,
2. Genişletilmiş Kalman Filtresi (EKF) Takipçisini,
3. Şartnamedeki 5 Kuralı (Kilitlenme Değerlendirici),
4. Görev Durum Makinesini (FSM: Arama -> Yönelme -> Takip -> Kilitlenme),
5. Oransal Seyrüsefer (PN) Güdüm Hesaplamalarını
çalıştırır ve ekranda canlı HUD nişangahı ile gösterir!

Çıkmak için 'q' tuşuna, duraklatmak için 'BOŞLUK' (SPACE) tuşuna basabilirsiniz.
===============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
import cv2
import numpy as np

# Proje kök dizini
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVASAN_DIR = os.path.join(PROJECT_ROOT, "Savaşan_İHA_Görevi")
sys.path.insert(0, SAVASAN_DIR)

from core.config import Config
from core.logger import init_logger
from perception.detector import Detector
from perception.tracker import SingleTargetTracker, TrackState
from perception.lock_evaluator import LockEvaluator
from perception.reacquisition import Reacquisition
from control.visual_servo import VisualServo
from control.distance_controller import DistanceController
from mission.fsm import MissionFSM, MissionState
from iol.hud import HUD


def run_demo(video_path: str = None):
    # Loglama
    init_logger(console=True)

    if video_path is None:
        video_path = os.path.join(SAVASAN_DIR, "assets", "savaşan_iha.mp4")

    if not os.path.exists(video_path):
        print(f"❌ Video dosyası bulunamadı: {video_path}")
        return

    # Yapılandırma yükle
    cfg_path = os.path.join(SAVASAN_DIR, "config", "default.yaml")
    cfg = Config.from_yaml(cfg_path)

    print("=" * 65)
    print("🎯 SAVAŞAN İHA - CANLI SİMÜLASYON OYNATICI BAŞLATILIYOR")
    print(f"🎥 Video Kaynağı: {os.path.basename(video_path)}")
    print("=" * 65)

    # Modülleri Başlat
    model_path = os.path.join(SAVASAN_DIR, cfg.detector.model_path)
    detector = Detector(
        model_path=model_path,
        confidence_threshold=cfg.detector.confidence_threshold,
        iou_threshold=cfg.detector.iou_threshold,
        imgsz=cfg.detector.imgsz,
        max_detections=cfg.detector.max_detections,
        target_classes=cfg.detector.target_classes,
    )

    tracker = SingleTargetTracker(
        process_noise_pos=cfg.tracker.process_noise_pos,
        process_noise_vel=cfg.tracker.process_noise_vel,
        process_noise_size=cfg.tracker.process_noise_size,
        measurement_noise_pos=cfg.tracker.measurement_noise_pos,
        measurement_noise_size=cfg.tracker.measurement_noise_size,
        gate_threshold=cfg.tracker.gate_threshold,
        min_hits_to_confirm=cfg.tracker.min_hits_to_confirm,
        max_coast_frames=cfg.tracker.max_coast_frames,
        max_coast_time=cfg.tracker.max_coast_time,
        min_bbox_size=cfg.tracker.min_bbox_size,
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
    )

    lock_eval = LockEvaluator(
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
        roi_width=cfg.detector.roi_width,
        roi_height=cfg.detector.roi_height,
        target_area_margin_x=cfg.lock.target_area_margin_x,
        target_area_margin_y=cfg.lock.target_area_margin_y,
        lock_rect_padding=cfg.lock.lock_rect_padding,
        min_target_size_ratio=cfg.lock.min_target_size_ratio,
        min_coverage_ratio=cfg.lock.min_coverage_ratio,
        max_center_offset_ratio=cfg.lock.max_center_offset_ratio,
        required_lock_duration=cfg.lock.required_lock_duration,
        lock_window_duration=cfg.lock.lock_window_duration,
        hold_grace_time=cfg.lock.hold_grace_time,
        bbox_scale_factor=getattr(cfg.lock, "bbox_scale_factor", 0.80),
    )

    servo = VisualServo(
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
        guidance_mode=str(getattr(cfg.control, "guidance_mode", "PN")),
        pn_gain=float(getattr(cfg.control, "pn_gain", 3.2)),
        horizontal_fov_deg=cfg.camera.horizontal_fov_deg,
        vertical_fov_deg=cfg.camera.vertical_fov_deg,
    )

    fsm = MissionFSM(
        detection_confirm_time=cfg.mission.detection_confirm_time,
        intercept_timeout=cfg.mission.intercept_timeout,
        reacquire_timeout=cfg.mission.reacquire_timeout,
        track_loss_grace=cfg.mission.track_loss_grace,
    )
    # Kalkış tamamlandı varsayalım
    fsm.trigger_takeoff()
    fsm.trigger_takeoff_complete()

    hud = HUD(
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
        target_area=lock_eval.target_area,
    )

    # Video Aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Video açılamadı: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 25.0
    frame_delay = max(int(1000.0 / fps), 1)

    start_sim_time = time.time()
    frame_idx = 0
    paused = False

    print("▶️ Canlı Görüntü Ekranı Açılıyor... (Çıkış: 'q' | Duraklat: 'BOŞLUK')")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Kare numarası/simülasyon zamanı geriye sarmadan önce durumu yenile.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                tracker.reset()
                lock_eval.reset()
                reacq.reset()
                servo.reset()
                fsm = MissionFSM(
                    detection_confirm_time=cfg.mission.detection_confirm_time,
                    intercept_timeout=cfg.mission.intercept_timeout,
                    reacquire_timeout=cfg.mission.reacquire_timeout,
                    track_loss_grace=cfg.mission.track_loss_grace,
                )
                fsm.trigger_takeoff()
                fsm.trigger_takeoff_complete()
                frame_idx = 0
                start_sim_time = time.time()
                continue

            frame_idx += 1
            now = start_sim_time + (frame_idx / fps)

            # 1. ROI Arama ve YOLOv11 Tespiti
            roi = None
            if tracker.is_active:
                roi = tracker.get_predicted_roi(dt_ahead=0.10, expand=1.8)

            detections = detector.detect_with_roi_fallback(
                frame, predicted_roi=roi, roi_imgsz=cfg.detector.roi_imgsz
            )
            best_det = detections[0] if detections else None
            has_detection = best_det is not None

            # 2. Kalman Filtresi ile Takip (EKF)
            track_result = tracker.update(detections, now)
            effective_bbox = None
            if track_result.state in (TrackState.CONFIRMED, TrackState.COASTING):
                effective_bbox = track_result.bbox
            elif best_det is not None:
                effective_bbox = np.array(best_det.bbox, dtype=float)

            # 3. Şartname Kilitlenme Değerlendirmesi
            lock_status = lock_eval.evaluate(effective_bbox, now)

            # 4. FSM Durum Güncellemesi
            fsm.update(
                track_state=track_result.state,
                has_detection=has_detection,
                lock_progress=lock_status.lock_progress,
                is_locked=lock_status.is_locked,
                now=now,
            )

            # 5. Görsel Servo Komutu (PN vs Pure Pursuit)
            if effective_bbox is not None:
                cx = (effective_bbox[0] + effective_bbox[2]) / 2.0
                cy = (effective_bbox[1] + effective_bbox[3]) / 2.0
                bw = effective_bbox[2] - effective_bbox[0]
                servo_cmd = servo.compute(
                    target_cx=cx,
                    target_cy=cy,
                    target_vx=track_result.velocity[0],
                    target_vy=track_result.velocity[1],
                    bbox_width=bw,
                    now=now,
                )

            # 6. HUD Görselleştirme
            display_bbox = effective_bbox
            raw_bbox_display = np.array(best_det.bbox, dtype=float) if best_det is not None else None

            hud_frame = hud.draw(
                frame=frame,
                mission_state=fsm.state,
                lock_status=lock_status,
                track_bbox=display_bbox,
                raw_bbox=raw_bbox_display,
                track_confidence=track_result.confidence,
                fps=fps,
                mission_elapsed=frame_idx / fps,
                vehicle_mode="FBWA (Fixed-Wing)",
                altitude_m=100.0,
                airspeed_mps=16.5,
                groundspeed_mps=15.8,
                guidance_mode=servo.guidance_mode,
                prey_behavior="EVADE_ZIGZAG" if lock_status.lock_progress > 0.3 else "PATROL",
            )

            # Ekranda göster
            cv2.imshow("Savasan IHA - Otonom Kilitlenme Simulatoru", hud_frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("⏹️ Simülasyon durduruldu.")


if __name__ == "__main__":
    run_demo()

