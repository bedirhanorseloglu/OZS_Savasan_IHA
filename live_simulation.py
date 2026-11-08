"""
Savaşan İHA - Canlı Kapalı Döngü 3D Uçuş ve Hava Muharebesi Simülatörü
(Live Closed-Loop Synthetic Flight & Combat Simulator)

===============================================================================
🧑‍🏫 BU SİMÜLATÖR NEDİR VE NASIL ÇALIŞIR?
===============================================================================
Bu simülatörde HİÇBİR VİDEO KAYDI KULLANILMAZ!
Her şey anlık olarak, canlı fizik motoru ve 3D kamera projeksiyonu ile üretilir:

1. CANLI 3D UZAY (Fizik Motoru):
   - Av İHA ve Avcı İHA gökyüzünde 3 boyutlu koordinatlarda (X, Y, Z) uçarlar.
   - Av İHA, Akıllı Kaçış Algoritmasıyla (Evader AI) zikzaklar, ani irtifa değişimleri
     ve virajlar yapar (İsterseniz klavyeyle siz de kaçırabilirsiniz!).

2. CANLI KAMERA (3D -> 2D Pinhole Projeksiyonu):
   - Avcı İHA'nın burnunda sanal bir kamera vardır (FOV: 110° x 75°).
   - Av İHA'nın 3D konumu, avcının kamerasına canlı olarak iz düşürülür (Perspective Projection).
   - Anlık olarak gerçek bir kamera görüntüsü (frame) render edilir.

3. KAPALI DÖNGÜ KONTROL (Closed-Loop Control):
   - Bu canlı görüntü YOLO / EKF Tracker'a verilir.
   - Oransal Seyrüsefer (PN) kontrolcümüz anlık yönelme komutu (yaw_rate, Vz) üretir.
   - Bu komut Avcı İHA'nın fizik motoruna uygulanır ve Avcı İHA canlı olarak hedefin
     peşinden uçar!

KLAVYE KONTROLLERİ:
- [A] / [D] : Av İHA'yı canlı olarak Sağa / Sola kaçırın!
- [W] / [S] : Av İHA'yı canlı olarak Tırmandırın / Daldırın!
- [SPACE]   : Otonom Kaçış Zekasını (Evader AI) Aç / Kapat!
- [Q]       : Çıkış
===============================================================================
"""

from __future__ import annotations

import math
import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVASAN_DIR = os.path.join(PROJECT_ROOT, "Savaşan_İHA_Görevi")
sys.path.insert(0, SAVASAN_DIR)

from core.config import Config
from core.logger import init_logger
from perception.tracker import SingleTargetTracker, TrackState
from perception.lock_evaluator import LockEvaluator
from control.visual_servo import VisualServo
from mission.fsm import MissionFSM, MissionState
from iol.hud import HUD


class SimulatedUAV:
    """3 Boyutlu Kinematik İHA Modeli"""
    def __init__(self, x=0.0, y=0.0, z=100.0, yaw=0.0, speed=16.0, name="UAV"):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)  # İrtifa (metre)
        self.yaw = float(yaw)  # Radyan
        self.pitch = 0.0
        self.roll = 0.0
        self.speed = float(speed)  # m/s
        self.vz = 0.0  # m/s dikey hız
        self.name = name

    def update(self, dt: float, yaw_rate_deg: float = 0.0, vz_cmd: float = 0.0, speed_cmd: float = None):
        if speed_cmd is not None:
            self.speed = np.clip(speed_cmd, 10.0, 24.0)

        # Yaw güncelleme
        yaw_rate_rad = math.radians(np.clip(yaw_rate_deg, -55.0, 55.0))
        self.yaw += yaw_rate_rad * dt

        # Dikey hız güncelleme
        self.vz = np.clip(vz_cmd, -5.0, 5.0)
        self.z += self.vz * dt
        self.z = np.clip(self.z, 20.0, 200.0)

        # Konum güncelleme (Kuzey-Doğu düzlemi)
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt


class SyntheticCameraSimulator:
    """3D Dünyayı Avcı Kamerasının Gözünden Canlı Render Eden Motor"""
    def __init__(self, width=1280, height=720, fov_h_deg=110.0, fov_v_deg=75.0):
        self.w = width
        self.h = height
        self.fov_h = math.radians(fov_h_deg)
        self.fov_v = math.radians(fov_v_deg)
        self.focal_x = (width / 2.0) / math.tan(self.fov_h / 2.0)
        self.focal_y = (height / 2.0) / math.tan(self.fov_v / 2.0)
        self.cx = width / 2.0
        self.cy = height / 2.0

    def render_view(self, hunter: SimulatedUAV, prey: SimulatedUAV) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
        """
        Avcının gözünden gökyüzünü ve Av İHA'yı canlı çizer.
        Dönen: (kamera_goruntusu, gercek_bbox)
        """
        # Gökyüzü arka planı (Gradient mavi/ufuk çizgisi)
        frame = np.full((self.h, self.w, 3), (215, 185, 140), dtype=np.uint8)  # Açık mavi gökyüzü

        # Ufuk çizgisi ve yer düzlemi
        horizon_y = int(self.cy + math.tan(hunter.pitch) * self.focal_y)
        if 0 <= horizon_y < self.h:
            cv2.rectangle(frame, (0, horizon_y), (self.w, self.h), (90, 130, 90), -1)  # Yeşilimsi zemin
            cv2.line(frame, (0, horizon_y), (self.w, horizon_y), (140, 160, 140), 2)

        # 3D Bağıl Konum (Hunter Body Frame)
        dx = prey.x - hunter.x
        dy = prey.y - hunter.y
        dz = prey.z - hunter.z  # Yukarı pozitif

        # Avcının heading açısına göre döndür (Body Frame)
        cos_y = math.cos(-hunter.yaw)
        sin_y = math.sin(-hunter.yaw)
        forward = dx * cos_y - dy * sin_y
        right = dx * sin_y + dy * cos_y
        up = dz

        bbox = None

        # Eğer hedef kameranın önündeyse (forward > 1m)
        if forward > 1.5:
            # Pinhole Kamera Perspektif İzdüşümü
            px = int(self.cx + (right / forward) * self.focal_x)
            py = int(self.cy - (up / forward) * self.focal_y)

            # Hedef ekran sınırları içindeyse
            if -100 <= px <= self.w + 100 and -100 <= py <= self.h + 100:
                # İHA'nın fiziksel boyutu (~1.8m kanat açıklığı)
                apparent_w = int((1.8 / forward) * self.focal_x)
                apparent_h = int((0.8 / forward) * self.focal_y)
                apparent_w = max(apparent_w, 8)
                apparent_h = max(apparent_h, 6)

                x1 = int(px - apparent_w / 2)
                y1 = int(py - apparent_h / 2)
                x2 = int(px + apparent_w / 2)
                y2 = int(py + apparent_h / 2)

                # Av İHA Silueti Çiz (Gerçekçi VTOL / Uçak gövdesi)
                self._draw_uav_model(frame, px, py, apparent_w, apparent_h, prey.yaw - hunter.yaw)

                # Ekran sınırlarına kırp
                bx1 = max(0, x1)
                by1 = max(0, y1)
                bx2 = min(self.w, x2)
                by2 = min(self.h, y2)
                if bx2 > bx1 and by2 > by1:
                    bbox = (bx1, by1, bx2, by2)

        return frame, bbox

    def _draw_uav_model(self, frame, cx, cy, w, h, rel_yaw):
        """Kamera görüntüsü üzerine gerçekçi bir VTOL İHA gövdesi ve kanatları çizer."""
        color_body = (50, 50, 50)     # Koyu gri gövde
        color_wing = (70, 70, 70)     # Kanatlar
        color_prop = (200, 50, 50)    # Kırmızı motor uçları

        # Gövde (elips/çizgi)
        cv2.ellipse(frame, (cx, cy), (max(w // 2, 4), max(h // 3, 2)), 0, 0, 360, color_body, -1)
        # Ana Kanat
        wing_span = int(w * 0.9)
        cv2.line(frame, (cx - wing_span // 2, cy), (cx + wing_span // 2, cy), color_wing, max(h // 4, 2))
        # Dikey Kuyruk
        cv2.line(frame, (cx, cy), (cx, cy - max(h // 2, 4)), color_body, max(w // 10, 2))
        # Motor Pervane Daireleri (VTOL Quad)
        m_offset_x = wing_span // 3
        m_offset_y = max(h // 3, 3)
        for dx in (-m_offset_x, m_offset_x):
            for dy in (-m_offset_y, m_offset_y):
                cv2.circle(frame, (cx + dx, cy + dy), max(w // 14, 2), color_prop, 1)


def run_live_simulation():
    """Canlı Kapalı Döngü Simülasyonu Başlatır"""
    init_logger(console=True)
    cfg_path = os.path.join(SAVASAN_DIR, "config", "default.yaml")
    cfg = Config.from_yaml(cfg_path)

    print("=" * 70)
    print("🚀 CANLI VE İNTERAKTİF SAVAŞAN İHA SİMÜLATÖRÜ BAŞLATILIYOR")
    print("🎮 Kontroller: [A/D] Sağa/Sola Kaçır | [W/S] Tırman/Dal | [SPACE] Oto-Kaçış | [Q] Çık")
    print("=" * 70)

    # 1. 3D Uçakları Başlat (Başlangıçta avcı hedefin arkasında)
    hunter = SimulatedUAV(x=0.0, y=0.0, z=100.0, yaw=0.0, speed=17.0, name="AVCI")
    prey = SimulatedUAV(x=85.0, y=10.0, z=100.0, yaw=0.0, speed=15.0, name="AV")

    camera_sim = SyntheticCameraSimulator(
        width=cfg.camera.frame_width,
        height=cfg.camera.frame_height,
        fov_h_deg=cfg.camera.horizontal_fov_deg,
        fov_v_deg=cfg.camera.vertical_fov_deg,
    )

    # 2. Otonom Algoritma Katmanları
    tracker = SingleTargetTracker(
        gate_threshold=cfg.tracker.gate_threshold,
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
    )

    lock_eval = LockEvaluator(
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
        target_area_margin_x=cfg.lock.target_area_margin_x,
        target_area_margin_y=cfg.lock.target_area_margin_y,
        min_target_size_ratio=cfg.lock.min_target_size_ratio,
        min_coverage_ratio=cfg.lock.min_coverage_ratio,
        max_center_offset_ratio=cfg.lock.max_center_offset_ratio,
        required_lock_duration=cfg.lock.required_lock_duration,
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

    fsm = MissionFSM(detection_confirm_time=0.2)
    fsm.trigger_takeoff()
    fsm.trigger_takeoff_complete()

    hud = HUD(
        frame_width=cfg.camera.frame_width,
        frame_height=cfg.camera.frame_height,
        target_area=lock_eval.target_area,
    )

    # Simülasyon Döngüsü (30 FPS)
    dt = 0.033
    auto_evade = True
    evade_timer = 0.0
    evade_turn_rate = 0.0
    evade_vz = 0.0
    prey_mode = "PATROL"
    prey_evasion_started_at = None
    prey_evasion_phase = -1
    prey_last_evasion_switch = 0.0
    prey_target_heading = prey.yaw
    prey_target_altitude = prey.z
    prey_patrol_center = (prey.x, prey.y + 80.0)
    search_scan_direction = 1.0
    search_scan_switch_at = 0.0

    cv2.namedWindow("Savasan IHA - CANLI 3D KAPALI DONOGU SIMULATORU", cv2.WINDOW_AUTOSIZE)
    sim_time = 0.0

    while True:
        sim_time += dt

        # -------------------------------------------------------------
        # 1. AV İHA HAREKETİ (Otonom Akıllı Kaçış veya Klavye)
        # -------------------------------------------------------------
        evade_turn_rate = 0.0
        evade_vz = 0.0
        prey_speed_cmd = 15.0
        if auto_evade:
            evade_timer += dt

            # Mirror the project SmartPreyPilot: threat must be close and in
            # the prey's rear cone. Then run 3-second S-turn / altitude / break
            # phases at the configured 20 m/s evasion speed.
            hunter_dx = hunter.x - prey.x
            hunter_dy = hunter.y - prey.y
            threat_distance = math.sqrt(
                hunter_dx * hunter_dx + hunter_dy * hunter_dy + (hunter.z - prey.z) ** 2
            )
            threat_bearing = math.atan2(hunter_dy, hunter_dx)
            threat_angle = abs((threat_bearing - prey.yaw + math.pi) % (2.0 * math.pi) - math.pi)
            rear_threat = threat_angle > math.pi - math.radians(70.0)
            threat_detected = threat_distance <= 85.0 and rear_threat

            if threat_detected and prey_evasion_started_at is None:
                prey_evasion_started_at = evade_timer
                prey_evasion_phase = -1
                prey_last_evasion_switch = evade_timer - 3.0

            if (
                prey_evasion_started_at is not None
                and not threat_detected
                and evade_timer - prey_evasion_started_at > 12.0
            ):
                prey_evasion_started_at = None
                prey_evasion_phase = -1
                prey_mode = "PATROL"

            if prey_evasion_started_at is not None:
                # In this synthetic pursuit scene the hunter is capped at 16 m/s.
                # Keep the prey maneuvering while allowing a physically possible intercept.
                prey_speed_cmd = 14.0
                if evade_timer - prey_last_evasion_switch >= 3.0:
                    prey_evasion_phase = (prey_evasion_phase + 1) % 4
                    prey_last_evasion_switch = evade_timer
                    turn_offsets_deg = (45.0, 0.0, -50.0, 85.0)
                    prey_target_heading = prey.yaw + math.radians(turn_offsets_deg[prey_evasion_phase])
                    previous_mode = prey_mode

                    if prey_evasion_phase == 0:
                        prey_mode = "EVADE_ZIGZAG"
                        prey_target_altitude = 100.0
                    elif prey_evasion_phase == 1:
                        prey_mode = "EVADE_ALT"
                        prey_target_altitude = 80.0 if prey_target_altitude >= 95.0 else 125.0
                    elif prey_evasion_phase == 2:
                        prey_mode = "EVADE_ZIGZAG"
                    else:
                        prey_mode = "BREAK_TURN"

                    if prey_mode != previous_mode:
                        print(f"🛩️ Av manevrası: {prey_mode} (tehdit {threat_distance:.0f} m)")

                heading_error = (prey_target_heading - prey.yaw + math.pi) % (2.0 * math.pi) - math.pi
                evade_turn_rate = float(np.clip(math.degrees(heading_error) * 1.8, -50.0, 50.0))
                evade_vz = float(np.clip((prey_target_altitude - prey.z) * 0.45, -4.0, 4.0))
            else:
                # Native LOITER approximation: stable ~80 m radius patrol.
                center_bearing = math.atan2(prey_patrol_center[1] - prey.y, prey_patrol_center[0] - prey.x)
                patrol_heading = center_bearing - math.pi / 2.0
                heading_error = (patrol_heading - prey.yaw + math.pi) % (2.0 * math.pi) - math.pi
                evade_turn_rate = float(np.clip(math.degrees(heading_error) * 1.2, -25.0, 25.0))

        prey.update(
            dt,
            yaw_rate_deg=evade_turn_rate,
            vz_cmd=evade_vz,
            speed_cmd=prey_speed_cmd,
        )

        # -------------------------------------------------------------
        # 2. CANLI KAMERA GÖRÜNTÜSÜ OLUŞTURMA (3D -> 2D Rendering)
        # -------------------------------------------------------------
        live_frame, true_bbox = camera_sim.render_view(hunter, prey)

        # -------------------------------------------------------------
        # 3. YAPAY ZEKA VE KALMAN TAKİPÇİSİ (SENSE & ESTIMATE)
        # -------------------------------------------------------------
        detections = []
        best_det_bbox = None
        if true_bbox is not None:
            # Kameranın tespitine hafif sensör gürültüsü ekle
            noise_x = int(np.random.normal(0, 1.2))
            noise_y = int(np.random.normal(0, 1.2))
            det_b = (
                max(0, true_bbox[0] + noise_x),
                max(0, true_bbox[1] + noise_y),
                min(cfg.camera.frame_width, true_bbox[2] + noise_x),
                min(cfg.camera.frame_height, true_bbox[3] + noise_y),
            )
            # YOLO tespit formatı mock objesi
            from collections import namedtuple
            DetMock = namedtuple("DetMock", ["bbox", "confidence", "cls_id"])
            detections.append(DetMock(bbox=det_b, confidence=0.88, cls_id=1))
            best_det_bbox = np.array(det_b, dtype=float)

        now = sim_time
        track_result = tracker.update(detections, now)
        has_detection = len(detections) > 0

        effective_bbox = None
        if track_result.state in (TrackState.CONFIRMED, TrackState.COASTING):
            effective_bbox = track_result.bbox
        elif best_det_bbox is not None:
            effective_bbox = best_det_bbox

        # -------------------------------------------------------------
        # 4. ŞARTNAME KİLİTLENME VE FSM GÜNCELLEMESİ (DECIDE)
        # -------------------------------------------------------------
        # Sentetik kamerada gerçek bbox elimizde olduğundan kilidi filtrelenmiş
        # kutuyla değil güncel ölçümle değerlendir. Aksi halde EKF'nin yavaş
        # küçülen kutusu, uzaklaşan hedefte kilidi gerçekte olduğundan uzun tutar.
        lock_status = lock_eval.evaluate(best_det_bbox, now)
        fsm.update(
            track_state=track_result.state,
            has_detection=has_detection,
            lock_progress=lock_status.lock_progress,
            is_locked=lock_status.is_locked,
            now=now,
        )

        # -------------------------------------------------------------
        # 5. AVCI KONTROL VE HAREKET (ACT - Kapalı Döngü Otopilot)
        # -------------------------------------------------------------
        # Hedef ile avcı arasındaki gerçek 3D açı ve mesafe (Radar / Global Kestirim)
        dx_global = prey.x - hunter.x
        dy_global = prey.y - hunter.y
        dz_global = prey.z - hunter.z
        rel_3d_dist = math.sqrt(dx_global**2 + dy_global**2 + dz_global**2)
        prey_bearing = math.atan2(dy_global, dx_global)
        bearing_diff_deg = math.degrees(
            (prey_bearing - hunter.yaw + math.pi) % (2.0 * math.pi) - math.pi
        )

        hunter_yaw_rate = 0.0
        hunter_vz = 0.0
        hunter_speed = 16.0

        if effective_bbox is not None:
            # A) HEDEF KADRAJDA: Hassas Görsel Servo & Oransal Seyrüsefer (PN)
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
            hunter_yaw_rate = servo_cmd.yaw_rate
            # VisualServo.vz, araç katmanındaki NED işaretini kullanır
            # (negatif tırmanış); sentetik modelde z yukarı pozitif olduğundan
            # simülasyon eksenine dönüştür.
            hunter_vz = -servo_cmd.vz

            # Gerçek mission controller'ın 16 m/s pursuit limitine uy.
            if bw < cfg.camera.frame_width * 0.045:
                hunter_speed = 16.0
            elif bw > cfg.camera.frame_width * 0.080:
                hunter_speed = 14.5
            else:
                hunter_speed = 15.0
        else:
            # Gerçek uçuşta hedef kaybolunca GPS'teki gerçek prey konumuna
            # dönülmez; hunter arama/yaw-scan yapar. 4 saniyede yön değiştir,
            # her yarım döngüde 2 saniye 45°/s tara.
            if now >= search_scan_switch_at:
                search_scan_direction *= -1.0
                search_scan_switch_at = now + 2.0
            hunter_yaw_rate = 45.0 * search_scan_direction
            hunter_vz = 0.0
            hunter_speed = float(getattr(cfg.mission, "search_speed", 12.0))

        # Kilit koşulları başladığında yaklaşmayı bırakıp hedef hızını eşleştir.
        # Böylece 4 saniyelik doğrulama sırasında simülasyon hedefin içinden geçmez.
        if lock_status.lock_progress > 0.0 or lock_status.is_locked:
            hunter_speed = prey.speed

        # Avcı İHA'nın fizik motorunu canlı güncelle (Kapalı döngü!)
        hunter.update(dt, yaw_rate_deg=hunter_yaw_rate, vz_cmd=hunter_vz, speed_cmd=hunter_speed)

        # -------------------------------------------------------------
        # 6. CANLI HUD VE TELEMETRİ GÖRSELLEŞTİRME
        # -------------------------------------------------------------
        # Gerçek 3D mesafe hesabı
        rel_3d_dist = math.sqrt((prey.x - hunter.x)**2 + (prey.y - hunter.y)**2 + (prey.z - hunter.z)**2)
        hud_frame = hud.draw(
            frame=live_frame,
            mission_state=fsm.state,
            lock_status=lock_status,
            track_bbox=effective_bbox,
            raw_bbox=best_det_bbox,
            track_confidence=track_result.confidence,
            fps=1.0 / dt,
            mission_elapsed=evade_timer,
            vehicle_mode="FBWA (Canli Fizik)",
            altitude_m=hunter.z,
            airspeed_mps=hunter.speed,
            groundspeed_mps=hunter.speed,
            guidance_mode=servo.guidance_mode,
            prey_behavior=prey_mode if auto_evade else "MANUEL_KLAVYE",
        )

        # Alt bilgi şeridi
        info_text = f"SIM | 3D: {rel_3d_dist:.1f}m | Oto-Kacis: {'ACIK' if auto_evade else 'KAPALI'}"
        cv2.putText(hud_frame, info_text, (480, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 6.B HEDEF KADRAJ DIŞINDAYSA: YÖN VE RADAR GÖSTERGESİ
        # -------------------------------------------------------------
        if effective_bbox is None:
            if bearing_diff_deg < 0:
                radar_msg = f"<< HEDEF SOLDA ({abs(bearing_diff_deg):.0f} deg | {rel_3d_dist:.0f}m) - OTOPILOT YONELIYOR"
                rx = 40
            else:
                radar_msg = f"OTOPILOT YONELIYOR - HEDEF SAGDA ({bearing_diff_deg:.0f} deg | {rel_3d_dist:.0f}m) >>"
                rx = cfg.camera.frame_width - 620

            cv2.rectangle(hud_frame, (rx - 10, 220), (rx + 600, 265), (0, 0, 180), -1)
            cv2.rectangle(hud_frame, (rx - 10, 220), (rx + 600, 265), (0, 255, 255), 2)
            cv2.putText(hud_frame, radar_msg, (rx + 6, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

        # Kamera merkezindeki nişangahın yanına da "BİZİM NİŞANGAH" etiketi yaz
        frame_cx = cfg.camera.frame_width // 2
        frame_cy = cfg.camera.frame_height // 2
        cv2.putText(hud_frame, "BİZİM NİŞANGAH", (frame_cx - 55, frame_cy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Savasan IHA - CANLI 3D KAPALI DONOGU SIMULATORU", hud_frame)

        # Klavye Kontrolleri
        key = cv2.waitKey(int(dt * 1000)) & 0xFF
        if key == ord("q"):
            break
        if lock_status.just_locked:
            print("✓ Kilit tamamlandi; basarili durum ekranda donduruldu. Cikis: [Q]")
            while (cv2.waitKey(30) & 0xFF) != ord("q"):
                cv2.imshow("Savasan IHA - CANLI 3D KAPALI DONOGU SIMULATORU", hud_frame)
            break
        elif key == ord(" "):
            auto_evade = not auto_evade
            evade_turn_rate = 0.0
            evade_vz = 0.0
            print(f"🔄 Otonom Kaçış Modu: {'AÇIK' if auto_evade else 'KAPALI'}")
        elif key == ord("a"):
            auto_evade = False
            evade_turn_rate = -35.0  # Sola sert kaçış
        elif key == ord("d"):
            auto_evade = False
            evade_turn_rate = +35.0  # Sağa sert kaçış
        elif key == ord("w"):
            auto_evade = False
            evade_vz = +3.5          # Yukarı tırman
        elif key == ord("s"):
            auto_evade = False
            evade_vz = -3.5          # Aşağı dal
        elif key == ord("r"):
            # Konumları sıfırla: Avcıyı hedefin tam arkasına yerleştir
            hunter.x, hunter.y, hunter.z, hunter.yaw = 0.0, 0.0, 100.0, 0.0
            prey.x, prey.y, prey.z, prey.yaw = 70.0, 0.0, 100.0, 0.0
            tracker.reset()
            lock_eval.reset()
            print("🔄 Konumlar Sıfırlandı: Avcı hedefin tam arkasına yerleşti!")

    cv2.destroyAllWindows()
    print("⏹️ Canlı simülasyon sonlandırıldı.")


if __name__ == "__main__":
    run_live_simulation()

