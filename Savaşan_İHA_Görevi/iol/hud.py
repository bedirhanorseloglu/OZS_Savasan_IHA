"""
Savasan IHA - HUD Overlay

Kamera goruntusu uzerine sartnameye uygun ve rapora eklenebilir
okunur bir HUD katmani cizer.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from mission.fsm import MissionState
from perception.lock_evaluator import LockStatus


class HUD:
    """Şartnameyle uyumlu HUD katmanı (kamera görüntüsü üzerine bilgi çizdirme)."""

    LOCK_COLOR = (0, 0, 255)          # #FF0000
    TARGET_AREA_COLOR = (0, 215, 255) # sari
    TRACK_COLOR = (80, 255, 80)       # yesil
    PANEL_BG = (28, 28, 28)
    PANEL_ALPHA = 0.62
    TEXT = (245, 245, 245)
    MUTED = (190, 190, 190)
    OK = (70, 220, 70)
    WARN = (0, 190, 255)
    FAIL = (70, 70, 255)
    GUIDE = (255, 220, 90)
    CENTER = (255, 255, 255)

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        target_area: Tuple[int, int, int, int] = (320, 72, 960, 648),
    ):
        self._fw = frame_width
        self._fh = frame_height
        self._target_area = target_area

    def draw(
        self,
        frame: np.ndarray,
        mission_state: MissionState,
        lock_status: LockStatus,
        track_bbox: Optional[np.ndarray] = None,
        raw_bbox: Optional[np.ndarray] = None,
        track_confidence: float = 0.0,
        server_time: Optional[float] = None,
        fps: float = 0.0,
        mission_elapsed: float = 0.0,
        mission_failed: bool = False,
        vehicle_mode: str = "",
        altitude_m: float = 0.0,
        airspeed_mps: float = 0.0,
        groundspeed_mps: float = 0.0,
        guidance_mode: str = "PN",
        prey_behavior: str = "PATROL",
    ) -> np.ndarray:
        """HUD ciz."""
        overlay = frame.copy()

        self._draw_target_area(overlay)
        self._draw_camera_center(overlay)
        self._draw_raw_bbox(overlay, raw_bbox)
        self._draw_track_bbox(overlay, track_bbox)
        self._draw_lock_rect(overlay, lock_status)
        self._draw_guidance_line(overlay, track_bbox)

        left_panel = overlay.copy()
        self._panel(left_panel, 12, 12, 380, 178)
        self._panel(left_panel, 12, self._fh - 178, 440, 178)

        right_panel = overlay.copy()
        self._panel(right_panel, self._fw - 354, 12, 342, 244)

        overlay = cv2.addWeighted(left_panel, self.PANEL_ALPHA, overlay, 1 - self.PANEL_ALPHA, 0)
        overlay = cv2.addWeighted(right_panel, self.PANEL_ALPHA, overlay, 1 - self.PANEL_ALPHA, 0)

        self._draw_header(
            overlay,
            mission_state=mission_state,
            server_time=server_time,
            mission_elapsed=mission_elapsed,
            mission_failed=mission_failed,
            fps=fps,
            vehicle_mode=vehicle_mode,
            altitude_m=altitude_m,
            airspeed_mps=airspeed_mps,
            groundspeed_mps=groundspeed_mps,
            guidance_mode=guidance_mode,
            prey_behavior=prey_behavior,
        )
        self._draw_metrics(
            overlay,
            lock_status=lock_status,
            track_confidence=track_confidence,
        )
        self._draw_criteria(overlay, lock_status)
        self._draw_legend(overlay, track_bbox)

        if lock_status.is_locked:
            h, w = overlay.shape[:2]
            cx, cy = w // 2, h // 2
            card_w, card_h = 520, 180
            cx1 = max(0, cx - card_w // 2)
            cy1 = max(0, cy - card_h // 2)
            cx2 = min(w, cx + card_w // 2)
            cy2 = min(h, cy + card_h // 2)

            sub_img = overlay[cy1:cy2, cx1:cx2]
            black_rect = np.zeros_like(sub_img)
            cv2.rectangle(black_rect, (0, 0), (cx2 - cx1, cy2 - cy1), (0, 45, 0), -1)
            cv2.addWeighted(sub_img, 0.20, black_rect, 0.80, 0, dst=sub_img)

            cv2.rectangle(overlay, (cx1, cy1), (cx2, cy2), self.OK, 3)

            text1 = "GOREV BASARILI"
            text2 = "Otonom Kilitlenme Saglandi!"
            text3 = f"Sure: {lock_status.lock_duration:.2f} sn"

            t1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)[0]
            t2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            t3_size = cv2.getTextSize(text3, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

            cv2.putText(overlay, text1, (cx - t1_size[0] // 2, cy1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.OK, 3, cv2.LINE_AA)
            cv2.putText(overlay, text2, (cx - t2_size[0] // 2, cy1 + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            cv2.putText(overlay, text3, (cx - t3_size[0] // 2, cy1 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2, cv2.LINE_AA)

        return overlay

    def _draw_target_area(self, frame: np.ndarray):
        ax1, ay1, ax2, ay2 = self._target_area
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), self.TARGET_AREA_COLOR, 2)
        cv2.putText(
            frame,
            "Hedef Vurus Alani",
            (ax1 + 8, ay1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.TARGET_AREA_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _draw_track_bbox(self, frame: np.ndarray, track_bbox: Optional[np.ndarray]):
        if track_bbox is None:
            return
        bx1, by1, bx2, by2 = track_bbox.astype(int)
        # Filtrelenmiş hedef kırmızı olacak (LOCK_COLOR) - kalınlık 3
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), self.LOCK_COLOR, 3)
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        cv2.circle(frame, (cx, cy), 4, self.LOCK_COLOR, -1)

    def _draw_raw_bbox(self, frame: np.ndarray, raw_bbox: Optional[np.ndarray]):
        if raw_bbox is None:
            return
        bx1, by1, bx2, by2 = raw_bbox.astype(int)
        # Ham hedef yeşil olacak (TRACK_COLOR) - kalınlık 2
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), self.TRACK_COLOR, 2)
        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2
        cv2.circle(frame, (cx, cy), 3, self.TRACK_COLOR, -1)

    def _draw_camera_center(self, frame: np.ndarray):
        cx = self._fw // 2
        cy = self._fh // 2
        cv2.drawMarker(
            frame,
            (cx, cy),
            self.CENTER,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(frame, (cx, cy), 16, self.CENTER, 1, cv2.LINE_AA)

    def _draw_guidance_line(self, frame: np.ndarray, track_bbox: Optional[np.ndarray]):
        if track_bbox is None:
            return
        bx1, by1, bx2, by2 = track_bbox.astype(int)
        target_cx = (bx1 + bx2) // 2
        target_cy = (by1 + by2) // 2
        frame_cx = self._fw // 2
        frame_cy = self._fh // 2
        cv2.line(frame, (frame_cx, frame_cy), (target_cx, target_cy), self.GUIDE, 2, cv2.LINE_AA)
        cv2.circle(frame, (target_cx, target_cy), 6, self.GUIDE, 1, cv2.LINE_AA)

    def _draw_lock_rect(self, frame: np.ndarray, lock_status: LockStatus):
        if lock_status.lock_rect is None:
            return
        lx1, ly1, lx2, ly2 = lock_status.lock_rect
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), self.LOCK_COLOR, 2)
        cv2.putText(
            frame,
            "Kilitlenme Dortgeni",
            (lx1, min(self._fh - 12, ly2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            self.LOCK_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _draw_header(
        self,
        frame: np.ndarray,
        mission_state: MissionState,
        server_time: Optional[float],
        mission_elapsed: float,
        mission_failed: bool,
        fps: float,
        vehicle_mode: str,
        altitude_m: float,
        airspeed_mps: float,
        groundspeed_mps: float,
        guidance_mode: str = "PN",
        prey_behavior: str = "PATROL",
    ):
        state_label = self._state_label(mission_state)
        state_color = self._state_color(mission_state)

        cv2.putText(frame, "Otonom Kilitlenme Durumu", (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.TEXT, 2, cv2.LINE_AA)
        cv2.putText(frame, state_label, (24, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, state_color, 2, cv2.LINE_AA)

        elapsed_text = "Gorev Basarisiz" if mission_failed else f"Gorev Suresi: {int(mission_elapsed)} sn"
        cv2.putText(frame, elapsed_text, (24, 102),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    self.FAIL if mission_failed else self.MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mod: {(vehicle_mode or '-').split()[0]}", (24, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Irtifa: {altitude_m:.1f} m", (210, 126),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hava: {airspeed_mps:.1f} m/s", (24, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yer: {groundspeed_mps:.1f} m/s", (210, 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.MUTED, 1, cv2.LINE_AA)
        guidance_text = f"Gudum: {guidance_mode}"
        prey_text = f"Hedef: {prey_behavior}"
        cv2.putText(frame, guidance_text, (24, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.GUIDE, 1, cv2.LINE_AA)
        cv2.putText(frame, prey_text, (210, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.WARN if "EVADE" in prey_behavior or "BREAK" in prey_behavior else self.MUTED, 1, cv2.LINE_AA)

        ts = server_time or time.time()
        clock_text = self._format_clock(ts)
        right_x = self._fw - 336
        cv2.putText(frame, "Sunucu Saati", (right_x, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, self.MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, clock_text, (right_x, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.86, self.TEXT, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}", (self._fw - 92, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.MUTED, 1, cv2.LINE_AA)

    def _draw_metrics(
        self,
        frame: np.ndarray,
        lock_status: LockStatus,
        track_confidence: float,
    ):
        x0 = self._fw - 336
        y = 98
        lines = [
            ("Takip Guveni", f"{track_confidence:.2f}"),
            ("Kilitlenme Suresi", f"{lock_status.lock_duration:.2f} / 4.00 sn"),
            ("Hedef Boyutu", f"%{lock_status.target_size_ratio * 100:.2f}"),
            ("Kapsama", f"%{lock_status.coverage_ratio * 100:.1f}"),
            (
                "Merkez Sapmasi",
                f"dx={lock_status.center_offset_px[0]:.1f}px  dy={lock_status.center_offset_px[1]:.1f}px",
            ),
            (
                "Hedef Merkezi",
                f"x={lock_status.target_center_px[0]:.0f}  y={lock_status.target_center_px[1]:.0f}",
            ),
            (
                "Kilit Merkezi",
                f"x={lock_status.lock_center_px[0]:.0f}  y={lock_status.lock_center_px[1]:.0f}",
            ),
        ]

        for idx, (label, value) in enumerate(lines):
            yy = y + idx * 18
            cv2.putText(frame, label, (x0, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, self.MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, value, (x0 + 144, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.TEXT, 1, cv2.LINE_AA)

        bar_x = x0
        bar_y = 232
        bar_w = 294
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (60, 60, 60), -1)
        fill_w = int(bar_w * min(lock_status.lock_progress, 1.0))
        fill_color = self.OK if lock_status.is_locked else self.WARN
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 12), fill_color, -1)
        cv2.putText(frame, "Kilitlenme Ilerlemesi", (bar_x, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.MUTED, 1, cv2.LINE_AA)

    def _draw_criteria(self, frame: np.ndarray, lock_status: LockStatus):
        x0 = 24
        y0 = self._fh - 150
        criteria = [
            ("Boyut >= %5", lock_status.target_size_ok),
            ("Hedef merkezde", lock_status.in_target_area),
            ("Kilit merkezi merkezde", lock_status.in_lock_zone),
            ("Kapsama >= %90", lock_status.coverage_ok),
            ("Merkez sapmasi uygun", lock_status.center_offset_ok),
            ("Tum sartlar saglandi", lock_status.all_criteria_met),
        ]

        cv2.putText(frame, "Sartname Kontrol Listesi", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.66, self.TEXT, 2, cv2.LINE_AA)

        for idx, (label, ok) in enumerate(criteria):
            yy = y0 + 28 + idx * 22
            color = self.OK if ok else self.FAIL
            prefix = "EVET" if ok else "HAYIR"
            cv2.putText(frame, prefix, (x0, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)
            cv2.putText(frame, label, (x0 + 66, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.TEXT, 1, cv2.LINE_AA)

    def _panel(self, frame: np.ndarray, x: int, y: int, w: int, h: int):
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.PANEL_BG, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 1)

    @staticmethod
    def _format_clock(ts: float) -> str:
        time_str = time.strftime("%H:%M:%S", time.localtime(ts))
        ms = int((ts % 1) * 1000)
        return f"{time_str}.{ms:03d}"

    @staticmethod
    def _state_label(state: MissionState) -> str:
        labels = {
            MissionState.INIT: "Hazirlaniyor",
            MissionState.TAKEOFF: "Dikey Kalkis",
            MissionState.SEARCH: "Hedef Araniyor",
            MissionState.DETECTED: "Hedef Tespit Edildi",
            MissionState.INTERCEPT: "Hedefe Yonelim",
            MissionState.TRACK: "Aktif Takip",
            MissionState.LOCK_HOLD: "Kilit Koruma",
            MissionState.LOCKED: "Kilitlenme Basarili",
            MissionState.REACQUIRE: "Hedef Yeniden Araniyor",
            MissionState.LAND: "Inis",
        }
        return labels.get(state, state.name)

    def _state_color(self, state: MissionState):
        colors = {
            MissionState.INIT: self.MUTED,
            MissionState.TAKEOFF: self.WARN,
            MissionState.SEARCH: self.WARN,
            MissionState.DETECTED: (0, 220, 255),
            MissionState.INTERCEPT: (0, 255, 255),
            MissionState.TRACK: self.OK,
            MissionState.LOCK_HOLD: self.OK,
            MissionState.LOCKED: self.OK,
            MissionState.REACQUIRE: (110, 170, 255),
            MissionState.LAND: self.FAIL,
        }
        return colors.get(state, self.TEXT)

    def _draw_legend(self, frame: np.ndarray, track_bbox: Optional[np.ndarray]):
        """Sağ alt köşeye kimin kim olduğunu gösteren anlaşılır lejant kartı çizer."""
        leg_w, leg_h = 360, 130
        leg_x = self._fw - leg_w - 16
        leg_y = self._fh - leg_h - 16

        # Arka plan paneli
        sub = frame[leg_y:leg_y + leg_h, leg_x:leg_x + leg_w]
        bg = np.zeros_like(sub)
        cv2.rectangle(bg, (0, 0), (leg_w, leg_h), (22, 22, 22), -1)
        frame[leg_y:leg_y + leg_h, leg_x:leg_x + leg_w] = cv2.addWeighted(sub, 0.35, bg, 0.65, 0)
        cv2.rectangle(frame, (leg_x, leg_y), (leg_x + leg_w, leg_y + leg_h), (0, 215, 255), 1)

        # Başlık
        cv2.putText(frame, "SIMULASYON KIMLIK LEJANTI", (leg_x + 12, leg_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1, cv2.LINE_AA)
        
        # 1. Biz kimiz?
        cv2.circle(frame, (leg_x + 20, leg_y + 46), 5, (255, 255, 255), -1)
        cv2.putText(frame, "BIZ  : AVCI IHA (Kamera Kokpiti / Hunter)", (leg_x + 32, leg_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Av kim?
        cv2.circle(frame, (leg_x + 20, leg_y + 72), 5, self.LOCK_COLOR, -1)
        cv2.putText(frame, "HEDEF: AV IHA (Onumuzdeki Ucak / Prey)", (leg_x + 32, leg_y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 160, 255), 1, cv2.LINE_AA)

        # 3. Renkler
        cv2.putText(frame, "Yesil: Ham Tespit | Kirmizi: Kalman Kilit Kutusu", (leg_x + 12, leg_y + 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, "Sari Cizgi: Hedefe Yonelim Vektoru (LOS)", (leg_x + 12, leg_y + 118),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.GUIDE, 1, cv2.LINE_AA)

        # Hedef uçağın hemen üzerine doğrudan canlı etiket
        if track_bbox is not None:
            bx1, by1, bx2, by2 = [int(v) for v in track_bbox]
            cv2.putText(frame, ">> HEDEF (AV IHA) <<", (bx1, max(24, by1 - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.LOCK_COLOR, 2, cv2.LINE_AA)
