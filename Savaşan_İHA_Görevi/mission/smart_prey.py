"""
Savaşan İHA - Akıllı Av Otopilotu (Smart Prey / Evader AI)

===============================================================================
🧑‍🏫 YENİ BAŞLAYANLAR İÇİN REHBER: OYUN TEORİSİ VE AKILLI KAÇIŞ ZEKASI
===============================================================================

1. NEDEN BU MODÜLE İHTİYAÇ VAR?
   Klasik sistemde av İHA, gökyüzünde dairesel bir yörüngede hiçbir şeyden habersiz
   dönen "kör ve sağır" bir hedefti. Gerçek bir hava muharebesinde (dogfight) hiçbir
   hedef arkasından bir avcının yaklaştığını ve kilitlenmek üzere olduğunu bile bile
   dümdüz uçmaya devam etmez!
   
   Bu modül, Otonom Zeki Sistemler dersinin ruhuna uygun olarak av İHA'ya bir
   "Tehdit Algılama & Kaçış Zekası (Evader AI)" kazandırır.

2. AV İHA TEHLİKEYİ NASIL HİSSEDER? (Tehdit Değerlendirmesi)
   Av İHA her saniye şu iki soruyu sorar:
   - "Avcı İHA bana 80 metreden daha yakın mı?" (Mesafe Kriteri)
   - "Avcı İHA benim kuyruk/arka açımda mı?" (Görüş Açısı Kriteri)
   Eğer iki cevap da "EVET" ise, avcı bize kilitlenmek üzeredir! Panik ve Kaçış Modu başlar!

3. KAÇIŞ TAKTİKLERİ (NASIL KAÇAR?):
   Yarışma şartnamesindeki "4.0 saniye kesintisiz hedefe bakma" kuralını bozmak için
   askeri havacılıkta kullanılan 3 taktiği sırayla uygular:
   
   - Taktik 1: S-Turn (Zikzak Kaçış):
     Sağa ve sola periyodik sert manevralar yaparak avcının nişangahını bozar.
     Avcının 4 saniyelik sayacı sürekli 1-2. saniyelerde sıfırlanır!
   
   - Taktik 2: İrtifa Kaçışı (Altitude Jink):
     Avcı kamerasının dikey açısı (75°) yatay açısına (110°) göre daha dardır.
     Hedef aniden 15-20 metre alçalıp tırmanarak kameranın alt/üst kadrajından kaçmaya çalışır.
   
   - Taktik 3: Kırılma Dönüşü (Hard Break Turn):
     Uçağın dönüş kabiliyetini sonuna kadar kullanarak avcının arkasından çıkacak
     keskin bir viraj alır.
===============================================================================
"""

from __future__ import annotations

import math
import time
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from core.logger import get_logger


class PreyBehaviorMode(Enum):
    """Av İHA Davranış Modları"""
    PATROL = auto()       # Tehlike yok, sakin devriye / daire uçuşu
    EVADE_ZIGZAG = auto() # S-Turn zikzak kaçış manevrası
    EVADE_ALT = auto()    # Ani dalış/tırmanış ile dikey kadrajdan kaçış
    BREAK_TURN = auto()   # Maksimum açılı keskin viraj


class SmartPreyPilot:
    """
    Otonom Kaçış Zekasına (Evader AI) Sahip Akıllı Av Otopilotu.
    Geriye dönük olarak orijinal PreyPilot ile %100 uyumludur.
    """

    def __init__(
        self,
        center_lat: float = -35.3632622,
        center_lon: float = 149.1652376,
        patrol_radius: float = 100.0,
        altitude: float = 100.0,
        min_altitude: float = 80.0,
        max_altitude: float = 125.0,
        hold_altitude: bool = True,
        # Kaçış Parametreleri
        threat_distance_m: float = 85.0,     # Tehlike algılama menzili (metre)
        rear_threat_cone_deg: float = 70.0,   # Avın arkasındaki tehdit açısı konisi
        evade_duration: float = 12.0,        # Kaçış taktiği süresi (saniye)
        smart_enabled: bool = True,           # Akıllı kaçış aktif mi?
        step_deg: float = 5.0,
        period: float = 1.0,
        fw_throttle: int = 1600,
    ):
        self._log = get_logger()
        self._center_lat = center_lat
        self._center_lon = center_lon
        self._radius = abs(patrol_radius)
        self._base_altitude = altitude
        self._min_altitude = min_altitude
        self._max_altitude = max_altitude
        self._hold_altitude = hold_altitude

        # Akıllı av parametreleri
        self.smart_enabled = smart_enabled
        self._threat_distance = threat_distance_m
        self._rear_cone_rad = math.radians(rear_threat_cone_deg)
        self._evade_duration = evade_duration

        # Durum yönetimi
        self._mode = PreyBehaviorMode.PATROL
        self._mode_start_time = 0.0
        self._last_cmd_time: Optional[float] = None
        self._cmd_count = 0
        self._zigzag_phase = 0
        self._last_evade_switch = 0.0
        self._current_target_alt = altitude

    @property
    def current_mode(self) -> PreyBehaviorMode:
        return self._mode

    def _meters_to_lat_lon(self, d_north: float, d_east: float) -> Tuple[float, float]:
        """Metre cinsinden yerel sapmayı GPS koordinatlarına çevirir."""
        earth_radius = 6378137.0
        d_lat = d_north / earth_radius
        d_lon = d_east / (earth_radius * math.cos(math.pi * self._center_lat / 180.0))
        return (
            self._center_lat + (d_lat * 180.0 / math.pi),
            self._center_lon + (d_lon * 180.0 / math.pi),
        )

    def assess_threat(self, prey_vehicle, hunter_vehicle) -> Tuple[bool, float]:
        """
        Avcı İHA'nın tehdit oluşturup oluşturmadığını değerlendirir.
        Dönen: (tehdit_var_mı, aradaki_mesafe)
        """
        if not self.smart_enabled or hunter_vehicle is None:
            return False, 999.0

        p_telem = prey_vehicle.get_telemetry() if hasattr(prey_vehicle, "get_telemetry") else {}
        h_telem = hunter_vehicle.get_telemetry() if hasattr(hunter_vehicle, "get_telemetry") else {}

        p_lat = p_telem.get("lat")
        p_lon = p_telem.get("lon")
        h_lat = h_telem.get("lat")
        h_lon = h_telem.get("lon")

        if None in (p_lat, p_lon, h_lat, h_lon):
            return False, 999.0

        # İki İHA arasındaki yaklaşık yatay mesafe (Haversine yaklaşımı)
        d_lat = (h_lat - p_lat) * 111319.5
        d_lon = (h_lon - p_lon) * 111319.5 * math.cos(math.radians(p_lat))
        d_alt = (h_telem.get("alt", 100.0) or 100.0) - (p_telem.get("alt", 100.0) or 100.0)
        distance = math.sqrt(d_lat**2 + d_lon**2 + d_alt**2)

        if distance > self._threat_distance:
            return False, distance

        # Tehdit açısı kontrolü: Avcı avın arkasından mı geliyor?
        prey_yaw = math.radians(p_telem.get("heading", 0.0) or 0.0)
        rel_angle = math.atan2(d_lon, d_lat)
        angle_diff = abs((rel_angle - prey_yaw + math.pi) % (2 * math.pi) - math.pi)

        # Eğer avcı arkadaki koni içindeyse tehdit doğrulanır
        is_rear_threat = angle_diff > (math.pi - self._rear_cone_rad)
        return is_rear_threat, distance

    def update(
        self,
        vehicle,
        now: Optional[float] = None,
        hunter_vehicle=None,
    ):
        """
        Av İHA döngü güncellemesi.
        Tehdit durumuna göre sakin devriye veya kaçış taktiklerini çalıştırır.
        """
        now = time.time() if now is None else float(now)
        if self._last_cmd_time is None:
            self._last_cmd_time = 0.0

        # 1. Tehdit Seviyesini Değerlendir
        is_threat, dist = self.assess_threat(vehicle, hunter_vehicle)

        # Durum Geçişleri
        if is_threat and self._mode == PreyBehaviorMode.PATROL:
            # Tehdit başladı! Zikzak kaçış ile başla
            self._mode = PreyBehaviorMode.EVADE_ZIGZAG
            self._mode_start_time = now
            self._last_evade_switch = now
            self._log.warn(
                "SMART_PREY",
                f"🚨 TEHLİKE ALGILANDI! Avcı {dist:.1f}m arkada! S-Turn Kaçış Manevrası Başlatılıyor!",
            )
        elif not is_threat and self._mode != PreyBehaviorMode.PATROL:
            # Tehdit ortadan kalktı veya avcı çok uzaklaştı
            if now - self._mode_start_time > self._evade_duration:
                self._mode = PreyBehaviorMode.PATROL
                self._log.info("SMART_PREY", "🛡️ Tehdit atlatıldı. Sakin devriye (Patrol) moduna dönüldü.")

        # 2. Aktif Moda Göre Uçuş Komutunu İlet
        if self._mode == PreyBehaviorMode.PATROL:
            self._execute_patrol(vehicle, now)
        else:
            self._execute_evasion(vehicle, now, dist)

    def _execute_patrol(self, vehicle, now: float):
        """Sakin dairesel devriye uçuşu (Native Loiter)."""
        if now - self._last_cmd_time >= 5.0:
            target_alt = self._base_altitude
            if hasattr(vehicle, "loiter_at"):
                vehicle.loiter_at(
                    lat=self._center_lat,
                    lon=self._center_lon,
                    alt=target_alt,
                    radius=self._radius,
                )
            elif hasattr(vehicle, "goto"):
                vehicle.goto(self._center_lat, self._center_lon, target_alt)

            self._last_cmd_time = now
            self._cmd_count += 1

    def _execute_evasion(self, vehicle, now: float, distance_to_hunter: float):
        """
        Kaçış Taktiklerini dinamik olarak çalıştırır.
        Her 2.5 - 3.5 saniyede bir manevra açısını veya irtifayı değiştirir.
        """
        if now - self._last_evade_switch >= 3.0:
            self._last_evade_switch = now
            self._zigzag_phase = (self._zigzag_phase + 1) % 4

            # Fazlara göre kaçış stratejisi:
            # Faz 0 & 2: Sağa / Sola S-Turn sapması
            # Faz 1: İrtifa değişimi (Dikey kaçış)
            # Faz 3: Keskin viraj dönüşü
            telem = vehicle.get_telemetry() if hasattr(vehicle, "get_telemetry") else {}
            curr_lat = telem.get("lat", self._center_lat) or self._center_lat
            curr_lon = telem.get("lon", self._center_lon) or self._center_lon
            curr_yaw = math.radians(telem.get("heading", 0.0) or 0.0)

            if self._zigzag_phase == 0:
                self._mode = PreyBehaviorMode.EVADE_ZIGZAG
                turn_yaw = curr_yaw + math.radians(45.0)  # 45 derece sağa kır
                self._current_target_alt = self._base_altitude
                self._log.info("SMART_PREY", "⚡ [TAKTİK 1] S-Turn: Sağa 45° sert kaçış!")
            elif self._zigzag_phase == 1:
                self._mode = PreyBehaviorMode.EVADE_ALT
                turn_yaw = curr_yaw
                # Ani irtifa düşür/yükselt (dikey kamerayı kadraj dışı bırak)
                self._current_target_alt = (
                    self._min_altitude
                    if abs(self._current_target_alt - self._base_altitude) < 5.0
                    else self._max_altitude
                )
                self._log.info("SMART_PREY", f"⚡ [TAKTİK 2] Dikey Kaçış: Ani irtifa değişimi -> {self._current_target_alt:.0f}m!")
            elif self._zigzag_phase == 2:
                self._mode = PreyBehaviorMode.EVADE_ZIGZAG
                turn_yaw = curr_yaw - math.radians(50.0)  # 50 derece sola kır
                self._log.info("SMART_PREY", "⚡ [TAKTİK 1] S-Turn: Sola 50° sert kaçış!")
            else:
                self._mode = PreyBehaviorMode.BREAK_TURN
                turn_yaw = curr_yaw + math.radians(85.0)  # 85 derece keskin viraj
                self._log.info("SMART_PREY", "⚡ [TAKTİK 3] Break Turn: 85° acil viraj manevrası!")

            # Yeni kaçış noktası üret (150 metre ileriye doğru)
            d_north = 150.0 * math.cos(turn_yaw)
            d_east = 150.0 * math.sin(turn_yaw)
            target_lat, target_lon = self._meters_to_lat_lon(d_north, d_east)

            if hasattr(vehicle, "goto_with_speed"):
                vehicle.goto_with_speed(
                    lat=target_lat,
                    lon=target_lon,
                    alt=self._current_target_alt,
                    airspeed=20.0,  # Kaçışta tam gaz (20 m/s)
                )
            elif hasattr(vehicle, "goto"):
                vehicle.goto(target_lat, target_lon, self._current_target_alt)

            self._last_cmd_time = now

