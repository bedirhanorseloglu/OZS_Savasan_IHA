"""
Savaşan İHA - Mesafe Kontrolcüsü (Distance Controller)

===============================================================================
🧑‍🏫 NEDEN LOGARİTMİK MESAFE KONTROLÜ?
===============================================================================

1. PROBLEM: MESAFE SENSÖRÜMÜZ (LIDAR) YOKKEN MESAFEYİ NASIL KORURUZ?
   Avcı uçağımız, öndeki av uçağını ne çok uzaktan takip etmeli (şartnamedeki %5
   boyut kuralı sağlanamaz) ne de çok yakınına girmelidir (çarpışma veya uçağın
   önüne geçip onu kaybetme - fly-past riski).
   Elimizdeki tek sensör kamera olduğu için mesafeyi "Hedefin ekrandaki kutu genişliği"
   (bbox_width) üzerinden kontrol ederiz.
   - Kutu küçükse -> Hedef uzak demektir -> Gaz ver, hızlan!
   - Kutu büyükse -> Hedef çok yakın demektir -> Gaz kes, yavaşla!

2. NEDEN DÜZ ÇIKARMA DEĞİL DE "LOGARİTMA (ln)" KULLANIYORUZ?
   Optik perspektif kanunları gereği:
   - Hedef 150 metreden 100 metreye geldiğinde ekrandaki boyutu sadece 5 piksel büyür.
   - Ama hedef 20 metreden 10 metreye geldiğinde ekrandaki boyutu aniden 150 piksel büyür!
   
   Eğer düz çıkarma (Hedef_Boyut - Mevcut_Boyut) yapsaydık:
   Uçak uzaktayken "Nasıl olsa sadece 5 piksel fark var" deyip kaplumbağa gibi yavaş giderdi,
   yakına gelince de 150 piksellik farkı görüp panikle aşırı sert fren yapardı.
   
   Logaritma (ln(İstenen_Oran / Mevcut_Oran)) bu optik dengesizliği matematiksel olarak
   tamamen pürüzsüzleştirir. Böylece uçak hem 100 metrede hem 20 metrede son derece
   akıcı ve kararlı bir hızla mesafeyi korur.
===============================================================================
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.logger import get_logger


class DistanceController:
    """
    Bbox genişliği (width) tabanlı mesafe kontrolü.

    width_ratio = bbox_width / frame_width
    desired_width_ratio → hedef mesafe (~70-100m arası)
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        fwd_kp: float = 3.0,
        fwd_ki: float = 0.1,
        fwd_kd: float = 1.0,
        max_forward_speed: float = 6.0,
        max_reverse_speed: float = 3.0,
        desired_width_ratio: float = 0.040,
        min_width_ratio: float = 0.015,
        max_width_ratio: float = 0.090,
        max_speed_delta: float = 0.8,
    ):
        self._log = get_logger()
        self._frame_width = float(frame_width)
        self._desired_ratio = desired_width_ratio
        self._min_ratio = min_width_ratio
        self._max_ratio = max_width_ratio
        self._max_fwd = max_forward_speed
        self._max_rev = max_reverse_speed
        self._kp = fwd_kp
        self._ki = fwd_ki
        self._kd = fwd_kd
        self._max_delta = max_speed_delta

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_cmd = 0.0
        self._prev_ratio: Optional[float] = None

    def compute(
        self,
        bbox_width: float,
        x_error: float = 0.0,
        y_error: float = 0.0,
        dt: float = 0.033,
    ) -> float:
        """
        İleri hızı (forward speed) hedef bbox genişliğine göre hesaplar.

        Parametreler:
            bbox_width: hedef bbox genişliği (piksel)
            x_error: normalleştirilmiş yatay hata (hizalama kontrolü)
            dt: zaman adımı

        Dönen:
            forward_speed (m/s), pozitif = ileri, negatif = geri
        """
        ratio = bbox_width / max(self._frame_width, 1.0)
        prev_ratio = self._prev_ratio
        self._prev_ratio = ratio

        if ratio < 1e-6:
            self._prev_cmd = min(self._max_fwd * 0.6, 10.0)
            return self._prev_cmd

        log_error = np.log(self._desired_ratio / ratio)  # pozitif = çok uzak

        # PID
        p = self._kp * log_error
        self._integral += log_error * dt
        self._integral = np.clip(self._integral, -5.0, 5.0)
        i = self._ki * self._integral

        if dt > 1e-6:
            d = self._kd * (log_error - self._prev_error) / dt
        else:
            d = 0.0
        self._prev_error = log_error

        speed = p + i + d

        size_ratio = ratio / max(self._desired_ratio, 1e-6)
        edge_error = max(abs(x_error), abs(y_error))
        
        # Hız profilleri - hedefe uzaklığa göre
        if size_ratio < 0.30:
            speed = max(speed, self._max_fwd * 0.95)
        elif size_ratio < 0.55:
            speed = max(speed, self._max_fwd * 0.80)
        elif size_ratio < 0.85:
            speed = max(speed, self._max_fwd * 0.55)
        elif size_ratio < 1.05:
            speed = min(speed, self._max_fwd * 0.38)
        else:
            speed = min(speed, self._max_fwd * 0.22)

        # Hedef merkezden kaçıyorsa hızı düşür ki kamera hedeften kopmasın
        if edge_error > 0.75:
            speed *= 0.35
        elif edge_error > 0.50:
            speed *= 0.55
        elif edge_error > 0.30:
            speed *= 0.75

        # Hedef hızlı yaklaşıyorsa fren yap
        if prev_ratio is not None and prev_ratio > 1e-6:
            growth = ratio / prev_ratio
            if growth > 1.10 and size_ratio > 0.45:
                speed *= 0.72
            elif growth > 1.04 and size_ratio > 0.75:
                speed *= 0.82

        # Sınırla
        speed = float(np.clip(speed, -self._max_rev, self._max_fwd))

        # İvme sınırlaması (slew limiting)
        delta = speed - self._prev_cmd
        if abs(delta) > self._max_delta:
            speed = self._prev_cmd + self._max_delta * np.sign(delta)
        self._prev_cmd = speed

        # Log
        self._log.debug("DIST",
            f"FWD: speed={speed:+.1f}m/s bbox_w={bbox_width:.1f}px ratio={ratio:.6f} "
            f"log_err={log_error:+.2f} x_err={x_error:+.2f}",
            debounce=0.15)

        return speed

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_cmd = 0.0
        self._prev_ratio = None
