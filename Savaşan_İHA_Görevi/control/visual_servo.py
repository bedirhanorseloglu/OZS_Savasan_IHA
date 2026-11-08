"""
Savaşan İHA - Image-Based Visual Servoing (IBVS) Controller

===============================================================================
🧑‍🏫 GÖRSEL SERVO VE PID KONTROLCÜ NEDİR?
===============================================================================

1. GÖRSEL SERVO (VISUAL SERVOING) NEDİR?
   Basitçe uçağa göz-el (göz-kanat) koordinasyonu kazandırmaktır.
   Tıpkı bir insanın uçan bir tenis topunu gözleriyle takip edip raketini topa
   yöneltmesi gibi; uçak da kameradan gelen görüntüde hedefin ekranın neresinde
   olduğuna bakar.
   - Hedef ekranın sağındaysa -> Uçak sağa döner (Yaw komutu).
   - Hedef ekranın yukarısındaysa -> Uçak tırmanır (Vz dikey hız komutu).
   - Amaç: Hedefi her zaman ekranın tam ortasında (nişangahın göbeğinde) tutmaktır!

2. PID KONTROLCÜ NEDİR? (HAYATTAN ÇOK BASİT BİR BENZETME)
   Bir arabayı şeridin ortasında tutmaya çalıştığınızı düşünün:
   - P (Proportional / Orantısal - ŞİMDİKİ ZAMAN):
     "Şeritten ne kadar uzaksam direksiyonu o kadar sert çeviririm."
     Hedef ekran merkezinden 200 piksel uzaktaysa sert dön, 10 piksel uzaktaysa tatlı dön.
   
   - I (Integral / Toplamsal - GEÇMİŞ ZAMAN):
     "Sürekli hafif rüzgar esiyor ve araba hep 2 cm sağa kayıyor."
     P bunu tek başına çözemez. I bileşeni zamanla biriken küçük hataları toplar
     ve "Sürekli sağa kayıyorsun, direksiyona hafifçe sürekli sola güç ver" der.
   
   - D (Derivative / Türevsel - GELECEK ZAMAN / FREN):
     "Şeride çok hızlı yaklaşıyorum, direksiyonu erkenden düzeltmezsem öbür şeride fırlarım!"
     Hatanın değişim hızına bakar. Hedefe çok hızlı yaklaşıyorsak uçağı frenler,
     aşırı savrulmayı (overshoot) engeller.

3. KRİTİK GÜVENLİK KAVRAMLARI:
   - Anti-windup (İntegral Sınırlandırma): Uçak dönüş sınırına dayandıysa ama hata hala
     sıfırlanmadıysa, I değeri sonsuza doğru büyümesin diye sınırlanır (patlamayı önler).
   - Deadband (Ölü Bölge): Hedef zaten merkeze çok yakınsa (örn: 1-2 piksel),
     uçağın motorlarını gereksiz yere titretmemek için hata "sıfır" kabul edilir.
===============================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from core.logger import get_logger


@dataclass
class ServoCommand:
    """Visual servo kontrolcüsünün uçağın otopilotuna verdiği emir paketi."""
    yaw_rate: float = 0.0       # Sağa/Sola dönüş açısı/hızı (derece)
    vz: float = 0.0             # Tırmanma/Alçalma hızı (m/s, negatif=tırman, pozitif=alçal)
    forward_speed: float = 0.0  # İleri uçuş hızı (m/s)
    lateral_speed: float = 0.0  # Yanal kayma hızı (m/s)
    x_error: float = 0.0       # Yatay hata payı (-1.0 ile +1.0 arasında normalize)
    y_error: float = 0.0       # Dikey hata payı (-1.0 ile +1.0 arasında normalize)
    is_centered: bool = False   # Hedef ekranın ortasındaki güvenli alanda mı?


class PIDController:
    """
    Anti-windup (aşırı şişme korumalı) ve türev filtreli klasik PID kontrolcü.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_limit: float,
        integral_limit: float = 10.0,
        deadband: float = 0.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.deadband = deadband

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0
        self._d_filter = 0.35  # low-pass filtre katsayısı

    def compute(self, error: float, dt: float) -> float:
        if abs(error) < self.deadband:
            self._integral *= 0.90  # integral sönümlemesi
            return 0.0

        p = self.kp * error

        # Anti-windup özellikli integral
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self._integral

        # Türev
        if dt > 1e-6:
            raw_d = (error - self._prev_error) / dt
            filtered_d = self._d_filter * raw_d + (1.0 - self._d_filter) * self._prev_derivative
            self._prev_derivative = filtered_d
            d = self.kd * filtered_d
        else:
            d = 0.0

        self._prev_error = error
        output = p + i + d
        return float(np.clip(output, -self.output_limit, self.output_limit))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_derivative = 0.0


class VisualServo:
    """
    Kamera açısı ve mesafe tahminiyle çalışan geometrik görsel servo kontrolcü.
    """

    def __init__(
        self,
        frame_width: int = 1280,
        frame_height: int = 720,
        # Yaw (Yatay Yönelim) PID
        yaw_kp: float = 1.1,       # Derece hatası başına heading değişimi
        yaw_ki: float = 0.05,
        yaw_kd: float = 0.25,
        max_yaw_rate: float = 45.0,  # Maksimum dönüş açısı (derece)
        yaw_deadband: float = 0.8,   # 0.8 derecenin altı sönümlenir
        yaw_integral_limit: float = 15.0,
        # Altitude (Dikey İrtifa) PID
        alt_kp: float = 0.8,       # Metre bazlı irtifa hatası kazancı
        alt_ki: float = 0.04,
        alt_kd: float = 0.18,
        max_vertical_speed: float = 4.0,  # m/s
        alt_deadband: float = 0.4,   # 0.4 metre tolerans
        alt_integral_limit: float = 5.0,
        # Lateral kontrol
        max_lateral_speed: float = 3.0,
        lateral_gain: float = 2.0,
        # Öngörü süresi
        lead_time: float = 0.18,
        fine_error_threshold: float = 0.12,
        fine_gain_multiplier: float = 0.6,
        # Değişim limitleri (slew rates)
        max_yaw_delta: float = 15.0,
        max_speed_delta: float = 1.0,
        max_vz_delta: float = 0.8,
        # Güdüm Modu: "PURE_PURSUIT" (Klasik PID) veya "PN" (Oransal Seyrüsefer)
        guidance_mode: str = "PN",
        pn_gain: float = 3.2,
        horizontal_fov_deg: float = 110.0,
        vertical_fov_deg: float = 75.0,
    ):
        self._log = get_logger()
        self._fw = frame_width
        self._fh = frame_height
        self._half_w = frame_width / 2.0
        self._half_h = frame_height / 2.0
        self.guidance_mode = guidance_mode.upper()
        self._pn_gain = pn_gain
        self._prev_heading_deg: Optional[float] = None
        self._filtered_los_rate = 0.0
        self._max_yaw_rate = max_yaw_rate
        self._fov_h = float(horizontal_fov_deg)
        self._fov_v = float(vertical_fov_deg)

        # PID controllers
        self._yaw_pid = PIDController(
            kp=yaw_kp, ki=yaw_ki, kd=yaw_kd,
            output_limit=max_yaw_rate,
            integral_limit=yaw_integral_limit,
            deadband=yaw_deadband,
        )
        self._alt_pid = PIDController(
            kp=alt_kp, ki=alt_ki, kd=alt_kd,
            output_limit=max_vertical_speed,
            integral_limit=alt_integral_limit,
            deadband=alt_deadband,
        )

        self._max_lateral = max_lateral_speed
        self._lateral_gain = lateral_gain
        self._lead_time = lead_time
        self._fine_threshold = fine_error_threshold
        self._fine_gain = fine_gain_multiplier

        self._max_yaw_delta = max_yaw_delta
        self._max_speed_delta = max_speed_delta
        self._max_vz_delta = max_vz_delta

        self._prev_yaw = 0.0
        self._prev_vz = 0.0
        self._prev_lateral = 0.0
        self._last_time: Optional[float] = None

    def compute(
        self,
        target_cx: float,
        target_cy: float,
        target_vx: float = 0.0,
        target_vy: float = 0.0,
        bbox_width: float = 40.0,
        desired_width_ratio: float = 0.040,
        now: Optional[float] = None,
    ) -> ServoCommand:
        """
        Görsel verilerden yönlendirme komutunu hesaplar.
        """
        now = time.time() if now is None else float(now)
        dt = 0.033
        if self._last_time is not None:
            dt = max(now - self._last_time, 1e-4)
        self._last_time = now

        # 1. GELECEĞİ TAHMİN ET (Lead Compensation):
        # Uçak komut verip dönene kadar milisaniyeler geçer.
        # Bu yüzden hedefin ŞU ANKİ yerine değil, hedefin hızını hesaba katarak
        # "biraz sonra olacağı yere" (pred_cx, pred_cy) nişan alıyoruz.
        pred_cx = target_cx + target_vx * self._lead_time
        pred_cy = target_cy + target_vy * self._lead_time

        # Pikselleri -1.0 (en sol/üst) ile +1.0 (en sağ/alt) arasına normalize ediyoruz.
        # 0.0 tam ekranın ortası (hedef noktamız) demektir.
        x_error = (pred_cx - self._half_w) / self._half_w  # [-1, 1]
        y_error = (pred_cy - self._half_h) / self._half_h  # [-1, 1]

        # Kameranın görüş açıları (Lens FOV - Field of View):
        # Kamera yatayda toplam 110 dereceyi, dikeyde 75 dereceyi görüyor.
        fov_h = self._fov_h
        fov_v = self._fov_v

        # 2. PİKSEL HATASINI AÇIYA ÇEVİRME:
        # Ekrandaki piksel kaçıklığını uçağın pusula açısına (dereceye) çeviriyoruz.
        # Örneğin hedef ekranın en sağındaysa (x_error = +1.0), uçak 55 derece sağa bakmalıdır.
        heading_error_deg = x_error * (fov_h / 2.0)

        # 3. TEK KAMERAYLA METRE CİNSİNDEN MESAFE TAHMİNİ (Pinhole Geometrisi):
        # Derinlik kameramız (LiDAR) yok, tek bir standart kameramız var.
        # Peki hedefin kaç metre uzakta olduğunu nasıl anlıyoruz?
        # Benzetme: Bir araba size yaklaştıkça gözünüzde büyür, uzaklaştıkça küçülür.
        # Hedef İHA kutusu ekranda ne kadar çok piksel kaplıyorsa o kadar yakındır!
        width_ratio = max(bbox_width / self._fw, 1e-5)
        estimated_dist = 70.0 * (desired_width_ratio / width_ratio)
        estimated_dist = np.clip(estimated_dist, 10.0, 180.0)

        # Dikey açı hatası ve trigonometri ile irtifa farkı (metre):
        # İrtifa Farkı = Mesafe * sin(Dikey Açı)
        # Image y grows downward: a target below center requires a descent command.
        pitch_error_deg = y_error * (fov_v / 2.0)
        altitude_error_m = estimated_dist * math.sin(math.radians(pitch_error_deg))

        # 4. KAZANÇ AYARLAMA (Gain Scheduling):
        # Hedef merkeze çok yakınsa PID kazançlarını biraz yumuşatıyoruz ki uçak
        # titreyip kilitlenmeyi bozmasın (ince ayar modu).
        error_mag = max(abs(x_error), abs(y_error))
        if error_mag < self._fine_threshold:
            gain_scale = self._fine_gain + (1.0 - self._fine_gain) * (error_mag / self._fine_threshold)
        else:
            gain_scale = 1.0

        # 5. GÜDÜM VE YÖNLENDİRME KOMUTU HESABI (PN vs Pure Pursuit):
        # -------------------------------------------------------------
        # A) ORANSAL SEYRÜSEFER (Proportional Navigation - PN):
        # Hedef manevra yapıyorsa (viraj alıyorsa), arkasından kovalamak yerine
        # görüş hattı dönüş hızına (LOS Rate: dλ/dt) bakarak hedefin önüne "kesme" atar.
        if self.guidance_mode == "PN":
            if self._prev_heading_deg is not None and dt > 1e-4:
                raw_los_rate = (heading_error_deg - self._prev_heading_deg) / dt
            else:
                raw_los_rate = 0.0
            self._prev_heading_deg = heading_error_deg

            # Gürültü yumuşatma (Low-pass filter)
            self._filtered_los_rate = 0.65 * raw_los_rate + 0.35 * self._filtered_los_rate
            
            # Augmented PN: a_turn = N * los_rate + K_align * heading_error
            pn_cmd = self._pn_gain * self._filtered_los_rate + (1.2 * heading_error_deg * gain_scale)
            yaw_rate = float(np.clip(pn_cmd, -self._max_yaw_rate, self._max_yaw_rate))
        else:
            # B) KLASİK PURE PURSUIT (PID):
            # Hedefin şu anki açısal hatasını doğrudan PID döngüsüyle sıfırlamaya çalışır.
            yaw_rate = self._yaw_pid.compute(heading_error_deg * gain_scale, dt)

        # Dikey irtifa kontrolü (Vz m/s)
        vz = self._alt_pid.compute(altitude_error_m * gain_scale, dt)

        # Lateral düzeltme (hızlı yan kayma)
        lateral = 0.0
        if abs(x_error) > 0.08:
            lateral = x_error * self._lateral_gain
            lateral = float(np.clip(lateral, -self._max_lateral, self._max_lateral))

        # Slew rate limitleri
        yaw_rate = self._slew(yaw_rate, self._prev_yaw, self._max_yaw_delta)
        vz = self._slew(vz, self._prev_vz, self._max_vz_delta)
        lateral = self._slew(lateral, self._prev_lateral, self._max_speed_delta)

        self._prev_yaw = yaw_rate
        self._prev_vz = vz
        self._prev_lateral = lateral

        # Merkeze yakınlık kontrolü
        is_centered = abs(x_error) < 0.12 and abs(y_error) < 0.12

        self._log.debug("SERVO",
            f"GEOMETRIC CMD: yaw_corr={yaw_rate:+.1f}° vz={vz:+.1f}m/s lat={lateral:+.1f}m/s "
            f"ang_err=({heading_error_deg:+.1f}°, {pitch_error_deg:+.1f}°) alt_err={altitude_error_m:+.1f}m dist={estimated_dist:.1f}m",
            debounce=0.15)

        return ServoCommand(
            yaw_rate=yaw_rate,
            vz=vz,
            forward_speed=0.0,
            lateral_speed=lateral,
            x_error=x_error,
            y_error=y_error,
            is_centered=is_centered,
        )

    @staticmethod
    def _slew(target: float, prev: float, max_delta: float) -> float:
        delta = target - prev
        if abs(delta) > max_delta:
            return prev + max_delta * np.sign(delta)
        return target

    def reset(self):
        self._yaw_pid.reset()
        self._alt_pid.reset()
        self._prev_heading_deg = None
        self._filtered_los_rate = 0.0
        self._prev_yaw = 0.0
        self._prev_vz = 0.0
        self._prev_lateral = 0.0
        self._last_time = None
