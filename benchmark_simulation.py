"""
Savaşan İHA - Akademik Karşılaştırma ve Benchmark Simülatörü
Otonom Zeki Sistemler Dersi Proje Değerlendirme Aracı

===============================================================================
🧑‍🏫 BU BETİK NE İŞE YARAR?
===============================================================================
Bu betik, Gazebo ve ROS simülasyonunu başlatmaya gerek kalmadan,
uçağın matematiksel uçuş dinamiklerini, kamerasını ve yapay zeka takipçisini
simüle eder.

3 Farklı Senaryoyu yarıştırır:
1. Senaryo A (Eski Sistem - Kolay Hedef): Sabit Devriye Gezen Av vs Klasik PID Pursuit
2. Senaryo B (Eski Sistem - Zor Hedef):   Akıllı Kaçan Av vs Klasik PID Pursuit
3. Senaryo C (YENİ SİSTEM - Yüksek Lisans): Akıllı Kaçan Av vs ORANSAL SEYRÜSEFER (PN)

Çıktı olarak:
- Kilitlenme Başarı Yüzdesi (%)
- Kesintisiz Kilit Kalma Süresi (sn)
- Takip Kaybı (Lost Track) Sayısı
hesaplar ve karşılaştırmalı analiz grafiğini 'benchmark_results.png' olarak kaydeder.
===============================================================================
"""

import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SAVASAN_DIR = os.path.join(PROJECT_ROOT, "Savaşan_İHA_Görevi")


def run_benchmark(duration_sec=60.0, dt=0.04):
    """3 Farklı senaryoyu simüle eder ve metrikleri toplar."""
    print("=" * 70)
    print("🚀 SAVAŞAN İHA - OTONOM ZEKİ SİSTEMLER AKADEMİK BENCHMARK TESTİ")
    print("=" * 70)

    steps = int(duration_sec / dt)
    time_series = np.linspace(0, duration_sec, steps)

    scenarios = [
        {"name": "1. Sabit Av + Klasik PID", "smart_prey": False, "guidance": "PURE_PURSUIT"},
        {"name": "2. Akıllı Kaçan Av + Klasik PID", "smart_prey": True, "guidance": "PURE_PURSUIT"},
        {"name": "3. Akıllı Kaçan Av + ORANSAL SEYRÜSEFER (PN)", "smart_prey": True, "guidance": "PN"},
    ]

    results = []

    for sc in scenarios:
        print(f"\n▶️ Test Ediliyor: {sc['name']} ...")
        # Simülasyon Başlangıç Değerleri
        # Av konumu ve yönü
        prey_x, prey_y = 120.0, 0.0
        prey_yaw = math.pi / 2.0
        prey_speed = 15.0

        # Avcı konumu ve yönü
        hunter_x, hunter_y = 60.0, -40.0
        hunter_yaw = math.pi / 2.0
        hunter_speed = 17.0

        lock_counter = 0.0
        total_lock_time = 0.0
        track_loss_count = 0
        error_history = []
        lock_history = []

        prev_los_deg = 0.0
        filtered_los_rate = 0.0

        for t in time_series:
            # 1. Av Hareketi
            if sc["smart_prey"]:
                # Avcı arkasındaysa zikzak ve viraj yap
                rel_x = hunter_x - prey_x
                rel_y = hunter_y - prey_y
                dist = math.hypot(rel_x, rel_y)

                if dist < 85.0:
                    # S-Turn zikzak manevrası (periyodik sinüs dalgası + ani kırılmalar)
                    maneuver_rate = 0.8 * math.sin(2.0 * math.pi * t / 4.0) + (0.5 if (int(t) % 6 == 0) else 0.0)
                    prey_yaw += maneuver_rate * dt
                else:
                    prey_yaw += (prey_speed / 80.0) * dt
            else:
                # Sabit dairesel yörünge (R=80m)
                prey_yaw += (prey_speed / 80.0) * dt

            prey_x += prey_speed * math.cos(prey_yaw) * dt
            prey_y += prey_speed * math.sin(prey_yaw) * dt

            # 2. Avcı Görüş Açısı (Kamera LOS)
            dx = prey_x - hunter_x
            dy = prey_y - hunter_y
            los_angle = math.atan2(dy, dx)
            heading_err = (los_angle - hunter_yaw + math.pi) % (2 * math.pi) - math.pi
            heading_err_deg = math.degrees(heading_err)

            # 3. Güdüm Kontrolü
            if sc["guidance"] == "PN":
                los_rate = (heading_err_deg - prev_los_deg) / dt
                filtered_los_rate = 0.65 * los_rate + 0.35 * filtered_los_rate
                # Oransal Seyrüsefer Komutu (N=3.2)
                turn_rate_deg = 3.2 * filtered_los_rate + 1.2 * heading_err_deg
            else:
                # Klasik Pure Pursuit PID (Orantısal)
                turn_rate_deg = 1.8 * heading_err_deg

            prev_los_deg = heading_err_deg
            turn_rate_rad = math.radians(np.clip(turn_rate_deg, -55.0, 55.0))
            hunter_yaw += turn_rate_rad * dt

            hunter_x += hunter_speed * math.cos(hunter_yaw) * dt
            hunter_y += hunter_speed * math.sin(hunter_yaw) * dt

            # 4. Kilitlenme Kriterleri Kontrolü (Hata < 8 derece ve Mesafe < 80m)
            err_abs = abs(heading_err_deg)
            error_history.append(err_abs)

            is_locked = (err_abs < 8.5) and (math.hypot(dx, dy) < 85.0)
            if is_locked:
                lock_counter += dt
                if lock_counter >= 4.0:
                    total_lock_time += dt
            else:
                if lock_counter > 0.0 and lock_counter < 4.0:
                    track_loss_count += 1
                lock_counter = max(0.0, lock_counter - dt * 1.5)

            lock_history.append(1 if is_locked else 0)

        rmse = math.sqrt(np.mean(np.square(error_history)))
        lock_ratio = (total_lock_time / duration_sec) * 100.0

        results.append({
            "name": sc["name"],
            "lock_time": total_lock_time,
            "lock_ratio": lock_ratio,
            "track_losses": track_loss_count,
            "rmse": rmse,
            "error_hist": error_history,
            "lock_hist": lock_history,
        })

        print(f"   ⏱️ Toplam Kilit Süresi: {total_lock_time:.1f} sn (%{lock_ratio:.1f})")
        print(f"   ⚠️ Kilit Kopması (Loss): {track_loss_count} kez")
        print(f"   🎯 Açısal Takip Hatası (RMSE): {rmse:.2f}°")

    # Grafik Çizdirme
    _plot_results(time_series, results)
    return results


def _plot_results(time_series, results):
    """Karşılaştırma grafiklerini çizer ve PNG olarak kaydeder."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    # 1. Grafik: Açısal Takip Hatası (° Derece)
    ax1 = axes[0]
    for idx, res in enumerate(results):
        ax1.plot(time_series, res["error_hist"], label=res["name"], color=colors[idx], lw=1.8)
    ax1.axhline(8.5, color="black", linestyle="--", label="Kilitlenme Tolerans Sınırı (8.5°)")
    ax1.set_ylabel("Nişangah Açı Hatası (°)")
    ax1.set_title("Hava Muharebesi Güdüm Başarımı: Nişangah Takip Hatası")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # 2. Grafik: Kilit Durumu (0 veya 1)
    ax2 = axes[1]
    for idx, res in enumerate(results):
        offset = idx * 0.04
        ax2.plot(time_series, np.array(res["lock_hist"]) + offset, label=res["name"], color=colors[idx], lw=2)
    ax2.set_xlabel("Zaman (saniye)")
    ax2.set_ylabel("Kilit Durumu")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Açık / Kayıp", "KİLİTLİ"])
    ax2.set_title("Kesintisiz Kilitlenme Süreci Karşılaştırması")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, "benchmark_results.png")
    plt.savefig(save_path, dpi=200)
    print(f"\n📊 Karşılaştırma Grafiği Kaydedildi: {save_path}")


if __name__ == "__main__":
    run_benchmark(duration_sec=60.0)

