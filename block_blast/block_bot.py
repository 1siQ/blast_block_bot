"""
YEDEK — Kalibre edilmiş, sabit koordinatlı sürüm (1220x2712 telefon)
Evrensel auto_config olmadan, doğrudan çalışır.
Çalıştır: python yedek.py
"""

import keyboard
import numpy as np
import cv2
import time
import subprocess
import itertools

# ==========================================
# AYARLAR 
# ==========================================

ADB_PATH     = r"C:\Users\Lenovo\Desktop\block_blast\platform-tools\adb.exe"
PHONE_WIDTH  = 1220
PHONE_HEIGHT = 2712
BLOCK_SIZE   = 60.0

BOARD_ROI_PHONE  = (649,  55,  1755, 1171)
PIECES_ROI_PHONE = (1803, 58,  2215, 1169)


TEMA_PIKSEL_Y = (1803 + 2215) // 2   # tray'in dikey ortası
TEMA_PIKSEL_X = 20                    # sol kenar, blokların dışı
TEMA_ESIK     = 20                    

# ==========================================
# ADB YARDIMCILARI
# ==========================================

def adb_screenshot():
    subprocess.run([ADB_PATH, 'shell', 'screencap', '-p', '/sdcard/_bot.png'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([ADB_PATH, 'pull', '/sdcard/_bot.png', '_bot_tmp.png'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cv2.imread('_bot_tmp.png')

def adb_swipe(x1, y1, x2, y2, sure_ms=200):
    subprocess.run(
        [ADB_PATH, 'shell', 'input', 'swipe',
         str(x1), str(y1), str(x2), str(y2), str(sure_ms)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

def adb_tap(x, y):
    subprocess.run(
        [ADB_PATH, 'shell', 'input', 'tap', str(x), str(y)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

# ==========================================
# 1. GÖRÜNTÜ İŞLEME
# ==========================================

def tahtayi_oku(ekran):
    t, l, b, r = BOARD_ROI_PHONE
    crop = ekran[t:b, l:r]
    gri  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w    = gri.shape
    hucre_h = h // 8
    hucre_w = w // 8
    merkez_y = np.arange(8) * hucre_h + hucre_h // 2
    merkez_x = np.arange(8) * hucre_w + hucre_w // 2
    return (gri[np.ix_(merkez_y, merkez_x)] > 50).astype(int)

def taslari_oku(ekran):
    t, l, b, r = PIECES_ROI_PHONE
    bgr = ekran[t:b, l:r]
    pixels       = bgr.reshape(-1, 3)
    unique, cnts = np.unique(pixels, axis=0, return_counts=True)
    bg_color     = unique[np.argmax(cnts)]
    diff     = cv2.absdiff(bgr, bg_color)
    sum_diff = np.sum(diff, axis=2)
    maske    = (sum_diff > 30).astype(np.uint8) * 255
    roi_w          = r - l
    parca_genislik = roi_w // 3
    maske_taslar   = [
        maske[:, 0              : parca_genislik],
        maske[:, parca_genislik : parca_genislik * 2],
        maske[:, parca_genislik * 2 : roi_w],
    ]
    tum_taslar = []
    for i in range(3):
        noktalar = cv2.findNonZero(maske_taslar[i])
        if noktalar is not None:
            x, y, w, h = cv2.boundingRect(noktalar)
            if w < 15 or h < 15:
                tum_taslar.append(None)
                continue
            cols = max(1, int(round(w / BLOCK_SIZE)))
            rows = max(1, int(round(h / BLOCK_SIZE)))
            kirpilmis = maske_taslar[i][y:y+h, x:x+w]
            ys = ((np.arange(rows) + 0.5) * (h / rows)).astype(int)
            xs = ((np.arange(cols) + 0.5) * (w / cols)).astype(int)
            tum_taslar.append((kirpilmis[np.ix_(ys, xs)] > 0).astype(int))
        else:
            tum_taslar.append(None)
    return tum_taslar

# ==========================================
# 2. HATA AYIKLAMA
# ==========================================

def referans_piksel_al(ekran):
    return ekran[TEMA_PIKSEL_Y, TEMA_PIKSEL_X].tolist()  # [B, G, R]

def tema_degisti_mi(ekran, referans):
    guncel = ekran[TEMA_PIKSEL_Y, TEMA_PIKSEL_X].tolist()
    fark   = sum(abs(int(g) - int(r)) for g, r in zip(guncel, referans))
    return fark > TEMA_ESIK, fark, guncel

def tema_duzelt():
    print("  Tema duzeltiliyor...")
    adb_tap(1107, 238);  time.sleep(0.5)
    adb_tap(567,  1959); time.sleep(0.5)
    adb_tap(1044, 568);  time.sleep(2)

def debug_tahta(tahta):
    print("\n--- TAHTA ---")
    for r in range(8):
        print(" ".join("[#]" if tahta[r, c] else "[ ]" for c in range(8)))

def debug_taslar(taslar):
    print("\n--- TASLAR ---")
    for i, tas in enumerate(taslar):
        if tas is None:
            print(f"{i+1}. BOS")
        else:
            print(f"{i+1}. Tas:")
            for r in range(tas.shape[0]):
                print(" ".join("#" if tas[r, c] else "." for c in range(tas.shape[1])))

# ==========================================
# 3. OYUN MOTORU & YAPAY ZEKA
# ==========================================

def yerlesebilir_mi(tahta, tas, r, c):
    tr, tc = tas.shape
    return not (tahta[r:r+tr, c:c+tc] & tas).any()

def tasi_koy(tahta, tas, r, c):
    yeni = tahta.copy()
    tr, tc = tas.shape
    yeni[r:r+tr, c:c+tc] |= tas
    return yeni

def temizle(tahta):
    satirlar = np.all(tahta == 1, axis=1)
    sutunlar = np.all(tahta == 1, axis=0)
    patlama  = int(np.sum(satirlar) + np.sum(sutunlar))
    if patlama > 0:
        tahta[satirlar, :] = 0
        tahta[:, sutunlar] = 0
    return tahta, patlama

def puanla(tahta):
    toplam = int(np.sum(tahta))
    if toplam > 52:
        return -500000
    ust  = np.zeros_like(tahta); ust[1:, :]  = tahta[:-1, :]
    alt  = np.zeros_like(tahta); alt[:-1, :] = tahta[1:, :]
    sol  = np.zeros_like(tahta); sol[:, 1:]  = tahta[:, :-1]
    sag  = np.zeros_like(tahta); sag[:, :-1] = tahta[:, 1:]
    komsu = ust + alt + sol + sag
    bos  = (tahta == 0)
    dolu = (tahta == 1)
    gap_ceza = (
        int(np.sum(bos & (komsu == 2))) * 15 +
        int(np.sum(bos & (komsu == 3))) * 70 +
        int(np.sum(bos & (komsu == 4))) * 160
    )
    izole_ceza = int(np.sum(dolu & (komsu == 0))) * 40
    if toplam < 22:
        density_ceza = (22 - toplam) * 14
    elif toplam > 38:
        density_ceza = (toplam - 38) * 14
    else:
        density_ceza = 0
    satir_dolu = np.sum(tahta, axis=1)
    sutun_dolu = np.sum(tahta, axis=0)
    yakin_clear = (
        int(np.sum(satir_dolu == 6)) * 30 +
        int(np.sum(satir_dolu == 7)) * 70 +
        int(np.sum(sutun_dolu == 6)) * 30 +
        int(np.sum(sutun_dolu == 7)) * 70
    )
    if toplam > 2:
        ys, xs = np.where(dolu)
        kompakt_ceza = int(np.var(xs.astype(float)) + np.var(ys.astype(float))) * 2
    else:
        kompakt_ceza = 0
    return -gap_ceza - izole_ceza - density_ceza - kompakt_ceza + yakin_clear

def clear_degeri(patlama, combo, son_hamle_mi):
    if patlama == 0:
        return 0
    temel       = patlama * 800
    combo_bonus = min(combo * 50, 2500)
    son_bonus   = 700 if son_hamle_mi else 0

    if combo == 0:
        tip_bonus = patlama * 300
    elif combo < 20:
        # Erken oyun: single öncelikli (combo büyütmek için)
        tip_bonus = 300 if patlama == 1 else (0 if patlama == 2 else -(patlama - 2) * 120)
    elif combo < 50:
        tip_bonus = patlama * 60
    else:
        tip_bonus = patlama * 180

    return temel + combo_bonus + son_bonus + tip_bonus

def dfs(tahta, kalan_sira, eldeki_taslar, cache, combo, clear_oldu_mu):
    state_key = (tahta.tobytes(), tuple(kalan_sira), clear_oldu_mu)
    if state_key in cache:
        return cache[state_key]
    if len(kalan_sira) == 0:
        board_puan = puanla(tahta)
        if not clear_oldu_mu:
            board_puan -= (1500 + combo * 150)
        sonuc = (board_puan, [])
        cache[state_key] = sonuc
        return sonuc
    en_iyi_puan = -999999
    en_iyi_yol  = None
    tas_idx     = kalan_sira[0]
    tas         = eldeki_taslar[tas_idx]
    yerlesti    = False
    ts, tc      = tas.shape
    son_hamle   = (len(kalan_sira) == 1)
    for r in range(9 - ts):
        for c in range(9 - tc):
            if yerlesebilir_mi(tahta, tas, r, c):
                yerlesti  = True
                yeni      = tasi_koy(tahta, tas, r, c)
                yeni, pat = temizle(yeni)
                yeni_clear = clear_oldu_mu or (pat > 0)
                ap, ay = dfs(yeni, kalan_sira[1:], eldeki_taslar, cache, combo, yeni_clear)
                cv = clear_degeri(pat, combo, son_hamle)
                gp = ap + cv
                if ay is not None and gp > en_iyi_puan:
                    en_iyi_puan = gp
                    en_iyi_yol  = [{'tas_idx': tas_idx, 'r': r, 'c': c}] + ay
    if not yerlesti:
        cache[state_key] = (-999999, None)
        return -999999, None
    cache[state_key] = (en_iyi_puan, en_iyi_yol)
    return en_iyi_puan, en_iyi_yol

def en_iyi_diziyi_bul(tahta, taslar, combo=0):
    gecerli_idx = [i for i, t in enumerate(taslar) if t is not None]
    if not gecerli_idx:
        return []
    siralar     = list(itertools.permutations(gecerli_idx))
    en_iyi_puan = -999999
    en_iyi_dizi = None
    cache       = {}
    for sira in siralar:
        puan, yol = dfs(tahta, sira, taslar, cache, combo, False)
        if yol is not None and puan > en_iyi_puan:
            en_iyi_puan = puan
            en_iyi_dizi = yol
            cache = {k: v for k, v in cache.items() if v[0] >= en_iyi_puan - 8000}
    return en_iyi_dizi

# ==========================================
# 4. OTOMASYON
# ==========================================

BASLANGIC_X = [251, 605, 962]
BASLANGIC_Y = [2054, 2054, 2055]

HARITALAR = [
    {'min_x': 171, 'max_x': 845,  'min_y': 1321, 'max_y': 1997},
    {'min_x': 269, 'max_x': 944,  'min_y': 1321, 'max_y': 1997},
    {'min_x': 362, 'max_x': 1042, 'min_y': 1324, 'max_y': 1997},
]

def hamleyi_uygula(hamle, eldeki_taslar):
    tas_idx          = hamle['tas_idx']
    hedef_r, hedef_c = hamle['r'], hamle['c']
    tas              = eldeki_taslar[tas_idx]
    ts, tc           = tas.shape
    harita  = HARITALAR[tas_idx]
    adim_x  = (harita['max_x'] - harita['min_x']) / 7.0
    adim_y  = (harita['max_y'] - harita['min_y']) / 7.0
    merk_c  = hedef_c + (tc - 1) / 2.0
    merk_r  = hedef_r + (ts - 1) / 2.0
    hdf_x   = int(harita['min_x'] + merk_c * adim_x)
    hdf_y   = int(harita['min_y'] + merk_r * adim_y)
    print(f"  [{tas_idx+1}. Tas] ({hedef_r+1},{hedef_c+1}) -> ({hdf_x},{hdf_y})")
    adb_swipe(BASLANGIC_X[tas_idx], BASLANGIC_Y[tas_idx], hdf_x, hdf_y, sure_ms=200)
    time.sleep(0.4)

# ==========================================
# ANA DÖNGÜ
# ==========================================

def oyun_evresi(combo):
    if combo < 20: return "ERKEN"
    if combo < 50: return "ORTA"
    return "YUKSEK"

if __name__ == "__main__":
    print("====================================")
    print("  YEDEK BOT - Sabit koordinatlar")
    print("====================================")
    print("ESC = dur | SPACE = duraklat\n")
    time.sleep(2)

    tur            = 1
    combo          = 0
    referans_renk  = None   

    try:
        while True:
            if keyboard.is_pressed('esc'):
                print("Durduruldu.")
                break
            if keyboard.is_pressed('space'):
                print("Duraklatildi...")
                time.sleep(5)
                continue

            print(f"\n--- TUR {tur} | COMBO: {combo} | {oyun_evresi(combo)} ---")

            print("  Ekran aliniyor...")
            ekran         = adb_screenshot()

 
            if referans_renk is None:
                referans_renk = referans_piksel_al(ekran)
                print(f"  Referans piksel kaydedildi: BGR={referans_renk}")

    
            degisti, fark, guncel = tema_degisti_mi(ekran, referans_renk)
            if degisti:
                print(f"  TEMA DEGISIKLIGI tespit edildi! (fark={fark}, ref={referans_renk}, guncel={guncel})")
                tema_duzelt()
                ekran         = adb_screenshot()
                referans_renk = referans_piksel_al(ekran)   
                print(f"  Yeni referans piksel: BGR={referans_renk}")

            guncel_tahta  = tahtayi_oku(ekran)
            eldeki_taslar = taslari_oku(ekran)

            debug_tahta(guncel_tahta)
            debug_taslar(eldeki_taslar)

            if all(t is None for t in eldeki_taslar):
                print("  Tas yok, bekleniyor...")
                time.sleep(1)
                continue

            print(f"  Hesaplaniyor... (combo={combo})")
            hamle_dizisi = en_iyi_diziyi_bul(guncel_tahta, eldeki_taslar, combo)

            if not hamle_dizisi:
                print("GAME OVER!")
                break

            sim       = guncel_tahta.copy()
            tur_clear = 0
            for h in hamle_dizisi:
                sim = tasi_koy(sim, eldeki_taslar[h['tas_idx']], h['r'], h['c'])
                sim, pat = temizle(sim)
                tur_clear += pat

            for hamle in hamle_dizisi:
                hamleyi_uygula(hamle, eldeki_taslar)

            if tur_clear > 0:
                combo += tur_clear
                print(f"  Clear! (+{tur_clear}) -> Combo: {combo}")
            else:
                if combo > 0:
                    print(f"  Clear YOK - Combo sifirlandir ({combo} -> 0)")
                combo = 0

            tur += 1
            print("  Yeni taslar bekleniyor...")
            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\nCtrl+C ile durduruldu.")
