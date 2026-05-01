import keyboard
import mss
import numpy as np
import cv2
import time
import pyautogui
import itertools
# Farenin çok ani tepkiler verip sistemi kilitlememesi için minik bir güvenlik molası
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True # Fareyi köşeye hızla çekersen program durur

# --- KOORDİNATLAR VE SABİTLER ---
BLOCK_SIZE = 33.0

PIECES_ROI = {
    'top': 1026,
    'left': 976,
    'width': 611,
    'height': 227
}

# LÜTFEN BURAYA KENDİ OYUN TAHTASI KOORDİNATLARINI GİR:
BOARD_ROI = {
    'top': 392,    # Sol üst Y koordinatı
    'left': 974,   # Sol üst X koordinatı
    'width': 614,  # Genişlik
    'height': 608 # Yükseklik
}
# ==========================================
# 1. GÖRÜNTÜ İŞLEME (GÖZLER)
# ==========================================

def tahtayi_oku(roi):
    with mss.MSS() as sct:
        ekran = np.array(sct.grab(roi))
        gri = cv2.cvtColor(ekran, cv2.COLOR_BGRA2GRAY)
        
        h, w = gri.shape
        hucre_h = h // 8
        hucre_w = w // 8
        
        tahta = np.zeros((8, 8), dtype=int)
        for r in range(8):
            for c in range(8):
                merkez_piksel = gri[r*hucre_h + hucre_h//2, c*hucre_w + hucre_w//2]
                if merkez_piksel > 50: # Parlaklık 50'den büyükse blok var
                    tahta[r, c] = 1
        return tahta

def taslari_oku(roi):
    with mss.MSS() as sct:
        ekran = np.array(sct.grab(roi))
        bgr = ekran[:, :, :3] # BGRA'dan alpha (şeffaflık) kanalını at
        
        # Tepsindeki arkaplan rengini otomatik tespit et (En yaygın piksel rengi)
        pixels = bgr.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        bg_color = unique_colors[np.argmax(counts)]
        
        # Her bir pikselin arkaplan ile olan renk farkını (mesafesini) hesapla
        diff = cv2.absdiff(bgr, bg_color)
        sum_diff = np.sum(diff, axis=2)
        
        # Renk farkı 30'dan büyükse (yani arkaplan değilse) o bir taştır (Beyaz yap)
        maske = (sum_diff > 30).astype(np.uint8) * 255
        
        parca_genislik = roi['width'] // 3
        maske_taslar = [
            maske[:, 0 : parca_genislik],
            maske[:, parca_genislik : parca_genislik * 2],
            maske[:, parca_genislik * 2 : roi['width']]
        ]
        
        tum_taslar = []
        for i in range(3):
            noktalar = cv2.findNonZero(maske_taslar[i])
            if noktalar is not None:
                x, y, w, h = cv2.boundingRect(noktalar)
                
                # Ufak tefek görüntü parazitlerini yoksay
                if w < 15 or h < 15:
                    tum_taslar.append(None)
                    continue
                
                cols = max(1, int(round(w / BLOCK_SIZE)))
                rows = max(1, int(round(h / BLOCK_SIZE)))
                
                kirpilmis_maske = maske_taslar[i][y:y+h, x:x+w]
                tas_matrisi = np.zeros((rows, cols), dtype=int)
                
                for r in range(rows):
                    for c in range(cols):
                        cy = int((r + 0.5) * (h / rows))
                        cx = int((c + 0.5) * (w / cols))
                        if kirpilmis_maske[cy, cx] > 0:
                            tas_matrisi[r][c] = 1
                            
                tum_taslar.append(tas_matrisi)
            else:
                tum_taslar.append(None) # Yer boş
                
        return tum_taslar

# ==========================================
# 2. HATA AYIKLAMA (DEBUGGING) EKRANLARI
# ==========================================
def debug_tahta(tahta):
    print("\n--- GÜNCEL TAHTA (BOTUN GÖZÜNDEN) ---")
    for r in range(8):
        satir = ""
        for c in range(8):
            if tahta[r, c] == 1:
                satir += "[■] "
            else:
                satir += "[ ] "
        print(satir)
    print("---------------------------------------")

def debug_taslar(taslar):
    print("\n--- ELDEKİ TAŞLAR (BOTUN GÖZÜNDEN) ---")
    for i, tas in enumerate(taslar):
        if tas is None:
            print(f"{i+1}. Taş: BOŞ")
            continue
        print(f"{i+1}. Taş:")
        for r in range(tas.shape[0]):
            satir = ""
            for c in range(tas.shape[1]):
                if tas[r, c] == 1:
                    satir += "■ "
                else:
                    satir += "  "
            print(satir)

# ==========================================
# 3. OYUN MOTORU & YAPAY ZEKA (BEYİN)
# ==========================================

# 1. OPTİMİZE YERLEŞEBİLİR Mİ (Daha Hızlı Çarpışma Kontrolü)
def yerlesebilir_mi(tahta, tas, r, c):
    tr, tc = tas.shape
    # Sınır kontrolü döngüden çıkarıldı. 
    # Sadece çakışma var mı diye bakıyoruz. Bitwise AND (&) işlemi en hızlı yöntemdir.
    return not (tahta[r:r+tr, c:c+tc] & tas).any()

# (İsteğe Bağlı Minik Hızlandırma) tasi_koy fonksiyonunu da böyle güncelle
def tasi_koy(tahta, tas, r, c):
    yeni = tahta.copy()
    tr, tc = tas.shape
    yeni[r:r+tr, c:c+tc] |= tas # Bitwise OR kullanarak yerleştirme
    return yeni
def temizle(tahta):
    satirlar = np.all(tahta == 1, axis=1)
    sutunlar = np.all(tahta == 1, axis=0)
    patlama = np.sum(satirlar) + np.sum(sutunlar)
    
    if patlama > 0:
        tahta[satirlar, :] = 0
        tahta[:, sutunlar] = 0
    return tahta, patlama

# 2. OPTİMİZE PUANLA (Patlama puanı algoritmadan bağımsızlaştırıldı)
def puanla(tahta):
    # Toplam patlamayı sildik, sadece tahtanın anlık "şekline" not veriyoruz.
    puan = -np.sum(tahta) * 10
    
    # Delik (Hole) cezası
    for r in range(8):
        for c in range(8):
            if tahta[r, c] == 0:
                komsular = 0
                if r > 0 and tahta[r-1, c] == 1: komsular += 1
                if r < 7 and tahta[r+1, c] == 1: komsular += 1
                if c > 0 and tahta[r, c-1] == 1: komsular += 1
                if c < 7 and tahta[r, c+1] == 1: komsular += 1
                if komsular == 3: puan -= 50
                elif komsular == 4: puan -= 100
    return puan

# 3. YENİ BEYİN DFS (Hafızalama ve Alan Daraltma İçerir)
def dfs(tahta, kalan_sira, eldeki_taslar, cache):
    # Bu tahta dizilimini ve elimizdeki kalan taşları bir "Anahtar (Key)" yapıyoruz.
    state_key = (tahta.tobytes(), tuple(kalan_sira))
    
    # Eğer bu senaryoyu daha önce hesapladıysak, amelelik yapma! Direkt hafızadan çek.
    if state_key in cache:
        return cache[state_key]

    # Eğer yerleştirilecek taş kalmadıysa, tahtanın son halini puanla ve bitir.
    if len(kalan_sira) == 0:
        sonuc = (puanla(tahta), [])
        cache[state_key] = sonuc
        return sonuc

    en_iyi_puan = -999999
    en_iyi_yol = None
    
    tas_idx = kalan_sira[0]
    tas = eldeki_taslar[tas_idx]
    yerlesti = False
    
    tas_satir, tas_sutun = tas.shape

    # BÜYÜK OPTİMİZASYON: Taşı sadece sığabileceği kadar aralıkta gezdiriyoruz! (Örn: 8x8 yerine 6x6)
    for r in range(9 - tas_satir):
        for c in range(9 - tas_sutun):
            if yerlesebilir_mi(tahta, tas, r, c):
                yerlesti = True
                yeni_tahta = tasi_koy(tahta, tas, r, c)
                yeni_tahta, patlama = temizle(yeni_tahta)
                
                # Derine in, puanı al
                alt_puan, alt_yol = dfs(yeni_tahta, kalan_sira[1:], eldeki_taslar, cache)
                
                # Patlama ödülü bağımsız eklendiği için "aynı tahta her zaman aynı puanı verir" ilkesi korundu.
                guncel_hamle_puani = alt_puan + (patlama * 1000)
                
                if alt_yol is not None and guncel_hamle_puani > en_iyi_puan:
                    en_iyi_puan = guncel_hamle_puani
                    en_iyi_yol = [{'tas_idx': tas_idx, 'r': r, 'c': c}] + alt_yol

    # Eğer sığmadıysa game over ihtimalidir
    if not yerlesti:
        cache[state_key] = (-999999, None)
        return -999999, None

    # Bulduğumuz bu en mükemmel yolu hafızaya kaydet
    cache[state_key] = (en_iyi_puan, en_iyi_yol)
    return en_iyi_puan, en_iyi_yol

# 4. EN İYİ DİZİYİ BUL (Ortak Hafıza Başlatıcısı)
def en_iyi_diziyi_bul(tahta, taslar):
    gecerli_idx = [i for i, t in enumerate(taslar) if t is not None]
    if not gecerli_idx:
        return []
        
    siralar = list(itertools.permutations(gecerli_idx))
    en_iyi_puan = -999999
    en_iyi_dizi = None
    
    # Tüm permütasyonların faydalanacağı ORTAK HAFIZA
    ortak_cache = {}
    
    for sira in siralar:
        # Eski koddaki 0 (sıfır) silindi, yerine ortak_cache eklendi
        puan, yol = dfs(tahta, sira, taslar, ortak_cache)
        if yol is not None and puan > en_iyi_puan:
            en_iyi_puan = puan
            en_iyi_dizi = yol
            
    return en_iyi_dizi
# ==========================================
# 3. OTOMASYON (ELLER)
# ==========================================

def hamleyi_uygula(hamle, eldeki_taslar):
    tas_idx = hamle['tas_idx']
    hedef_r = hamle['r']
    hedef_c = hamle['c']
    
    tas = eldeki_taslar[tas_idx]
    tas_satir, tas_sutun = tas.shape
    
    # 1. 1x1 Blok ile kaydettiğin KESİN Başlangıç (Tutma) Koordinatları
    baslangic_taslar_x = [1082, 1277, 1473]
    baslangic_taslar_y = [1164, 1164, 1165]
    
    baslangic_x = baslangic_taslar_x[tas_idx]
    baslangic_y = baslangic_taslar_y[tas_idx]
    
    # ========================================================
    # KUSURSUZ 1x1 KALİBRASYON HARİTASI
    # Her slotun kendine has yatay/dikey kaymasını otomatik düzeltir!
    # ========================================================
    haritalar = [
        {'min_x': 1038, 'max_x': 1409, 'min_y': 761, 'max_y': 1133}, # 1. Taş (Sol)
        {'min_x': 1092, 'max_x': 1463, 'min_y': 761, 'max_y': 1133}, # 2. Taş (Orta)
        {'min_x': 1143, 'max_x': 1517, 'min_y': 763, 'max_y': 1133}  # 3. Taş (Sağ)
    ]
    
    harita = haritalar[tas_idx]
    
    # 8x8 tahtada 0'dan 7'ye toplam 7 birim aralık vardır
    adim_x = (harita['max_x'] - harita['min_x']) / 7.0
    adim_y = (harita['max_y'] - harita['min_y']) / 7.0
    
    # Senin mükemmel merkez hesaplama formülün (Artık 1x1 haritaya oturduğu için HATA PAYI 0)
    merkez_c = hedef_c + (tas_sutun - 1) / 2.0
    merkez_r = hedef_r + (tas_satir - 1) / 2.0
   
    gerçek_hedef_x = int(harita['min_x'] + (merkez_c * adim_x))
    gerçek_hedef_y = int(harita['min_y'] + (merkez_r * adim_y))
    
    print(f"\n[*] {tas_idx + 1}. Taş ({hedef_r + 1}, {hedef_c + 1}) hücresine taşınıyor...")
    print(f"[+] Hedef Ekran Pikselleri -> X: {gerçek_hedef_x}, Y: {gerçek_hedef_y}")
    
    # GÜNCEL SÜRÜKLEME STRATEJİSİ
    # Farenin scrcpy'yi kilitlememesi için ufak delaylar hayat kurtarır
    pyautogui.moveTo(baslangic_x, baslangic_y)
    pyautogui.mouseDown(button='left') 
    #time.sleep(0.3) # Animasyonun fırlamasını bekle
    
    # Hedefe sürükle (duration'u 0.4 civarı tutmak dokunmatiğin algılaması için iyidir)
    pyautogui.moveTo(gerçek_hedef_x, gerçek_hedef_y, duration=0.2) 
    
    # Bırak ve etabın oturmasını bekle
    pyautogui.mouseUp(button='left')
    time.sleep(0.5)
# ==========================================
# ANA DÖNGÜ
# ==========================================

if __name__ == "__main__":
    print("===========================================")
    print("🤖 BLOCK BLAST OTONOM BOTU BAŞLATILIYOR 🤖")
    print("===========================================")
    print("Lütfen oyunu ekrana getirin. 3 saniye içinde başlıyor...")
    time.sleep(3)
    
    tur_sayaci = 1
    try:
        while True:
            if keyboard.is_pressed('space'):
                time.sleep(5)
                print("bekleniyor")
            print(f"\n--- TUR {tur_sayaci} ---")
            
            if keyboard.is_pressed('esc'):
                break
            print(f"\n--- TUR {tur_sayaci} ---")
            
            # 1. Ekranı oku
            guncel_tahta = tahtayi_oku(BOARD_ROI)
            eldeki_taslar = taslari_oku(PIECES_ROI)
            
            # Debugging (Hata Ayıklama) görüntülerini terminale bas
            debug_tahta(guncel_tahta)
            debug_taslar(eldeki_taslar)
            
            # Eğer ekranda taş yoksa oyun animasyon oynatıyordur, bekle ve tekrar dene
            if all(tas is None for tas in eldeki_taslar):
                time.sleep(1)
                continue
                
            # 2. Düşün
            print("\n⏳ Yapay Zeka olası tüm kombinasyonları (DFS) hesaplıyor... (Bu biraz sürebilir)")
            hamle_dizisi = en_iyi_diziyi_bul(guncel_tahta, eldeki_taslar)
            if not hamle_dizisi:
                print("❌ Olası Game Over veya Tema Değişikliği! Tema düzeltilmesi deneniyor...")
                
                # Tema değiştirme tıklamaları
                pyautogui.click(x=1553, y=166)
                time.sleep(0.5)
                pyautogui.click(x=1256, y=1112)
                time.sleep(0.5)
                pyautogui.click(x=1518, y=347)
                
                # Temanın değişmesi ve tahtanın ekrana gelmesi için bekle
                time.sleep(2) 
                
                # Ekranı yeni temayla tekrar oku
                guncel_tahta = tahtayi_oku(BOARD_ROI)
                eldeki_taslar = taslari_oku(PIECES_ROI)
                
                # Eğer tema değişirken oyun animasyona girdiyse bekle
                if all(tas is None for tas in eldeki_taslar):
                    time.sleep(1)
                    eldeki_taslar = taslari_oku(PIECES_ROI)
                    
                # Yeniden hamle bulmayı dene
                hamle_dizisi = en_iyi_diziyi_bul(guncel_tahta, eldeki_taslar)
                
                # EĞER HALA HAMLE YOKSA O ZAMAN GERÇEKTEN GAME OVER!
                if not hamle_dizisi:
                    print("❌ GAME OVER! Oynanacak geçerli bir hamle gerçekten kalmadı.")
                    break
                else:
                    print("✅ Tema düzeltildi! Hamle bulundu, devam ediliyor.")
                
    
                
            # 3. Oyna
            for hamle in hamle_dizisi:
                hamleyi_uygula(hamle, eldeki_taslar)
                
            tur_sayaci += 1
            
            # Oyunun yeni taşları vermesi ve tahtanın oturması için bekle
            print("[*] Yeni taşların gelmesi bekleniyor...")

            time.sleep(1.5)
    except ImportError:
        print("HATA VAR KRAL")

        