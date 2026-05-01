import pyautogui
import keyboard
import time

print("===========================================")
print("📍 KOORDİNAT BULUCU BAŞLATILDI 📍")
print("===========================================")
print("Fareyi ekranda istediğiniz yere götürün ve 'SPACE' (Boşluk) tuşuna basın.")
print("Uygulamadan çıkmak için 'ESC' tuşuna basabilirsiniz.\n")

try:
    while True:
        if keyboard.is_pressed('space'):
            x, y = pyautogui.position()
            print(f"Kaydedildi -> X: {x}, Y: {y}")
            
            # Tuşa basılı tutulduğunda aynı koordinatı yüzlerce kez yazmaması için
            # boşluk tuşu bırakılana kadar bekle:
            while keyboard.is_pressed('space'):
                time.sleep(0.01)
                
        elif keyboard.is_pressed('esc'):
            print("\nÇıkış yapılıyor...")
            break
            
        time.sleep(0.01)
except ImportError:
    print("\n[HATA] 'keyboard' kütüphanesi eksik!")
    print("Lütfen terminale şunu yazıp kurun: pip install keyboard")