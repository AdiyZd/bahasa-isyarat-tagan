# pip install mediapipe opencv-python

# main
import cv2
import mediapipe as mp 
import numpy as np
import math


# connect ke cv2 camp
tangan_saya = mp.solutions.hands
gestur_gambar = mp.solutions.drawing_utils
gerakan_tangan = tangan_saya.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# data pola tangan 
Data = {
    'A' : [0, 1, 0, 0, 0],
    'B' : [1, 1, 1, 1, 1],
    'C' : [0, 1, 0, 1, 0],
    'D' : [0, 1, 1, 1, 0],
    'E' : [1, 1, 0, 0, 0],
    'F' : [0, 1, 1, 0, 0],
    'G' : [0, 1, 0, 1, 1],
    'H' : [1, 1, 1, 1, 0],
    'I' : [1, 0, 1, 0, 1],
    'J' : [0, 0, 1, 0, 1],
    'K' : [1, 1, 0, 0, 1],
    'L' : [1, 1, 1, 0, 0],
    'M' : [1, 1, 1, 1, 1],
    'N' : [1, 1, 1, 0, 0],
    'O' : [1, 1, 1, 1, 1],
    'P' : [0, 1, 0, 1, 1],
    'Q' : [1, 1, 1, 1, 0],
    'R' : [0, 1, 0, 1, 1],
    'S' : [0, 1, 0, 0, 1],
    'T' : [1, 0, 0, 0, 0],
    'U' : [0, 1, 1, 1, 0],
    'V' : [0, 1, 1, 0, 0],
    'W' : [1, 1, 1, 0, 0],
    'X' : [0, 1, 0, 0, 1],
    'Y' : [0, 1, 0, 0, 1],
    'Z' : [0, 1, 1, 0, 0],
}

# hitung jarak antar jari
def jari_ku_cok(jt1, jt2): # jt = jarak tangan
    return math.sqrt((jt2.x - jt1.x)**2 + (jt2.y - jt1.y)**2 + (jt2.z - jt1.z)**2)


# deteksi gerakan tangan 
def deteksi_postur_tangan(landmaks):
    jarak_antar_jari = []

    jari_jari = 0.07 # kasih jarak antar jar 7mm

    for i in [8, 12, 16, 20]: # id ujung jadi
        if landmaks[i].y < landmaks[i - 2].y:
            jarak_antar_jari.append(1)
        else:
            jarak_antar_jari.append(0)

    # jempol tangan / kan punyaku pendek
    jempol = 1 if landmaks[4].x < landmaks[3].x else 0
    jarak_antar_jari.insert(0, jempol)


    # cocokan data pola 
    for huruf, pola in Data.items():
        if jarak_antar_jari == pola:
            return huruf
    
    return None

# buka camp / kamera
camp = camp = cv2.VideoCapture(0)

while camp.isOpened():
    berhasil, frame = camp.read()
    if not berhasil:
        print("Kamera tidak dapat di acces silahkan coba lagi!")
        break

    frem = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # tangan 
    tangan_lo = gerakan_tangan.process(frem)

    # kondisi akhir 
    if tangan_lo.multi_hand_landmarks:
        for tangan_ku in tangan_lo.multi_hand_landmarks:
            # gambar tangan
            gestur_gambar.draw_landmarks(frame, tangan_ku, tangan_saya.HAND_CONNECTIONS)

            # pindah data ke array
            jarak_antar_jari = tangan_ku.landmark
            huruf_isyarat = deteksi_postur_tangan(jarak_antar_jari)
            
            if huruf_isyarat:
                cv2.putText(frame, f"Huruf: {huruf_isyarat}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # tampilkan hasil 
    cv2.imshow("Gestur tangan isyarat: ", frame)

    # tekan s untuk stop
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
# done
camp.release()
cv2.destroyWindow()