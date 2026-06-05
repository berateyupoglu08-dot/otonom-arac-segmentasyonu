from ultralytics import YOLO
import cv2

# YOLOv8 Segmentasyon modelini indir ve yükle
model = YOLO('yolov8n-seg.pt') 

video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

# Video özelliklerini al
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Çıktı videosunu kaydetmek için ayarlar
out = cv2.VideoWriter('ciktili_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

ekran_alani = width * height
frame_sayisi = 0

print("Video işleniyor, lütfen bekleyin...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_sayisi += 1
    
    # Model ile segmentasyon tahmini yap
    results = model(frame, verbose=False)
    
    # Segmentasyon maskelerini görselleştir (Yoldaki engelleri boyar)
    annotated_frame = results[0].plot()
    
    # ---- KARAR DESTEK SİSTEMİ (KDS) MANTIĞI ----
    karar = "GUVENLI SEYIR"
    renk = (0, 255, 0) # Yeşil
    
    # Algılanan nesneleri kontrol et
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        
        # Eğer insan, araba, kamyon veya bisiklet tespit edildiyse
        if label in ['person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle']:
            # Nesnenin koordinatları ve kapladığı alanı hesapla
            x1, y1, x2, y2 = box.xyxy[0]
            alan = (x2 - x1) * (y2 - y1)
            
            # Engel ekranda %5'ten fazla yer kaplıyorsa (Bize Çok Yakınsa)
            if alan / ekran_alani > 0.05:
                karar = "ACIL FREN!"
                renk = (0, 0, 255) # Kırmızı
                break # En riskli durumu bulduk, diğerlerine bakmaya gerek yok
                
            # Engel var ama uzaktaysa (Ekranda %1 ile %5 arası yer kaplıyorsa)
            elif alan / ekran_alani > 0.01:
                if karar != "ACIL FREN!": # Öncelik daima Acil Frende olmalı
                    karar = "SURUCUYU UYAR!"
                    renk = (0, 165, 255) # Turuncu
    
    # Kararı videonun üzerine yazdır
    cv2.putText(annotated_frame, f"KARAR DESTEK: {karar}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 3)
                
    # İşlenmiş kareyi kaydet
    out.write(annotated_frame)
    
cap.release()
out.release()
print("Mükemmel! İşlem Tamamlandı!")
