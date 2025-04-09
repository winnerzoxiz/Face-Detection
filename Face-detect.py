import cv2 as cv

# โหลดรูปภาพ
img = cv.imread('Me-2.jpg')
if img is None:
    print("Error: ไม่พบไฟล์ภาพ 'Me-2.jpg'")
    exit()

# โหลดโมเดลตรวจจับใบหน้า
face_model = cv.CascadeClassifier('Face-detect-model.xml')
if face_model.empty():
    print("Error: ไม่พบไฟล์โมเดล 'Face-detect-model.xml'")
    exit()

# แปลงเป็นภาพขาวดำ
gray_scale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# ตรวจจับใบหน้า
faces = face_model.detectMultiScale(gray_scale)

# วาดกรอบรอบใบหน้าที่ตรวจจับได้
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

# ย่อภาพก่อนแสดง (ขนาดหน้าต่างที่เหมาะสม)
resized_img = cv.resize(img, (800, 800))

# แสดงผลลัพธ์
cv.imshow('Image', resized_img)

# รอให้ผู้ใช้กดปุ่มเพื่อปิด
key = cv.waitKey(0)
if key == 27:
    cv.destroyAllWindows()
