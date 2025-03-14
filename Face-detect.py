import cv2 as cv

# โหลดรูปภาพ
img = cv.imread('8arm.jpg')
if img is None:
    print("Error: ไม่พบไฟล์ภาพ '8arm.jpg'")
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



# แสดงผลลัพธ์
cv.imshow('Image', img)

# ใช้ waitKey() ให้แน่ใจว่าหน้าต่างจะไม่ปิดเอง
key = cv.waitKey(0)  # รอให้ผู้ใช้กดปุ่ม
if key == 27:  # ถ้ากด ESC -> ปิดหน้าต่าง
    cv.destroyAllWindows()
