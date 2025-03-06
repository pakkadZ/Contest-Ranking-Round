import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ขนาดของภาพที่โมเดลใช้
IMAGE_SIZE = (128, 128)

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = load_model('image_pair_classify.keras')

# ฟังก์ชันสำหรับการทดสอบภาพ
def test_image_pair(img1_path, img2_path):
    # อ่านภาพที่ 1
    test_im_1 = cv2.imread(img1_path)
    test_im_1 = cv2.cvtColor(test_im_1, cv2.COLOR_BGR2RGB)
    test_im_1 = cv2.resize(test_im_1, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im_1 = test_im_1 / 255.0  # Normalize
    test_im_1 = np.expand_dims(test_im_1, axis=0)  # เพิ่มมิติของ batch

    # อ่านภาพที่ 2
    test_im_2 = cv2.imread(img2_path)
    test_im_2 = cv2.cvtColor(test_im_2, cv2.COLOR_BGR2RGB)
    test_im_2 = cv2.resize(test_im_2, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    test_im_2 = test_im_2 / 255.0  # Normalize
    test_im_2 = np.expand_dims(test_im_2, axis=0)  # เพิ่มมิติของ batch

    # ทำนายผล
    y_pred = model.predict([test_im_1, test_im_2])
    return y_pred[0][0]  # คืนค่า accuracy (0 ถึง 1)

# โหลดข้อมูลจาก CSV
test_data = pd.read_csv('D:\\tenserflow\\testset\\test.csv')

# เพิ่มคอลัมน์ใหม่สำหรับผลลัพธ์
predicted_winners = []

# ลูปเพื่อทดสอบภาพทุกคู่ใน CSV
for index, row in test_data.iterrows():
    test_image_1 = 'D:\\tenserflow\\testset\\' + row['Image 1']
    test_image_2 = 'D:\\tenserflow\\testset\\' + row['Image 2']

    # ทำนาย
    accuracy = test_image_pair(test_image_1, test_image_2)

    # ตัดสินใจว่าโมเดลเลือกภาพไหน
    predicted_winner = "Image 1" if accuracy > 0.5 else "Image 2"
    predicted_winners.append(predicted_winner)

    # แสดงผล
    print(f"Predicted: {predicted_winner} for {test_image_1} and {test_image_2} (Score: {accuracy:.2f})")

# เพิ่มผลลัพธ์เข้าไปใน DataFrame
test_data["Predicted Winner"] = predicted_winners

# บันทึกกลับไปที่ไฟล์ CSV
test_data.to_csv('D:\\tenserflow\\testset\\test_results.csv', index=False)

print("✅ ผลลัพธ์ถูกบันทึกลงใน test_results.csv เรียบร้อยแล้ว!")
