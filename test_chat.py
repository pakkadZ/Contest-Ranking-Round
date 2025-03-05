import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ขนาดของภาพที่โมเดลใช้
IMAGE_SIZE = (128, 128)

#Create Model
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
    return y_pred

# ฟังก์ชันเพื่อบันทึกผลการทดสอบ
def save_accuracy_to_file(img1_path, img2_path, accuracy, file_path='test_accuracy_results.txt'):
    with open(file_path, 'a') as f:
        f.write(f"Test images: {img1_path}, {img2_path} => Accuracy: {accuracy}\n")

# โหลดข้อมูลจาก CSV
test_data = pd.read_csv('D:\\tenserflow\\testset\\test.csv')

# ตัวแปรสำหรับคำนวณค่าเฉลี่ย
total_accuracy = 0
count = 0

# ลูปเพื่อทดสอบภาพทุกคู่ใน CSV
for index, row in test_data.iterrows():
    test_image_1 = 'D:\\tenserflow\\testset\\' + row['Image 1']
    test_image_2 = 'D:\\tenserflow\\testset\\' + row['Image 2']

    # ทำนายความแม่นยำ
    y_pred = test_image_pair(test_image_1, test_image_2)
    accuracy = y_pred[0][0]  # ค่าความแม่นยำจากโมเดล (0 ถึง 1)

    # แสดงผล
    print(f"Predicted Accuracy for {test_image_1} and {test_image_2}: {accuracy * 100:.2f}%")

    # บันทึกผลลัพธ์ในไฟล์
    save_accuracy_to_file(test_image_1, test_image_2, accuracy * 100)

    # รวมค่า accuracy สำหรับการคำนวณเฉลี่ย
    total_accuracy += accuracy
    count += 1

# คำนวณค่าเฉลี่ยของ accuracy
average_accuracy = total_accuracy / count
print(f"Average Accuracy: {average_accuracy * 100:.2f}%")
