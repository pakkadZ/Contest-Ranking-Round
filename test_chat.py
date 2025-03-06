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
    test_im_1 = cv2.imread(img1_path)
    test_im_1 = cv2.cvtColor(test_im_1, cv2.COLOR_BGR2RGB)
    test_im_1 = cv2.resize(test_im_1, IMAGE_SIZE)
    test_im_1 = test_im_1 / 255.0
    test_im_1 = np.expand_dims(test_im_1, axis=0)

    test_im_2 = cv2.imread(img2_path)
    test_im_2 = cv2.cvtColor(test_im_2, cv2.COLOR_BGR2RGB)
    test_im_2 = cv2.resize(test_im_2, IMAGE_SIZE)
    test_im_2 = test_im_2 / 255.0
    test_im_2 = np.expand_dims(test_im_2, axis=0)

    y_pred = model.predict([test_im_1, test_im_2])
    return y_pred[0][0]

# โหลดข้อมูล
test_data = pd.read_csv('D:\\tenserflow\\testset\\test.csv')

# ลูปเพื่ออัปเดตค่าในคอลัมน์ Winner
for index, row in test_data.iterrows():
    img1 = 'D:\\tenserflow\\testset\\' + row['Image 1']
    img2 = 'D:\\tenserflow\\testset\\' + row['Image 2']
    accuracy = test_image_pair(img1, img2)
    test_data.at[index, "Winner"] = 1 if accuracy > 0.5 else 2

# บันทึกกลับไปที่ไฟล์ CSV
test_data.to_csv('D:\\tenserflow\\testset\\test.csv', index=False)

print("✅ อัปเดตคอลัมน์ Winner เรียบร้อยแล้ว!")
