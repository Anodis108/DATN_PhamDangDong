from __future__ import annotations

import os

import cv2
from network.call_api import APICaller  # Sửa lại nếu bạn đặt tên khác

# 🛠️ Cấu hình
# ✅ Sửa đường dẫn folder ảnh của bạn
folder_path = '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_PhamDangDong/DATN_PhamDangDong/resource/data/data/processed_data'
api_url = 'http://localhost:5001/v1/height'  # ✅ Sửa URL API phù hợp

# 🖼️ Hỗ trợ các đuôi ảnh phổ biến
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

# 🧠 Hàm kiểm tra ảnh


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


# 📂 Duyệt ảnh trong thư mục
for file_name in os.listdir(folder_path):
    if not is_image_file(file_name):
        continue

    file_path = os.path.join(folder_path, file_name)
    image = cv2.imread(file_path)

    if image is None:
        print(f'❌ Không thể đọc ảnh: {file_path}')
        continue

    print(f'📤 Đang gửi ảnh: {file_name}')
    api_result = APICaller.call_api(api_url, image, file_name)

    if api_result is None:
        print(f'⚠️  Lỗi API với ảnh: {file_name}')
    else:
        print(f'✅  Kết quả từ API cho {file_name}:')
        print(f'  - Chiều cao dự đoán: {api_result.heights}')
        print(f'  - Ảnh đã xử lý lưu tại: {api_result.out_path}')
