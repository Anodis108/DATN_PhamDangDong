from __future__ import annotations

import numpy as np
from PIL import Image
# Đọc ảnh bằng PIL
image = Image.open(
    '/home/anodi108/Desktop/project/Do_An_Tot_Nghiep/DATN_LuThiSen/src/model_deployed/text_detector_preproces.png',
)
image_np = np.array(image)  # Chuyển thành mảng NumPy

# Lưu dưới dạng .npy
np.save('demo_card.npy', image_np)
