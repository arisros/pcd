import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Load Image
# =========================
# change the image_path to test with different images
image_path = "lantai.jpg"
img = cv2.imread(image_path)

if img is None:
    raise ValueError("Gambar tidak ditemukan!")

# =========================
# 2. Grayscale 
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================
# 3. Gaussian Blur
# =========================
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# =========================
# 4. Extract Statistic Feature
# =========================
mean_intensity = np.mean(blur)
std_intensity = np.std(blur)

# =========================
# 5. Threshold Decision
# =========================

THRESHOLD_STD = 20  # can be adjust data from test

if std_intensity > THRESHOLD_STD:
    status = "Lantai Kotor - Perlu Dipel"
    color = (0, 0, 255)  # red mark
else:
    status = "Lantai Bersih"
    color = (0, 255, 0)  # green mark

# =========================
# 6. Result
# =========================
output = img.copy()
cv2.putText(output, status, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# =========================
# 7. Visualization
# =========================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Citra Asli")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Grayscale + Blur")
plt.imshow(blur, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Hasil Deteksi")
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()

# =========================
# 8. Print Statistic
# =========================
print("Mean Intensitas      :", round(mean_intensity, 2))
print("Standar Deviasi      :", round(std_intensity, 2))
print("Keputusan Sistem     :", status)
