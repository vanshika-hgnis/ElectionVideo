import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# CONFIG
image_path = "frame.png"
target_text = "ब्रजेश"
replacement_text = "वंशिका"
font_path = "font/Mukta-Bold.ttf"

# (Optional) Only for Windows users
pytesseract.pytesseract.tesseract_cmd = 'D:/C-Drive/Tesseract-OCR/tesseract.exe'

# STEP 1: Load Image
image_cv = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

# STEP 2: OCR - Hindi Text Detection
ocr_data = pytesseract.image_to_data(image_rgb, lang='hin', config='--psm 6', output_type=pytesseract.Output.DICT)

# STEP 3: Search for Target Word
box = None
for i, word in enumerate(ocr_data['text']):
    if target_text in word:
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        box = (x, y, w, h)
        break

if not box:
    print("❌ Target text not found via OCR.")
    exit()

x, y, w, h = box

# STEP 4: Background Patch Sampling + Blending
patch = image_cv[y:y+h, x:x+w].copy()
blurred_patch = cv2.GaussianBlur(patch, (7, 7), 0)
image_cv[y:y+h, x:x+w] = blurred_patch

# STEP 5: Text Rendering (PIL)
image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)

# Helper: Best font fit
def get_best_fit_font(text, box_w, box_h, font_path):
    font_size = 10
    while True:
        font = ImageFont.truetype(font_path, font_size)
        bbox = font.getbbox(text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w >= box_w or text_h >= box_h:
            break
        font_size += 1
    return ImageFont.truetype(font_path, font_size - 1)

font = get_best_fit_font(replacement_text, w, h, font_path)

# STEP 6: Center Text Precisely (baseline aware)
tb = draw.textbbox((0, 0), replacement_text, font=font)
tw, th = tb[2] - tb[0], tb[3] - tb[1]
tx = x + (w - tw) // 2
ty = y + (h - th) // 2 - tb[1]  # align to baseline

draw.text((tx, ty), replacement_text, font=font, fill=(62, 198, 255))  # sky blue

# STEP 7: Save & Show
final_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
cv2.imshow("Modified", final_image)
cv2.imwrite("output_ocr_modified.jpg", final_image)
print("✅ Output saved to output_ocr_modified.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
