import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Config ---
image_path = "frame.png"
# replacement_text = "वंशिका जी"
replacement_text = "वंशिका"
font_path = "font/Mukta-Bold.ttf"  # Ensure this font supports Hindi

# --- Load Image ---
image = cv2.imread(image_path)
clone = image.copy()
drawing = False
start_point = None
end_point = None

# --- Font Size Detection Helper ---
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

# --- Draw Replacement Text in Selected Box ---
def replace_text_box(image_cv, box):
    (x1, y1), (x2, y2) = box
    x, y = min(x1, x2), min(y1, y2)
    w, h = abs(x2 - x1), abs(y2 - y1)

    # Step 1: Sample center background color
    center_color = image_cv[y + h // 2, x + w // 2].tolist()
    cv2.rectangle(image_cv, (x, y), (x + w, y + h), center_color, thickness=-1)

    # Step 2: Convert to PIL
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Step 3: Auto-fit font
    font = get_best_fit_font(replacement_text, w, h, font_path)

    # Step 4: Get precise bounding box
    bbox = draw.textbbox((0, 0), replacement_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Step 5: Final position with vertical correction (baseline shift)
    tx = x + (w - text_w) // 2
    ty = y + (h - text_h) // 2 - bbox[1]  # shift to baseline

    # Step 6: Draw text
    draw.text((tx, ty), replacement_text, font=font, fill=(62, 198, 255))  # "#3ec6ff"

    # Step 7: Save
    final_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Modified", final_image)
    cv2.imwrite("output_modified.jpg", final_image)
    print("✅ Aligned and saved to output_modified.jpg")
    return final_image




# --- Mouse Callback for Drawing Box ---
def mouse_handler(event, x, y, flags, param):
    global drawing, start_point, end_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = clone.copy()
        cv2.rectangle(temp, start_point, (x, y), (255, 0, 255), 2)
        cv2.imshow("Draw box to replace", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(clone, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Draw box to replace", clone)
        replace_text_box(clone.copy(), (start_point, end_point))

# --- Launch Window ---
cv2.imshow("Draw box to replace", clone)
cv2.setMouseCallback("Draw box to replace", mouse_handler)
cv2.waitKey(0)
cv2.destroyAllWindows()


