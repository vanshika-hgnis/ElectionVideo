import cv2
import pytesseract

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"D:\C-Drive\Tesseract-OCR\tesseract.exe"

# Load image
img = cv2.imread("rframe.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clone = img.copy()

# OCR data
# 🧠 Get OCR data with Hindi language enabled
data = pytesseract.image_to_data(gray, lang='hin', output_type=pytesseract.Output.DICT)


# Variables for selection
start_point = None
end_point = None
selecting = False

# Mouse callback
def select_text(event, x, y, flags, param):
    global start_point, end_point, selecting, img

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        img = clone.copy()
        cv2.rectangle(img, start_point, (x, y), (255, 0, 255), 2)
        cv2.imshow("Drag to select word", img)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        selecting = False
        x1, y1 = start_point
        x2, y2 = end_point

        # Normalize coordinates
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)

        selected_words = []

        for i in range(len(data['text'])):
            bx = data['left'][i]
            by = data['top'][i]
            bw = data['width'][i]
            bh = data['height'][i]
            word = data['text'][i]

            if word.strip() == "":
                continue

            # Check if word box is inside selected area
            if (bx >= xmin and bx + bw <= xmax and
                by >= ymin and by + bh <= ymax):
                selected_words.append((word, bx, by, bw, bh))

        if selected_words:
            print("🟩 Selected words:")
            for word, bx, by, bw, bh in selected_words:
                print(f"  → '{word}' at ({bx},{by}) size {bw}x{bh}")
                # Draw box
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            cv2.imshow("Drag to select word", img)
        else:
            print("❌ No words found in selected area.")

# Show and wait
cv2.imshow("Drag to select word", img)
cv2.setMouseCallback("Drag to select word", select_text)
cv2.waitKey(0)
cv2.destroyAllWindows()
