import easyocr
import cv2

class OCREngine:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read(self, img):
        if img is None or img.size == 0:
            return ""

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray)

        text = ""
        for (_, t, conf) in results:
            if conf > 0.6:
                text += t
        return text.strip()
