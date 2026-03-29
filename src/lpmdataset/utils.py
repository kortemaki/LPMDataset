import pytesseract
from stop_words import get_stop_words

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

STOPWORDS = set(w for w in get_stop_words('en') if w != 'microsoft')  # I don't think microsoft is a stop word, personally!
