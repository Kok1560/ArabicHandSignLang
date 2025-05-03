import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import arabic_reshaper
from bidi.algorithm import get_display


def draw_arabic_text(frame, text, position, font_size=48, color=(0, 0, 0)):
    # Reshape and prepare Arabic text
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)

    # Convert to PIL image
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    # Load a TTF font that supports Arabic
    font = ImageFont.truetype("arial.ttf", font_size)  # Make sure arial.ttf or another Arabic font exists

    # Draw the text
    draw.text(position, bidi_text, font=font, fill=color)

    # Convert back to OpenCV image
    return np.array(img_pil)


model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


labels_dict = {
    0: chr(1575),  # ا - Alif
    1: chr(1576),  # ب - Ba
    2: chr(1578),  # ت - Ta
    3: chr(1579),  # ث - Tha
    4: chr(1580),  # ج - Jeem
    5: chr(1581),  # ح - Hha
    6: chr(1582),  # خ - Kha
    7: chr(1583),  # د - Dal
    8: chr(1584),  # ذ - Dhal
    9: chr(1585),  # ر - Ra
    10: chr(1586), # ز - Zay
    11: chr(1587), # س - Seen
    12: chr(1588), # ش - Sheen
    13: chr(1589), # ص - Saad
    14: chr(1590), # ض - Daad
    15: chr(1591), # ط - Tta
    16: chr(1592), # ظ - Dha
    17: chr(1593), # ع - Ain
    18: chr(1594), # غ - Ghayn
    19: chr(1601), # ف - Fa
    20: chr(1602), # ق - Qaf
    21: chr(1603), # ك - Kaf
    22: chr(1604), # ل - Lam
    23: chr(1605), # م - Meem
    24: chr(1606), # ن - Noon
    25: chr(1607), # هـ - Ha
    26: chr(1608), # و - Waw
    27: chr(1610)  # ي - Ya
}
while True:

    data_aux = []
    x_ = []
    y_ = []


    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Use only the first hand

        # Extract features
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
            x_.append(x)
            y_.append(y)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Make prediction
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        print(predicted_character)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        frame = draw_arabic_text(frame, predicted_character, (x1, y1 - 50))




    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
