import cv2
import numpy as np
from tensorflow.keras.models import load_model

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Input, Flatten, Dense

# model = Sequential()
# model.add(Input(shape=(28, 28, 1)))
# model.add(Flatten())
# model.add(Dense(64, activation="relu"))
# model.add(Dense(32, activation="relu"))
# model.add(Dense(10, activation="softmax"))

model = load_model("./Model/my_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Make a prediction
    prediction = model.predict(reshaped)
    predicted_digit = np.argmax(prediction)

    # Display the result
    cv2.putText(
        frame,
        f"Predicted Digit: {predicted_digit}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
