from flask import Flask, request, jsonify
from sklearn.naive_bayes import GaussianNB
import numpy as np

app = Flask(__name__)

# Dữ liệu huấn luyện mẫu (có thể thay đổi/ mở rộng theo thực tế)
# Các đặc trưng: [temperature, humidity, sunny, wind]
# sunny: 1 = Có, 0 = Không
# Nhãn: 1 = Mưa, 0 = Không mưa
X_train = np.array([
    [30, 70, 1, 15],   # mưa
    [32, 65, 1, 12],   # mưa
    [27, 90, 0, 20],   # mưa
    [25, 85, 0, 18],   # mưa
    [35, 50, 1, 10],   # không mưa
    [38, 45, 1, 8],    # không mưa
    [22, 96, 0, 22],   # mưa
    [28, 80, 0, 17],   # mưa
    [33, 60, 1, 10],   # không mưa
    [36, 40, 1, 9],    # không mưa
    [29, 78, 0, 19],   # mưa
    [31, 63, 1, 13],   # mưa
    [39, 42, 1, 7],    # không mưa
    [21, 99, 0, 24],   # mưa
    [34, 55, 1, 11],   # không mưa
])
y_train = np.array([
    1, 1, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 0, 1, 0
])

# Khởi tạo và huấn luyện mô hình Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        sunny = int(data['sunny'])
        wind = float(data['wind'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Thiếu hoặc sai định dạng thông tin đầu vào!'}), 400

    X = np.array([[temp, humidity, sunny, wind]])
    prediction = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    response = {
        'prediction': prediction,
        'probability': {
            'rain': float(proba[1]),
            'no_rain': float(proba[0])
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)