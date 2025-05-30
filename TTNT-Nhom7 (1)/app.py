from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Dữ liệu huấn luyện mẫu [nhiệt độ, độ ẩm, áp suất khí quyển, tốc độ gió]
X_train = np.array([
    [30, 70, 1006, 15],  # mưa
    [32, 65, 1004, 12],  # mưa
    [27, 90, 1012, 20],  # mưa
    [25, 85, 1013, 18],  # mưa
    [35, 50, 1001, 10],  # không mưa
    [38, 45, 998, 8],    # không mưa
    [22, 96, 1015, 22],  # mưa
    [28, 80, 1010, 17],  # mưa
    [33, 60, 1002, 10],  # không mưa
    [36, 40, 997, 9],    # không mưa
    [29, 78, 1011, 19],  # mưa
    [31, 63, 1007, 13],  # mưa
    [39, 42, 995, 7],    # không mưa
    [21, 99, 1017, 24],  # mưa
    [34, 55, 1000, 11],  # không mưa
    # Thêm các trường hợp chồng lấn
    [30, 65, 1003, 12],  # không mưa
    [30, 65, 1008, 16],  # mưa
    [33, 70, 1010, 15],  # mưa
    [33, 70, 1003, 11],  # không mưa
    [28, 55, 1005, 14],  # không mưa
    [28, 55, 1009, 18],  # mưa
    [35, 78, 1012, 13],  # mưa
    [35, 78, 1001, 9],   # không mưa
    [25, 50, 1002, 16],  # không mưa
    [25, 50, 1007, 20],  # mưa
])
y_train = np.array([
    1, 1, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 0, 1, 0,
    # các trường hợp bổ sung
    0, 1, 1, 0, 0, 1, 1, 0, 0, 1
])

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-6  # tránh chia cho 0
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_prob(self, x, mean, var):
        coef = 1.0 / np.sqrt(2 * np.pi * var)
        exp = np.exp(- (x - mean) ** 2 / (2 * var))
        return coef * exp

    def predict_proba(self, X):
        probs = []
        for x in X:
            class_probs = []
            for c in self.classes:
                prior = np.log(self.priors[c])
                conditional = np.sum(np.log(self.gaussian_prob(x, self.means[c], self.vars[c])))
                class_probs.append(prior + conditional)
            # Chuyển log prob sang xác suất thường
            max_log = np.max(class_probs)
            exp_probs = np.exp(class_probs - max_log)
            norm_probs = exp_probs / np.sum(exp_probs)
            probs.append(norm_probs)
        return np.array(probs)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

# Khởi tạo và huấn luyện mô hình Naive Bayes
model = NaiveBayes()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('weather.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        pressure = float(data['pressure'])
        wind = float(data['wind'])
    except (KeyError, ValueError):
        return jsonify({'error': 'Thiếu hoặc sai định dạng thông tin đầu vào!'}), 400

    X = np.array([[temp, humidity, pressure, wind]])
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
    app.run(debug=True)