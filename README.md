# Machine-learning-1
Motor Temperature Prediction (Predictive Maintenance)
Dự án này ứng dụng Machine Learning (XGBoost) để xây dựng hệ thống "Cảm biến ảo" (Virtual Sensing), dự báo chính xác nhiệt độ các thành phần bên trong motor (Nam châm, Stator) dựa trên các thông số vận hành thực tế. Hệ thống giúp chuyển đổi từ bảo trì định kỳ sang bảo trì dự báo, ngăn chặn rủi ro quá nhiệt và hỏng hóc thiết bị trong hệ thống robot/công nghiệp.

🚀 Tính năng nổi bật
Dự báo thời gian thực: Độ trễ (latency) chỉ dưới 50ms, đáp ứng yêu cầu giám sát tức thời.

Cảm biến ảo (Virtual Sensing): Ước tính nhiệt độ nội tại mà không cần gắn cảm biến vật lý đắt tiền trên Rotor đang quay.

Trực quan hóa: Dashboard Web App cho phép kỹ sư theo dõi thông số và trạng thái cảnh báo (An toàn/Cảnh báo/Nguy hiểm).

Học tập từ đặc tính nhiệt: Tích hợp kỹ thuật Rolling Mean để mô hình hiểu rõ đặc tính trễ nhiệt (thermal inertia) của động cơ.

🛠 Công nghệ sử dụng
Ngôn ngữ: Python 3.x

Machine Learning: XGBoost, Scikit-learn, Pandas, NumPy

Web Framework: Flask

Dữ liệu: Electric Motor Temperature Dataset (Kaggle)
📂 Cấu trúc dự án
Plaintext
├── models/             # Chứa file model (.joblib) và metadata
├── static/             # Chứa CSS, JS cho giao diện web
├── templates/          # HTML templates cho Flask
├── traindata.py        # Script huấn luyện mô hình & feature engineering
├── demo.py             # Script chạy Web App dự báo thời gian thực
├── requirements.txt    # Danh sách thư viện cần cài đặt
└── README.md
📋 Hướng dẫn cài đặt
Clone repository này về máy:

Bash
git clone [link-repo-cua-ban]
cd motor-temp-prediction
Cài đặt các thư viện cần thiết:

Bash
pip install -r requirements.txt
Chuẩn bị dữ liệu:

Tải file dữ liệu pmsm_temperature_data.csv từ Kaggle.

Đặt file vào thư mục gốc của project.
💻 Cách sử dụng
1. Huấn luyện mô hình (Training)
Chạy script để huấn luyện mô hình XGBoost và lưu artifacts:

Bash
python traindata.py
Script sẽ tự động tiền xử lý dữ liệu, train model và lưu kết quả vào thư mục ./models/.
2. Chạy Demo Web App
Khởi động giao diện giám sát thời gian thực:

Bash
python demo.py
Sau đó, truy cập vào địa chỉ: http://127.0.0.1:5000 trên trình duyệt để trải nghiệm.
📊 Kết quả đạt được
Độ chính xác: MAE ≈ 8.83°C, R² ≈ 0.89.

Khả năng phản hồi: Phản ứng tức thì với các thay đổi của dòng, áp và tốc độ.

Trực quan: Hiển thị cảnh báo màu dựa trên ngưỡng an toàn (Nam châm vĩnh cửu: 100°C, Stator: 120°C).

🔮 Hướng phát triển
Edge AI: Tối ưu hóa mô hình sang TensorFlow Lite để chạy trên các vi điều khiển (MCU) nhúng.

Online Learning: Nghiên cứu cơ chế cập nhật mô hình theo sự lão hóa của động cơ.

Chống nhiễu: Áp dụng các bộ lọc số (digital filters) để làm sạch tín hiệu cảm biến trong môi trường công nghiệp có nhiễu điện từ cao.
