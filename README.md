🩺 Breast Cancer Prediction Web App

📌 Giới thiệu

Breast Cancer Prediction Web App là một ứng dụng web đơn giản nhưng hữu ích, được xây dựng nhằm hỗ trợ dự đoán xem khối u vú có khả năng là ác tính (malignant) hay lành tính (benign) dựa trên các đặc trưng được lấy từ xét nghiệm y tế.

Ứng dụng này minh họa sự kết hợp giữa mô hình học máy Random Forest và web framework Flask, mang lại cho người dùng trải nghiệm trực quan khi áp dụng trí tuệ nhân tạo (AI) vào lĩnh vực y tế hỗ trợ chẩn đoán.

🔎 Chức năng chính

Ứng dụng cho phép người dùng nhập vào các thông số xét nghiệm quan trọng của khối u, bao gồm:

Concave points (mean, worst) – đặc trưng hình dạng mô khối u.

Area worst – diện tích lớn nhất được ghi nhận.

Concavity mean – độ lõm trung bình.

Radius worst – bán kính lớn nhất.

Dựa trên các giá trị được nhập, mô hình đã huấn luyện sẵn sẽ đưa ra kết quả:

❌ Khối u ác tính (Malignant) – có khả năng là ung thư.

✅ Khối u lành tính (Benign) – ít nguy hiểm.

⚠️ Invalid Input – khi dữ liệu nhập không hợp lệ.

✨ Giao diện người dùng

Ứng dụng sử dụng Bootstrap 5 để tạo ra giao diện gọn gàng, hiện đại và dễ sử dụng. Người dùng chỉ cần nhập số liệu vào form, nhấn nút “Dự đoán”, kết quả sẽ hiển thị ngay lập tức với biểu tượng trực quan (❌, ✅).

Giao diện cũng được thiết kế responsive, tương thích tốt trên cả máy tính và điện thoại.

⚙️ Công nghệ sử dụng

Python 3 – ngôn ngữ chính.

Pandas, Scikit-learn – xử lý dữ liệu và huấn luyện mô hình Random Forest.

Flask – framework web backend.

Bootstrap 5 – xây dựng giao diện đẹp và thân thiện.

Pickle – lưu và tải lại mô hình đã huấn luyện.

Random Forest Classifier – thuật toán học máy chính, mạnh mẽ và ổn định.

🎯 Ý nghĩa và Ứng dụng

Ứng dụng không thay thế cho chẩn đoán y tế thực tế, nhưng mang lại nhiều giá trị học thuật và thực hành:

Minh họa học thuật – ví dụ điển hình về việc áp dụng Random Forest trong y học.

Thực hành Machine Learning – trải nghiệm toàn bộ quy trình: xử lý dữ liệu → train mô hình → lưu mô hình → tích hợp vào Flask → xây dựng giao diện web.

Trực quan hóa kết quả – giúp người học thấy rõ sự khác biệt giữa khối u ác tính và lành tính qua mô hình AI.

Mở rộng nghiên cứu – có thể bổ sung thêm đặc trưng, so sánh với các mô hình khác (SVM, Neural Network, Logistic Regression).

<img width="830" height="471" alt="{2E6C4478-F266-48E2-9199-23A576377398}" src="https://github.com/user-attachments/assets/ce8c4345-21cc-40f2-80ff-a4079899240f" />
