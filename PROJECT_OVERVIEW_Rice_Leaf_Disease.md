# Rice Leaf Disease Classification System

> **Hệ thống nhận diện và phân loại bệnh trên lá lúa bằng Computer
> Vision**\
> Phiên bản: `1.0` · Loại dự án: Cá nhân · Định hướng: AI Engineer

------------------------------------------------------------------------

## 1. Giới thiệu dự án

**Rice Leaf Disease Classification System** là dự án Computer Vision xây
dựng hệ thống phân loại tình trạng lá lúa từ ảnh RGB. Người dùng tải ảnh
lá lúa lên ứng dụng web; hệ thống xử lý ảnh, đưa ảnh qua mô hình Deep
Learning và trả về:

-   Nhãn dự đoán.
-   Confidence của lớp dự đoán.
-   Xác suất dự đoán của toàn bộ 6 lớp.
-   Bản đồ Grad-CAM giúp quan sát vùng ảnh có ảnh hưởng đến dự đoán.
-   Kết quả so sánh giữa hai mô hình: Custom CNN và ResNet50.

Dự án được phát triển theo hướng **end-to-end AI application**, bao gồm
thu thập và xử lý dữ liệu, huấn luyện mô hình, đánh giá, xây dựng API,
phát triển giao diện web và đóng gói triển khai.

### Mục tiêu chính

1.  Xây dựng pipeline dữ liệu có thể tái lập và hạn chế data leakage.
2.  Huấn luyện hai mô hình độc lập: Custom CNN và ResNet50 Transfer
    Learning.
3.  So sánh mô hình bằng cùng tập test và cùng hệ thống metrics.
4.  Tích hợp Grad-CAM để hỗ trợ giải thích dự đoán.
5.  Triển khai mô hình qua FastAPI và giao diện React.
6.  Hoàn thiện mã nguồn, tài liệu và demo để đưa vào portfolio AI
    Engineer.

------------------------------------------------------------------------

## 2. Phạm vi bài toán

### 2.1. Bài toán

-   **Task:** Multi-class Image Classification.
-   **Input:** Một ảnh RGB có chứa lá lúa.
-   **Output:** Một trong sáu lớp, kèm xác suất dự đoán.
-   **Image size:** `224 × 224 × 3`.
-   **Số lớp:** 6.

### 2.2. Danh sách lớp

    Class ID Tên lớp         Tên hiển thị tham khảo
  ---------- --------------- ------------------------
           0 Healthy         Lá khỏe mạnh
           1 Blast           Bệnh đạo ôn
           2 Brown Spot      Bệnh đốm nâu
           3 Leaf Smut       Leaf Smut
           4 Rice Tungro     Bệnh Tungro
           5 Sheath Blight   Bệnh khô vằn

> Tên tiếng Việt chỉ mang tính tham khảo và cần được xác minh với tài
> liệu chuyên ngành hoặc chuyên gia bảo vệ thực vật trước khi công bố.
> Một số bệnh có triệu chứng trên nhiều bộ phận cây, không chỉ riêng
> phiến lá.

### 2.3. Giới hạn phiên bản 1.0

Phiên bản đầu tiên chỉ thực hiện **phân loại ảnh**. Hệ thống không:

-   Phát hiện nhiều đối tượng hoặc khoanh vùng từng vết bệnh.
-   Phân đoạn chính xác vùng mô bị bệnh.
-   Định lượng phần trăm diện tích bị tổn thương.
-   Chẩn đoán toàn bộ tình trạng cây chỉ từ một ảnh.
-   Đưa ra đơn thuốc, liều lượng hoặc phác đồ điều trị.

Nếu muốn định vị vết bệnh, dự án cần mở rộng sang Object Detection hoặc
Image Segmentation với dữ liệu gán nhãn phù hợp.

------------------------------------------------------------------------

## 3. Dataset

### 3.1. Dataset chính

Dự án sử dụng **RiceLeafDiseaseBD -- Version 3** làm dataset chính.

-   Nguồn: [Mendeley Data -- RiceLeafDiseaseBD Version
    3](https://data.mendeley.com/datasets/86s4jzj2m4/3)
-   Quy mô được công bố: khoảng **9.769 ảnh**.
-   Kích thước ảnh được mô tả: `1024 × 1024`.
-   Số lớp: 6.
-   Version 3 công bố 21.460 bounding-box annotations; phiên bản 1.0 của
    dự án không sử dụng annotation này để huấn luyện detector.
-   Bối cảnh thu thập: ảnh lá lúa chụp ngoài đồng bằng smartphone tại
    Bangladesh.
-   Giấy phép: **CC BY 4.0**; cần ghi công nguồn theo điều kiện giấy
    phép.

Sau khi tải dataset, cần kiểm tra lại số lượng ảnh thực tế, cấu trúc thư
mục, nhãn và metadata thay vì mặc định dữ liệu trên máy khớp hoàn toàn
với mô tả.

### 3.2. Dataset bổ sung

Có thể khảo sát [Paddy Doctor](https://paddy-doctor.github.io/) làm
nguồn dữ liệu bổ sung hoặc kiểm tra khả năng tổng quát hóa. Dataset này
có các nhóm dữ liệu và điều kiện chụp khác nhau. Nếu sử dụng, cần:

-   Chỉ sử dụng ảnh RGB trong thí nghiệm so sánh ảnh màu.
-   Xây dựng taxonomy nhãn tương thích.
-   Kiểm tra nhãn và chất lượng ảnh.
-   Không gộp các lớp có định nghĩa khác nhau một cách máy móc.
-   Tách dataset ngoài thành external test nếu mục tiêu là đánh giá
    domain generalization.

### 3.3. Chia dữ liệu

Tỷ lệ đề xuất:

  Tập            Tỷ lệ Mục đích
  ------------ ------- ----------------------------------------------
  Train            70% Huấn luyện mô hình
  Validation       15% Chọn checkpoint và điều chỉnh hyperparameter
  Test             15% Đánh giá cuối cùng

Quy tắc quan trọng:

1.  Thực hiện split trước khi augmentation.
2.  Dùng stratified split để giữ tỷ lệ lớp tương đối ổn định.
3.  Nếu có thông tin cây/lá/phiên chụp, dùng group split để tránh ảnh
    liên quan nằm ở nhiều tập.
4.  Kiểm tra ảnh trùng và gần trùng giữa các tập.
5.  Không sử dụng test set để chọn mô hình hoặc điều chỉnh
    hyperparameter.

### 3.4. Data card

Nên tạo `docs/DATA_CARD.md` ghi lại:

-   Tên, phiên bản, nguồn và giấy phép dataset.
-   Ngày tải và cách tải.
-   Số ảnh thực tế theo từng lớp.
-   Định dạng, kích thước và chất lượng ảnh.
-   Quy trình làm sạch và xử lý ảnh.
-   Quy tắc chia train/validation/test.
-   Hạn chế của dữ liệu và nguy cơ domain shift.

------------------------------------------------------------------------

## 4. Quy trình xử lý dữ liệu

### 4.1. Exploratory Data Analysis (EDA)

Các nội dung EDA bắt buộc:

-   Thống kê tổng số ảnh và số ảnh theo từng lớp.
-   Trực quan hóa phân bố lớp.
-   Hiển thị ảnh mẫu của từng lớp.
-   Thống kê kích thước, định dạng, số kênh và dung lượng ảnh.
-   Kiểm tra ảnh lỗi, ảnh mờ, quá tối hoặc quá sáng.
-   Kiểm tra ảnh trùng lặp và gần trùng.
-   Quan sát background, điều kiện ánh sáng và góc chụp.
-   Kiểm tra các nhãn có khả năng sai hoặc không rõ ràng.

### 4.2. Preprocessing

Pipeline cơ bản:

1.  Đọc ảnh bằng Pillow hoặc OpenCV.
2.  Chuyển ảnh về RGB.
3.  Kiểm tra ảnh hợp lệ.
4.  Resize/crop về `224 × 224`.
5.  Chuyển ảnh thành tensor.
6.  Scale pixel về `[0, 1]`.
7.  Normalize theo cấu hình của mô hình.

Đối với ResNet50 pretrained ImageNet, sử dụng normalization:

``` python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Cần đảm bảo preprocessing khi inference giống preprocessing lúc
validation/test của checkpoint tương ứng.

### 4.3. Data augmentation

Các phép biến đổi khởi điểm cho tập train:

  Augmentation          Cấu hình gợi ý
  --------------------- ----------------------------
  Horizontal Flip       `p = 0.5`
  Rotation              Khoảng `-15°` đến `+15°`
  Brightness/Contrast   Thay đổi nhẹ khoảng `±15%`
  Random Resized Crop   Scale khoảng `0.85–1.0`

Không áp dụng augmentation ngẫu nhiên lên validation và test. Tránh biến
đổi màu quá mạnh vì màu sắc có thể là đặc trưng quan trọng của triệu
chứng bệnh.

### 4.4. Xử lý mất cân bằng lớp

Nếu phân bố lớp không đồng đều, cân nhắc:

-   Class-weighted Cross Entropy.
-   `WeightedRandomSampler`.
-   Augmentation có kiểm soát cho lớp ít ảnh.

Công thức class weight khởi điểm:

\[ w_i = `\frac{N}{K \times n_i}`{=tex} \]

Trong đó:

-   `N`: tổng số ảnh train.
-   `K`: số lớp.
-   `n_i`: số ảnh của lớp `i`.

Nên thử từng kỹ thuật riêng biệt và so sánh bằng validation macro-F1.

------------------------------------------------------------------------

## 5. Hai mô hình Deep Learning

Dự án triển khai hai mô hình độc lập để so sánh hai hướng tiếp cận: tự
xây dựng CNN và transfer learning từ mô hình pretrained.

### 5.1. Model A -- Custom CNN

Custom CNN được xây dựng từ đầu, không sử dụng trọng số pretrained.

Kiến trúc đề xuất:

``` text
Input: 224 × 224 × 3
        |
Conv Block 1: Conv(32) + BatchNorm + ReLU + MaxPool
        |
Conv Block 2: Conv(64) + BatchNorm + ReLU + MaxPool
        |
Conv Block 3: Conv(128) + BatchNorm + ReLU + MaxPool
        |
Conv Block 4: Conv(256) + BatchNorm + ReLU + MaxPool
        |
Global Average Pooling
        |
Linear(128) + ReLU
        |
Dropout(0.5)
        |
Linear(6)
        |
Output logits
```

Cấu hình train khởi điểm:

  Tham số          Giá trị
  ---------------- -------------------------------
  Optimizer        Adam
  Learning rate    `0.001`
  Loss             CrossEntropyLoss
  Batch size       32
  Epoch tối đa     50
  Early stopping   Patience 8
  Scheduler        ReduceLROnPlateau, factor 0.5

### 5.2. Model B -- ResNet50 Transfer Learning

ResNet50 sử dụng trọng số pretrained trên ImageNet. Thay classifier cuối
bằng classifier có 6 output.

``` text
Input: 224 × 224 × 3
        |
ResNet50 Backbone (ImageNet pretrained)
        |
Global Average Pooling
        |
Dropout(0.5)
        |
Linear(2048, 6)
        |
Output logits
```

Quy trình huấn luyện gồm hai phase:

**Phase 1 -- Feature Extraction**

-   Freeze toàn bộ backbone.
-   Chỉ train classifier mới.
-   Learning rate classifier: `1e-3`.
-   Thời gian: khoảng 10--15 epochs.

**Phase 2 -- Fine-tuning**

-   Unfreeze `layer4` trước.
-   Tiếp tục train với learning rate nhỏ.
-   Learning rate backbone khởi điểm: `1e-5`.
-   Có thể dùng AdamW.
-   Thời gian: khoảng 15--30 epochs, tùy validation.

Các con số trên là cấu hình khởi điểm, không phải kết quả tối ưu được
đảm bảo.

### 5.3. Nguyên tắc so sánh công bằng

-   Dùng cùng dataset và split manifest.
-   Dùng cùng thứ tự class mapping.
-   Dùng cùng kích thước ảnh đầu vào trong thí nghiệm chính.
-   Dùng cùng test set.
-   Chỉ chọn checkpoint dựa trên validation.
-   Ghi rõ preprocessing riêng của từng model.
-   Nếu có đủ tài nguyên, chạy nhiều random seeds và báo cáo mean ±
    standard deviation.

------------------------------------------------------------------------

## 6. Huấn luyện và quản lý thí nghiệm

Quy trình train:

1.  Đọc cấu hình và split manifest.
2.  Khởi tạo Dataset và DataLoader.
3.  Khởi tạo model, loss và optimizer.
4.  Train theo epoch.
5.  Đánh giá trên validation sau mỗi epoch.
6.  Lưu checkpoint tốt nhất theo validation metric.
7.  Điều chỉnh learning rate khi cần.
8.  Dừng sớm nếu validation không cải thiện.
9.  Khóa cấu hình và đánh giá trên test set.

Nên sử dụng TensorBoard hoặc CSV/JSON logger để ghi:

-   `experiment_id`
-   `model_name`
-   `dataset_version`
-   `split_version`
-   `random_seed`
-   `image_size`
-   `augmentation`
-   `optimizer`
-   `learning_rate`
-   `batch_size`
-   `epoch`
-   `train_loss`
-   `val_loss`
-   `val_macro_f1`
-   `checkpoint_path`
-   `training_duration`

Mỗi checkpoint cần có metadata tương ứng để đảm bảo có thể tái lập quá
trình inference.

------------------------------------------------------------------------

## 7. Đánh giá mô hình

### 7.1. Metrics

  -----------------------------------------------------------------------
  Metric                              Ý nghĩa
  ----------------------------------- -----------------------------------
  Accuracy                            Tỷ lệ dự đoán đúng trên toàn bộ mẫu

  Precision                           Mức độ chính xác của các dự đoán
                                      thuộc một lớp

  Recall                              Khả năng tìm ra các mẫu thực sự
                                      thuộc một lớp

  F1-score                            Cân bằng Precision và Recall

  Macro F1                            Trung bình F1 giữa các lớp, mỗi lớp
                                      có trọng số ngang nhau

  Weighted F1                         F1 trung bình có trọng số theo số
                                      lượng mẫu

  Confusion Matrix                    Thể hiện các cặp lớp thường bị nhầm
                                      lẫn

  ROC-AUC OvR                         Chỉ số bổ sung cho bài toán đa lớp
                                      nếu phù hợp
  -----------------------------------------------------------------------

**Macro F1** là một trong những chỉ số quan trọng vì nó giúp đánh giá
hiệu năng giữa các lớp, đặc biệt khi dữ liệu mất cân bằng.

### 7.2. Báo cáo so sánh

Tạo bảng kết quả thực nghiệm:

  Metric                   Custom CNN   ResNet50
  ---------------------- ------------ ----------
  Accuracy                    Chưa đo    Chưa đo
  Macro Precision             Chưa đo    Chưa đo
  Macro Recall                Chưa đo    Chưa đo
  Macro F1                    Chưa đo    Chưa đo
  Weighted F1                 Chưa đo    Chưa đo
  Inference time/image        Chưa đo    Chưa đo

Không điền số liệu giả định. Kết quả phải được lấy từ checkpoint và test
set thực tế, kèm thông tin môi trường đánh giá.

### 7.3. Error Analysis

Ngoài metrics tổng quan, cần phân tích:

-   Những cặp bệnh thường bị nhầm lẫn.
-   Lớp có recall thấp.
-   Ảnh có ánh sáng hoặc background bất thường.
-   Ảnh mờ, triệu chứng chưa rõ hoặc nhiều triệu chứng cùng xuất hiện.
-   Trường hợp mô hình dự đoán tự tin nhưng sai.
-   Sự khác biệt giữa kết quả Custom CNN và ResNet50.

------------------------------------------------------------------------

## 8. Explainability -- Grad-CAM

Grad-CAM được sử dụng để tạo heatmap biểu diễn vùng ảnh có ảnh hưởng đến
score của lớp mục tiêu.

Layer mục tiêu gợi ý:

  Model        Layer
  ------------ -----------------------------------------------------
  ResNet50     Convolution cuối trong `layer4`, ví dụ `layer4[-1]`
  Custom CNN   Convolution cuối của Block 4

Luồng xử lý:

1.  Forward pass để lấy logits.
2.  Chọn lớp mục tiêu cần giải thích.
3.  Tính gradient score của lớp theo feature maps.
4.  Tính trọng số cho từng feature-map channel.
5.  Tạo và chuẩn hóa heatmap.
6.  Resize heatmap về kích thước ảnh gốc.
7.  Overlay heatmap lên ảnh đầu vào.

Grad-CAM chỉ hỗ trợ quan sát hành vi mô hình. Heatmap không chứng minh
mô hình đã học đúng đặc trưng sinh học và có thể bị ảnh hưởng bởi
background, ánh sáng hoặc yếu tố phụ.

Tài liệu phương pháp: [Grad-CAM -- Visual Explanations from Deep
Networks](https://arxiv.org/abs/1610.02391).

------------------------------------------------------------------------

## 9. Kiến trúc hệ thống

### 9.1. Tổng quan

``` text
                 USER
                  |
                  v
        React + TypeScript UI
                  |
             HTTP Request
                  |
                  v
            FastAPI Backend
                  |
          Input Validation
                  |
          Image Preprocessing
                  |
             Model Registry
              /         \
             v           v
        Custom CNN    ResNet50
             \           /
              v         v
             Prediction
                  |
          +-------+-------+
          |               |
          v               v
    Probability       Grad-CAM
       Output          Service
          |               |
          +-------+-------+
                  |
                  v
             JSON Response
                  |
                  v
             React UI
```

PostgreSQL có thể được bổ sung để lưu lịch sử dự đoán và metadata.
Database không bắt buộc trong bản demo đầu tiên.

### 9.2. Tech stack

  -----------------------------------------------------------------------
  Thành phần              Công nghệ               Vai trò
  ----------------------- ----------------------- -----------------------
  Programming Language    Python 3.11             Data pipeline,
                                                  training, backend

  Deep Learning           PyTorch, TorchVision    CNN, ResNet50,
                                                  inference

  Image Processing        Pillow, OpenCV          Đọc và xử lý ảnh

  Data Science            NumPy, Pandas,          EDA và metrics
                          scikit-learn            

  Visualization           Matplotlib, Seaborn     Biểu đồ và phân tích

  Experiment Tracking     TensorBoard             Theo dõi thí nghiệm

  Explainability          Grad-CAM                Giải thích dự đoán

  Backend                 FastAPI, Uvicorn        REST API

  Frontend                React, TypeScript       Giao diện web

  Database                PostgreSQL (optional)   Lịch sử dự đoán

  Deployment              Docker, Docker Compose  Đóng gói hệ thống
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 10. Backend -- FastAPI

### 10.1. Trách nhiệm

Backend đảm nhiệm:

-   Nhận ảnh từ frontend.
-   Validate file và tham số.
-   Tiền xử lý ảnh.
-   Nạp và quản lý các model.
-   Thực hiện inference.
-   Tính xác suất dự đoán.
-   Gọi Grad-CAM service.
-   Trả kết quả JSON.
-   Ghi lịch sử dự đoán nếu bật database.

Model nên được load một lần khi ứng dụng khởi động, không load lại trong
mỗi request.

### 10.2. API endpoints

  Method   Endpoint             Mô tả
  -------- -------------------- ----------------------------------
  GET      `/`                  Thông tin API
  GET      `/health`            Kiểm tra trạng thái service
  GET      `/models`            Danh sách model khả dụng
  POST     `/predict`           Dự đoán ảnh bằng model được chọn
  POST     `/predict/compare`   So sánh hai model trên cùng ảnh
  POST     `/explain`           Tạo Grad-CAM cho ảnh

### 10.3. Response dự đoán mẫu

``` json
{
  "model_name": "resnet50",
  "predicted_class": "Brown Spot",
  "confidence": 0.87,
  "probabilities": {
    "Healthy": 0.02,
    "Blast": 0.03,
    "Brown Spot": 0.87,
    "Leaf Smut": 0.02,
    "Rice Tungro": 0.01,
    "Sheath Blight": 0.05
  },
  "warning": "Kết quả chỉ mang tính tham khảo."
}
```

> Đây là ví dụ định dạng response, không phải kết quả mô hình thực tế.

### 10.4. Validation và bảo mật

-   Giới hạn dung lượng upload, ví dụ 10 MB.
-   Kiểm tra phần mở rộng, MIME type và nội dung file.
-   Từ chối file không giải mã được.
-   Giới hạn kích thước ảnh cực lớn.
-   Không tin tưởng tên file từ client.
-   Không trả stack trace hoặc đường dẫn nội bộ.
-   Sử dụng `model.eval()` và `torch.inference_mode()` khi inference.

------------------------------------------------------------------------

## 11. Frontend -- React + TypeScript

### 11.1. Định hướng giao diện

Giao diện theo phong cách dashboard AI hiện đại, chuyên nghiệp, dễ sử
dụng. Màu sắc đề xuất:

-   Màu chủ đạo: xanh lá đậm.
-   Màu phụ: xanh dương và xám trung tính.
-   Background: trắng hoặc xám rất nhạt.
-   Typography: dễ đọc, phân cấp rõ ràng.

### 11.2. Các màn hình

  Page                 Chức năng
  -------------------- ---------------------------------------
  Dashboard            Giới thiệu hệ thống và truy cập nhanh
  Disease Detection    Upload ảnh và chọn model
  Prediction Result    Hiển thị nhãn, confidence và xác suất
  Explainability       Hiển thị ảnh Grad-CAM
  Model Comparison     So sánh Custom CNN và ResNet50
  Prediction History   Xem lịch sử nếu có database
  About Diseases       Thông tin tham khảo về sáu lớp
  About Model          Kiến trúc, dataset và metrics

### 11.3. Luồng người dùng

1.  Truy cập trang Disease Detection.
2.  Chọn hoặc kéo thả ảnh vào vùng upload.
3.  Xem preview và chọn model.
4.  Nhấn Predict.
5.  Chờ backend xử lý.
6.  Xem nhãn dự đoán và confidence.
7.  Xem xác suất từng lớp.
8.  Mở Grad-CAM để tham khảo vùng ảnh có ảnh hưởng đến dự đoán.
9.  Có thể chạy Compare để xem kết quả của cả hai model.

------------------------------------------------------------------------

## 12. Testing

### 12.1. Data testing

-   Kiểm tra số lượng ảnh và nhãn.
-   Kiểm tra ảnh hỏng.
-   Kiểm tra class mapping.
-   Kiểm tra duplicate/near-duplicate giữa các tập.
-   Kiểm tra split trước augmentation.

### 12.2. Model testing

-   Forward pass trả logits có shape `[batch_size, 6]`.
-   Softmax có tổng xác suất xấp xỉ 1.
-   Checkpoint load thành công.
-   Class mapping khớp metadata.
-   Grad-CAM trả heatmap hợp lệ.

### 12.3. API testing

  Test case   Input                 Expected result
  ----------- --------------------- -----------------------
  TC-API-01   Ảnh JPG hợp lệ        HTTP 200
  TC-API-02   Ảnh PNG hợp lệ        HTTP 200
  TC-API-03   File không phải ảnh   HTTP 415/422
  TC-API-04   File rỗng             HTTP 400/422
  TC-API-05   File quá dung lượng   HTTP 413
  TC-API-06   Model không tồn tại   HTTP 400
  TC-API-07   Model chưa sẵn sàng   HTTP 503
  TC-API-08   Compare endpoint      Trả kết quả hai model

### 12.4. Frontend testing

-   Upload, preview và xóa ảnh.
-   Chọn model và gửi request.
-   Hiển thị loading/error state.
-   Hiển thị đúng class mapping và xác suất.
-   Responsive trên desktop/tablet.
-   Luôn hiển thị disclaimer ở trang kết quả.

------------------------------------------------------------------------

## 13. Cấu trúc thư mục GitHub

``` text
rice-leaf-disease-classification/
│
├── configs/
│   ├── custom_cnn.yaml
│   └── resnet50.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── manifests/
│   └── external/
│
├── notebooks/
│   ├── 01_dataset_eda.ipynb
│   ├── 02_data_quality.ipynb
│   ├── 03_custom_cnn.ipynb
│   ├── 04_resnet50_transfer_learning.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_gradcam_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── transforms.py
│   │   └── split.py
│   ├── models/
│   │   ├── custom_cnn.py
│   │   └── resnet50.py
│   ├── training/
│   │   ├── train.py
│   │   ├── engine.py
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── explainability/
│   │   └── gradcam.py
│   └── inference/
│       ├── predictor.py
│       └── model_registry.py
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       ├── schemas/
│       └── services/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   └── types/
│   └── package.json
│
├── artifacts/
│   ├── checkpoints/
│   └── reports/
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_gradcam.py
│
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── PROJECT_SPECIFICATION.md
│   ├── DATA_CARD.md
│   ├── MODEL_CARD.md
│   └── EXPERIMENT_LOG.md
│
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

Không commit trực tiếp `.venv`, `node_modules`, dataset dung lượng lớn
hoặc checkpoint lớn nếu không cần thiết. Có thể sử dụng Git LFS hoặc
dịch vụ lưu trữ model phù hợp.

------------------------------------------------------------------------

## 14. Kế hoạch triển khai 8 tuần

  ----------------------------------------------------------------------------
                          Tuần Công việc             Deliverables
  ---------------------------- --------------------- -------------------------
                             1 Tải dataset, kiểm tra EDA notebook, Data Card,
                               license, EDA, tạo     split manifest
                               split                 

                             2 Xây dựng              Data pipeline, unit tests
                               preprocessing và      
                               DataLoader            

                             3 Huấn luyện Custom CNN CNN checkpoint, training
                                                     logs

                             4 Huấn luyện ResNet50   ResNet50 checkpoint,
                                                     training logs

                             5 Đánh giá, phân tích   Evaluation report,
                               lỗi, Grad-CAM         confusion matrix

                             6 Xây dựng FastAPI      Predict/compare/explain
                                                     API

                             7 Xây dựng React        Giao diện tích hợp API
                               frontend              

                             8 Docker, tài liệu,     Portfolio project hoàn
                               demo, hoàn thiện      chỉnh
                               GitHub                
  ----------------------------------------------------------------------------

### Thứ tự ưu tiên

Nên hoàn thành theo thứ tự:

1.  Dataset và EDA.
2.  Preprocessing và split.
3.  Custom CNN.
4.  ResNet50.
5.  Evaluation và Grad-CAM.
6.  Backend API.
7.  Frontend.
8.  Docker và tài liệu.

Không nên dành quá nhiều thời gian cho giao diện trước khi có baseline
model và kết quả đánh giá đáng tin cậy.

------------------------------------------------------------------------

## 15. Tiêu chí hoàn thành dự án

Dự án được xem là hoàn thành phiên bản 1.0 khi đáp ứng:

-   [ ] Có dataset source, license và Data Card.
-   [ ] Có EDA và kiểm tra chất lượng dữ liệu.
-   [ ] Có split manifest cố định, hạn chế data leakage.
-   [ ] Có pipeline preprocessing và augmentation.
-   [ ] Custom CNN train và inference được.
-   [ ] ResNet50 transfer learning/fine-tuning được.
-   [ ] Có đánh giá hai mô hình trên cùng test set.
-   [ ] Có confusion matrix và per-class metrics.
-   [ ] Có phân tích lỗi và Grad-CAM.
-   [ ] FastAPI có endpoint predict.
-   [ ] Frontend upload ảnh và hiển thị kết quả.
-   [ ] Có luồng so sánh hai mô hình.
-   [ ] Có README hướng dẫn chạy từ đầu.
-   [ ] Có Model Card nêu rõ giới hạn và rủi ro.
-   [ ] Có demo hoàn chỉnh.

------------------------------------------------------------------------

## 16. Rủi ro và giới hạn

### 16.1. Domain shift

Dataset chính được thu thập tại Bangladesh, trong khi điều kiện tại Việt
Nam có thể khác về:

-   Giống lúa.
-   Mùa vụ và giai đoạn sinh trưởng.
-   Khí hậu và điều kiện ánh sáng.
-   Thiết bị và góc chụp.
-   Nền ruộng và môi trường chụp.

Do đó, kết quả trên dataset gốc không chứng minh mô hình hoạt động tốt
trên mọi vùng trồng lúa tại Việt Nam.

### 16.2. Hướng cải thiện dữ liệu Việt Nam

Khi thu thập ảnh địa phương, nên lưu:

-   `image_id`
-   `disease_label`
-   `rice_variety`
-   `location`
-   `capture_date`
-   `growth_stage`
-   `capture_device`
-   `lighting_condition`
-   `expert_verified`

Nên có chuyên gia bảo vệ thực vật xác minh nhãn và xây dựng một tập
external test độc lập.

### 16.3. Các rủi ro khác

  -----------------------------------------------------------------------
  Rủi ro                              Hướng xử lý
  ----------------------------------- -----------------------------------
  Class imbalance                     Class weights, sampler, macro-F1

  Data leakage                        Split trước augmentation, group
                                      split, duplicate audit

  Overfitting                         Augmentation, dropout, early
                                      stopping

  Label noise                         Kiểm tra ảnh và xác minh nhãn

  Background bias                     EDA, Grad-CAM, kiểm tra ảnh ngoài
                                      miền

  Thiếu GPU                           Giảm batch size hoặc dùng
                                      Colab/Kaggle
  -----------------------------------------------------------------------

**Disclaimer:** Kết quả AI chỉ mang tính tham khảo và hỗ trợ sàng lọc
ban đầu. Không sử dụng dự đoán làm căn cứ duy nhất để quyết định biện
pháp xử lý bệnh. Người dùng nên tham khảo cán bộ bảo vệ thực vật hoặc
chuyên gia nông học.

------------------------------------------------------------------------

## 17. Hướng phát triển phiên bản tiếp theo

Sau khi hoàn thành phiên bản 1.0, có thể phát triển:

1.  **Model benchmarking:** bổ sung EfficientNet, DenseNet121, ResNet101
    hoặc MobileNet.
2.  **Object Detection:** định vị các vùng tổn thương trên lá.
3.  **Image Segmentation:** phân vùng chính xác mô bị bệnh.
4.  **Out-of-Distribution Detection:** phát hiện ảnh không thuộc phạm vi
    dữ liệu huấn luyện.
5.  **Vietnamese Dataset:** thu thập và xác minh dữ liệu tại các vùng
    trồng lúa Việt Nam.
6.  **MLOps:** quản lý model version, experiment tracking, monitoring và
    quy trình cập nhật model.
7.  **Mobile deployment:** tối ưu mô hình để chạy trên thiết bị di động.

------------------------------------------------------------------------

## 18. Tài liệu tham khảo

-   [RiceLeafDiseaseBD -- Version 3, Mendeley
    Data](https://data.mendeley.com/datasets/86s4jzj2m4/3)
-   [Paddy Doctor](https://paddy-doctor.github.io/)
-   [PyTorch Transfer Learning
    Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
-   [TorchVision
    ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)
-   [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
-   [FastAPI Documentation](https://fastapi.tiangolo.com/)
-   [React Documentation](https://react.dev/)
-   [PyTorch Documentation](https://pytorch.org/docs/stable/)

------------------------------------------------------------------------

## 19. Định hướng portfolio

Dự án này không chỉ dừng ở việc train một mô hình phân loại ảnh. Giá trị
chính nằm ở việc thể hiện toàn bộ quy trình phát triển một sản phẩm AI:

-   **Data:** thu thập, kiểm tra, phân tích và quản lý dữ liệu.
-   **Deep Learning:** xây dựng CNN, transfer learning và fine-tuning.
-   **Evaluation:** đánh giá có kiểm soát, phân tích lỗi và so sánh mô
    hình.
-   **Explainability:** trực quan hóa Grad-CAM.
-   **Backend:** triển khai inference service bằng FastAPI.
-   **Frontend:** xây dựng giao diện web tương tác.
-   **Engineering:** tổ chức source code, testing, Docker và tài liệu.

Khi đưa dự án vào CV/GitHub, cần trình bày kết quả thực nghiệm thực tế,
dataset, cách chia dữ liệu, metrics, demo và những giới hạn đã nhận
diện. Không nên chỉ ghi tên mô hình hoặc một con số accuracy mà thiếu
bối cảnh đánh giá.
