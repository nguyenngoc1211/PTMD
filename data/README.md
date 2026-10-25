# Dữ liệu đầu vào

Git chỉ lưu hướng dẫn và manifest schema trong thư mục này. CSV được tải/cung cấp
riêng; không đổi tên dataset mới thành `Obfuscated-MalMem2022.csv`.

## Dataset tham chiếu

Truy cập [trang chính thức CIC-MalMem-2022](https://www.unb.ca/cic/datasets/malmem-2022.html),
chọn **Download the dataset**, lấy/giải nén CSV `Obfuscated-MalMem2022.csv` và đặt tại:

```text
data/Obfuscated-MalMem2022.csv
```

Chỉ cần bảng đặc trưng đã trích xuất. Không cần tải/chạy mẫu mã độc hay tạo memory dump.
Trang nguồn mô tả 58.596 bản ghi, hai lớp benign và malware. Trích dẫn dataset theo
thông tin tác giả trên trang nguồn khi viết báo cáo.

Kiểm tra đúng bản CSV đã dùng tại dự án:

```bash
sha256sum data/Obfuscated-MalMem2022.csv
python3 -c 'import json; print(json.load(open("data/manifests/malmem2022.json"))["expected_sha256"])'
```

Hai chuỗi phải giống nhau để tái lập đúng bản đầu vào. SHA-256 thay đổi cả khi chỉ
đổi xuống dòng; nếu khác, kiểm tra nguồn và định dạng trước. Nếu là phiên bản dữ liệu
khác có chủ đích, tạo manifest riêng, ghi phiên bản mới và SHA-256, không âm thầm
ghi đè manifest tham chiếu.

## Dataset mới

Đặt CSV ở `data/new_dataset.csv` hoặc truyền đường dẫn ngoài repository. Sao chép
`manifests/new_dataset.example.json`, sửa nguồn, bối cảnh thu thập, nhãn và danh sách
feature theo CSV thật. Những tên cột trong file example chỉ minh họa, không phải
một dataset đã được kiểm chứng. Xem [hướng dẫn đầy đủ](../docs/REPRODUCIBILITY_VI.md).
