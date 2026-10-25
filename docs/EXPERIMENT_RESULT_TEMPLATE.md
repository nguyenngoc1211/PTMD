# Biên bản kiểm chứng thực nghiệm

## Người chạy và môi trường

- Người chạy, ngày, máy/OS:
- Repository, branch, commit SHA:
- Có sửa mã nguồn sau commit không? Nếu có, đính kèm diff:
- Python, R, CPU/RAM, threads:
- File `python-environment.txt`, `r-environment.txt`:
- Lệnh đầy đủ, thời gian bắt đầu/kết thúc, thời gian thực chạy:
- Mode: `full` / `smoke` (smoke chỉ kiểm tra kỹ thuật):

## Dữ liệu và giao thức

- Tên/phiên bản, nguồn, trích dẫn, quyền chia sẻ:
- SHA-256 CSV nguồn và CSV chuẩn hóa:
- Bối cảnh: thời gian thu thập, OS, công cụ trích xuất, đơn vị quan sát:
- Vì sao độc lập với MalMem? Có dùng lại/mirror/trùng mẫu không?
- Manifest; ý nghĩa nhãn và ánh xạ Benign/Malware:
- Các ID/hash/tên file/timestamp/nhãn khác đã bỏ:
- Số mẫu/lớp/cột trước và sau xử lý; số bản ghi trùng:
- Quan hệ giữa các dòng (cùng máy/mẫu/phiên) và nguy cơ rò rỉ còn lại:
- Giao thức split; seed; ngân sách ARF/permutation/draw; thay đổi so với baseline:

## Kết quả (điền từ artifact, không sao chép số lịch sử)

| Dataset | Seed | Train/test | F1 | Precision | Recall | ROC-AUC | Accuracy | CFI coverage min/max | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 42 | | | | | | | | |
| A | 43 | | | | | | | | |
| A | 44 | | | | | | | | |
| B | 42 | | | | | | | | |
| B | 43 | | | | | | | | |
| B | 44 | | | | | | | | |

- Mean ± SD qua seed (không coi seed là dataset độc lập):
- Top CFI từng seed: feature, importance, SE, q-value, coverage:
- Độ ổn định top-k giữa seed; so A/B chỉ khi feature tương thích:
- Diagnostics ARF: draws thiếu, âm, ngoài khoảng train, discriminator accuracy:
- Lỗi, cảnh báo và lần chạy thất bại:
- Giải thích chênh lệch với lần chạy tham chiếu, nếu có:

## Kết luận và artifact

- Tái lập trên máy khác thành công đến mức nào?
- Dataset B bổ sung bằng chứng gì? Kết quả nào không nhất quán?
- Giới hạn: retrain mỗi dataset, random holdout, chất lượng sampler, cỡ mẫu:
- Không suy luận nhân quả hay khả năng train A → test B nếu chưa làm phép thử đó.
- Link gói artifact + SHA-256 archive:
- Link báo cáo/bảng/hình/log/verification/config/manifest:
