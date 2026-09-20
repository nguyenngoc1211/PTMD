# Phân tích đặc trưng mã độc trên RAM bằng XGBoost và cARFi

Project nghiên cứu feature importance trên `Obfuscated-MalMem2022.csv`: huấn luyện XGBoost, tính marginal importance, lấy mẫu có điều kiện bằng ARF, tính CPI và sinh báo cáo tiếng Việt.

**Báo cáo kết quả:** [report/research_results.md](report/research_results.md).

## Chạy trong môi trường hiện tại

```bash
bash run_all.sh
```

Lệnh chạy toàn bộ pipeline, cập nhật bảng/hình/báo cáo và xác minh artifact. Cần vài phút đến hàng chục phút tùy CPU; bước ARF là phần tốn thời gian nhất. Không cần chạy malware hoặc thu thập RAM mới. Mỗi lần chạy sẽ ghi đè các kết quả hiện tại.

## Thiết kế thí nghiệm

- Dữ liệu gốc: 58.596 mẫu, 55 đặc trưng thuộc 9 nhóm forensic.
- Loại bản ghi trùng vector trước khi chia stratified train/test 80/20; median chỉ học từ train.
- XGBoost: 300 cây, depth=6, learning_rate=0.08, seed=42.
- MFI F1: toàn test, 30 lần permutation; là kết quả bổ sung.
- MFI log-loss: 1.500 mẫu test, 30 lần; là đối chứng cho CFI.
- ARF/FORDE: chỉ 3.000 mẫu train, không học từ test.
- CFI: 5 conditional draws mỗi hàng; trung bình loss qua draws trước khi kiểm định.
- So sánh MFI/CFI cùng log-loss trên các hàng hợp lệ tương ứng; báo cáo coverage, SE, p-value và BH q-value.
- Bảy case studies, mỗi case có ba tập điều kiện và cùng tập hàng hợp lệ.

Cấu hình tài nguyên, seed và số lần lấy mẫu nằm trong [config.json](config.json). Số cây và các siêu tham số XGBoost nằm trong `python/train_xgboost.py`. Sampling song song theo feature, mỗi feature có seed riêng; không chạy RNG song song bên trong một phép lấy mẫu. Tái lập được kiểm tra trong môi trường Linux hiện tại; thay phiên bản thư viện có thể thay đổi kết quả.

## Cài mới

Python 3.10+ và R 4.1+ trên Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
Rscript r/install_dependencies.R
bash run_all.sh
```

`requirements.txt` khóa các thư viện Python trực tiếp theo môi trường đã kiểm chứng. `requirements-lock.txt` lưu toàn bộ package của virtualenv. Cài R cần kết nối CRAN/GitHub và bộ công cụ biên dịch cho `ranger`/`RcppEigen`. Script cài dependencies vào `RLibrary/` trong project, ghim ARF theo commit đã dùng. Không cần `mlr3`, `cpi`, `knockoff` hay `Rdsdp` cho pipeline này.

Có thể chọn Python khác bằng `PYTHON=/absolute/path/to/python bash run_all.sh`.

## Các bước riêng

Chạy từ thư mục project:

```bash
.venv/bin/python python/preprocess.py
.venv/bin/python python/train_xgboost.py
.venv/bin/python python/export_for_carfi.py
.venv/bin/python python/marginal_importance.py
Rscript r/conditional_sampling.R
.venv/bin/python python/compute_cfi.py
.venv/bin/python python/analyze_results.py
.venv/bin/python python/generate_report.py
.venv/bin/python python/verify_results.py
```

Đổi dữ liệu/cấu hình cần chạy lại toàn bộ pipeline để các bước dùng cùng tập chia và schema. `inspect_data.py` chỉ kiểm tra CSV gốc; không ghi đè profile sau tiền xử lý.

## Cấu trúc

| Đường dẫn | Vai trò |
| --- | --- |
| `config.json` | Seed, kích thước tập con, repetitions, tài nguyên |
| `python/preprocess.py` | Kiểm tra nhãn, loại trùng, chia tập, median train |
| `python/train_xgboost.py` | Huấn luyện và đánh giá mô hình |
| `python/export_for_carfi.py` | Xuất đúng chỉ số split đã lưu và chọn tập con |
| `python/marginal_importance.py` | MFI F1 và MFI log-loss |
| `r/conditional_sampling.R` | ARF/FORDE, conditional draws và diagnostics |
| `python/compute_cfi.py` | Kiểm tra row IDs, tính CPI, SE và p/q-value |
| `python/analyze_results.py` | So sánh, tương quan, case studies và hình |
| `python/generate_report.py` | Sinh báo cáo và manifest mã nguồn |
| `python/verify_results.py` | Xác minh split, mô hình, CFI và báo cáo |
| `tests/` | Kiểm thử hồi quy |
| `results/models/` | XGBoost và ARF/FORDE |
| `results/intermediate/` | Split, draws, loss differences, manifest và verification |
| `results/tables/` | Metrics, importance, correlations, sampling diagnostics |
| `results/figures/` | Hình dùng trong báo cáo |

`data/Obfuscated-MalMem2022.csv` là đầu vào mặc định. File CSV cùng tên ở root là bản sao cũ và không được pipeline sử dụng. `RLibrary/` và `.venv/` là dependencies cài tại máy.

## Kiểm chứng

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python python/verify_results.py
```

`results/intermediate/verification.json` ghi các phép kiểm tra đã qua. `run_manifest.json` lưu SHA-256 dataset/mã nguồn, cấu hình và phiên bản. Kiểm tra tích hợp tính lại metrics và CFI top-3 từ mô hình và conditional samples.

## Cách đọc kết quả

- Không so trực tiếp giá trị MFI F1 với CFI log-loss. Bảng so sánh chính dùng cùng log-loss.
- `conditional_importance_se` là sai số chuẩn giữa các hàng; SD của MFI F1 là biến động giữa permutations.
- `ci_lo` là cận dưới tin cậy **một phía** 95%; q-value là p-value sau điều chỉnh BH.
- FORGE dùng `nomatch="na"`; evidence ngoài support của biến phân loại cũng bị loại trước khi gọi FORGE. `coverage` cho biết tỷ lệ hàng dùng được; không âm thầm dùng unconditional fallback.
- `sampling_diagnostics.csv` ghi số draw thiếu, âm và ngoài phạm vi train. ARF có thể không tái tạo đầy đủ ràng buộc dữ liệu; hoàn thành pipeline không đồng nghĩa mọi kết luận khoa học đã được xác nhận.
- Importance không phải nhân quả, không biểu diễn phần trăm thông tin, và accuracy trên benchmark không đại diện cho malware ngoài thực tế.

## Nguồn phương pháp

- [CIC-MalMem-2022 — Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/malmem-2022.html)
- [cARFi: mã nguồn tham khảo](https://github.com/bips-hb/cARFi_paper)
- [ARF: Watson et al., AISTATS 2023](https://proceedings.mlr.press/v206/watson23a.html)
- [FORDE](https://bips-hb.github.io/arf/reference/forde.html), [FORGE](https://bips-hb.github.io/arf/reference/forge.html)

Triển khai R/Python theo nguyên lý ARF + CPI; không tuyên bố là bản sao chính xác toàn bộ implementation cARFi chính thức.
