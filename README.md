# Phân tích đặc trưng mã độc trên RAM bằng XGBoost và cARFi

Project nghiên cứu feature importance trên `Obfuscated-MalMem2022.csv`: huấn luyện XGBoost, tính marginal importance, lấy mẫu có điều kiện bằng ARF, tính CPI và sinh báo cáo tiếng Việt.

## Thành viên mới: bắt đầu ở đây

- [Hướng dẫn cài đặt, tái lập và chạy dataset mới](docs/REPRODUCIBILITY_VI.md): lệnh từng bước, cấu hình nhãn/cột, chạy thử, chạy 3 seed và đọc kết quả.
- [Đưa dự án lên GitHub và bàn giao cho nhóm](docs/GITHUB_HANDOFF_VI.md).
- [Mẫu biên bản kết quả cần gửi lại](docs/EXPERIMENT_RESULT_TEMPLATE.md).
- [Lấy dữ liệu và kiểm tra SHA-256](data/README.md).

Luồng bàn giao mới dùng `python/run_dataset_experiments.py`: mỗi seed có thư mục riêng,
không ghi đè pipeline gốc. Hỗ trợ CSV đặc trưng **số**, phân loại **Benign/Malware**,
với danh sách feature và ánh xạ nhãn khai báo rõ trong manifest. Không cần chạy file malware.
Dataset khác schema được huấn luyện một mô hình riêng; đây là lặp phương pháp trên
dataset mới, chưa phải đem mô hình MalMem đi dự đoán trực tiếp dữ liệu bên ngoài.

```bash
# Sau khi cài môi trường và lấy dữ liệu theo hướng dẫn:
.venv/bin/python python/run_dataset_experiments.py \
  --data data/Obfuscated-MalMem2022.csv \
  --manifest data/manifests/malmem2022.json \
  --output results/reproduction/malmem2022-full --seeds 42,43,44
```

CSV, `results/`, `report/` và thư viện cài tại máy không đưa vào Git. Các liên kết
báo cáo/kết quả lịch sử bên dưới chỉ hoạt động khi có artifact tương ứng tại máy;
clone mới cần chạy thí nghiệm hoặc nhận gói kết quả từ nhóm. GitHub Actions chỉ chạy
kiểm thử Python, không chạy các thí nghiệm R/ARF đầy đủ.

**Báo cáo chính thức (bản duy nhất để nộp):** [report/bao_cao_tong_hop/Bao_cao_nghien_cuu_XAI_bo_nho_ma_doc.pdf](report/bao_cao_tong_hop/Bao_cao_nghien_cuu_XAI_bo_nho_ma_doc.pdf) — 22 trang, cập nhật 18/10/2026. Kịch bản nói 3–4 phút: [report/KICH_BAN_BAO_CAO_3_PHUT.md](report/KICH_BAN_BAO_CAO_3_PHUT.md). Các báo cáo rời là nguyên liệu truy vết; xem [report/README.md](report/README.md).

## Thí nghiệm riêng từng nhóm malware với benign

```bash
.venv/bin/python python/run_family_experiments.py
```

Chạy đầy đủ XGBoost, MFI F1/log-loss, CFI/cARFi và kiểm chứng riêng cho
Ransomware + Benign, Spyware + Benign, Trojan + Benign. Đây là ba nhóm cấp cao
theo `Category`, chưa tách 15 family con. Giữ nguyên cấu hình thí nghiệm gốc
và toàn bộ benign, loại trùng trước khi chia tập trong từng bài toán.

Kết quả ở `results/family_experiments/{ransomware,spyware,trojan}/`, gồm bản chụp
mã nguồn/cấu hình, dữ liệu đã lọc, chỉ số dòng nguồn, model, bảng, hình và log.
Pipeline tổng ban đầu vẫn được giữ riêng. Chạy lại lệnh sẽ cập nhật các thí nghiệm này.

- [Báo cáo so sánh và kết luận](report/family_results.md)
- [Bốn báo cáo cột mốc: dữ liệu, mô hình, importance, kiểm chứng](report/family_milestones/README.md)
- [Metrics ba mô hình](results/family_experiments/metrics_summary.csv)
- [Toàn bộ importance của 55 feature × 3 nhóm](results/family_experiments/feature_importance_all.csv)
- [Hình top-10 CFI](results/family_experiments/cfi_top10_comparison.png)

Để tổng hợp lại các lần chạy đã hoàn tất: `.venv/bin/python python/run_family_experiments.py --summary-only`.
Nếu bị ngắt, chạy `.venv/bin/python python/run_family_experiments.py --resume` để kiểm chứng
và dùng lại các nhóm đã hoàn tất, chạy lại nhóm còn thiếu.

## Kiểm tra tổng quát hóa theo family chưa thấy (leave-one-family-out)

```bash
.venv/bin/python python/run_generalization.py --seeds 42,43,44 --cv-folds 3
```

Với từng family malware trong 15 family con (Ransomware/Spyware/Trojan), giữ trọn
family đó làm test cộng phần benign test, huấn luyện bằng 14 family còn lại cộng
phần benign train. Family bị giữ không tham gia huấn luyện, chọn feature hay ngưỡng.
Thí nghiệm so hai bộ feature trên cùng fold: `baseline` (55 feature gốc) và
`engineered` (thêm tỉ lệ theo số tiến trình `ratio.*` và tổng/trung bình theo nhóm
forensic `group.*`). Lặp nhiều seed để đo biến động; `--cv-folds` chạy stratified
k-fold bên trong tập train để tách độ ổn định của mô hình khỏi vấn đề họ chưa thấy.
Đối chứng là chia stratified ngẫu nhiên như pipeline chính.

Kết quả ở `results/generalization/`; báo cáo ở
[report/generalization_results.md](report/generalization_results.md). Module này độc
lập với pipeline đã kiểm chứng và không sửa `config.json`. Dựng lại báo cáo/hình từ
bảng đã lưu mà không huấn luyện lại bằng `--report-only`.

Kết quả hiện tại (3 seed, CV 3-fold): F1 leave-one-family-out ≈ **0.9988** so với
**0.9999** khi chia ngẫu nhiên; sai số tập trung ở `Transponder` (~0.987) và
`Reconyc` (~0.996). Bộ feature engineered **không** cải thiện thực chất (ΔF1 ≈ +1e-5).

## Đối chiếu phương pháp với benchmark độc lập

```bash
.venv/bin/python python/cross_check_benchmark.py
```

Module này không huấn luyện lại model: nó đọc các bảng đã sinh của pipeline cộng
bản sao nhỏ của hai benchmark công khai dùng cùng CIC-MalMem-2022 (gói leakage-audit
trên Zenodo 22726771 và OACSP-Forensics trên Zenodo 21435602), rồi đối chiếu các
đại lượng so sánh được: số nhóm vector sau loại trùng, mức bão hoà nhị phân, độ khó
ở mức family và thứ hạng feature. Bản sao artifact nằm ở `results/external_benchmarks/`.

Báo cáo ở [report/methodology_cross_check.md](report/methodology_cross_check.md).
Kết quả: số nhóm vector duy nhất khớp tuyệt đối (58,027), độ chính xác nhị phân khớp
đến ~1e-6, và 5/5 (top-5) feature quan trọng nhất trùng với xếp hạng gain của OACSP.

## Phân loại đa lớp family (cải thiện thuật toán)

```bash
.venv/bin/python python/run_multiclass_family.py --seeds 42,43,44
```

Phát hiện nhị phân đã bão hoà (F1 1.0 khi chia ngẫu nhiên, 0.9988 khi
leave-one-family-out), nên thí nghiệm này nhắm vào bài toán chưa bão hoà: gán nhãn
family. Hai mục tiêu song song: **16 lớp** (Benign + 15 family, macro-F1 là metric
chính) và **4 lớp** (Benign + Ransomware/Spyware/Trojan, để so trực tiếp với mốc
accuracy 0.878 của OACSP-Forensics). So sánh XGBoost, HistGradientBoosting,
RandomForest, ExtraTrees, LogisticRegression và một mô hình phân cấp type→family,
lặp nhiều seed trên giao thức chia stratified đã loại trùng vector. Các nhóm vector
trùng nhau nhưng khác family bị loại bỏ thay vì gán nhãn tuỳ tiện.

Kết quả cuối (3 seed 42/43/44, kèm biến thể cân bằng lớp):

- **16 lớp:** XGBoost tốt nhất, macro-F1 **0.5528 ± 0.0054** (accuracy 0.7633 ±
  0.0027); kế tiếp RandomForest 0.5439, HistGradientBoosting 0.5409, mô hình phân
  cấp 0.5359, ExtraTrees 0.5147; LogisticRegression chỉ 0.2945.
- **4 lớp:** XGBoost accuracy **0.8763 ± 0.0028**, macro-F1 0.8126 ± 0.0042 — sát
  mốc OACSP-Forensics 0.8783, xác nhận độ khó nằm ở mức family chứ không phải type.
- Toàn bộ sai số nằm trong nội bộ từng nhóm: **0** nhầm Benign↔malware. Các cặp
  nhầm nhiều nhất là `Transponder`→`CWS`, `CWS`→`Transponder`, `Transponder`→`Gator`.
- Cân bằng lớp giúp XGBoost tăng nhẹ (macro-F1 0.5467 → 0.5528).

Kết quả ở `results/family_multiclass/`; báo cáo ở
[report/family_multiclass.md](report/family_multiclass.md). Module độc lập với
pipeline đã kiểm chứng và không sửa `config.json`. Dựng lại báo cáo/hình từ bảng đã
lưu bằng `--report-only`; chạy thêm biến thể cân bằng lớp bằng `--balanced`.

## Conditional feature importance cho bài toán family (xAI)

```bash
.venv/bin/python python/cfi_family.py
```

CFI hiện có chỉ ở dạng nhị phân "type vs Benign", và cả ba nhóm đều lấy `svcscan` làm
feature đứng đầu nên không đặc thù. Module này tái dùng **nguyên** bộ lấy mẫu có điều
kiện ARF của pipeline đã kiểm chứng, chỉ thay log-loss nhị phân bằng **multiclass
log-loss** để CFI xác định được cho mọi số lớp, rồi chạy ba bài toán trong đó hai bài
toán bỏ hẳn Benign:

- `malware3`: Ransomware / Spyware / Trojan, không có Benign.
- `family15`: 15 họ malware, không có Benign.
- `type4`: Benign + ba nhóm, dùng làm đối chứng.

Kết quả (seed 42, 3.000 dòng ARF, 1.500 dòng test, 5 draws, BH trên 55 feature,
coverage 1.0):

- **`svcscan` là bộ phân biệt benign-vs-malware, không phải đặc trưng family.** Vị trí
  tốt nhất của nhóm `svcscan` là hạng **1** ở `type4` (có Benign), nhưng tụt xuống
  hạng **14** ở `malware3` và hạng **10** ở `family15`.
- Khi bỏ Benign, thứ hạng dẫn đầu chuyển sang nhóm `handles`, `malfind` và
  `ldrmodules` ở cả hai mức phân giải (3 nhóm và 15 họ).
- Feature lặp lại ở nhiều bài toán: `handles.nsemaphore`, `handles.nkey`,
  `malfind.commitCharge`, `handles.nfile`; kế đó là `handles.nsection`,
  `malfind.uniqueInjections`, `ldrmodules.not_in_load_avg`, `dlllist.avg_dlls_per_proc`,
  `callbacks.ncallbacks`.
- Số feature có q < 0.05: **31/55** ở `malware3` và **35/55** ở `family15`.
- Hướng tác động: `handles.nfile` giảm dần theo Ransomware > Spyware > Trojan
  (842 / 763 / 646); `malfind.commitCharge` cao vượt trội ở Spyware (2394).

Mỗi bài toán chạy trong thư mục riêng `results/cfi_family/<task>/` (bản sao
`config.json` + `conditional_sampling.R` + symlink `RLibrary`), không sửa file nào của
pipeline đã kiểm chứng. Kết quả ở `results/cfi_family/`; báo cáo ở
[report/cfi_family.md](report/cfi_family.md). Dựng lại báo cáo bằng `--report-only`;
giảm chi phí khi thử bằng `--arf-train`, `--test-samples`, `--repeats`.

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
| `python/run_generalization.py` | Leave-one-family-out, feature engineered, nhiều seed/CV |
| `python/cross_check_benchmark.py` | Đối chiếu phương pháp với benchmark độc lập (chỉ đọc) |
| `python/run_multiclass_family.py` | Phân loại đa lớp family (16 lớp + 4 lớp), nhiều model |
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
