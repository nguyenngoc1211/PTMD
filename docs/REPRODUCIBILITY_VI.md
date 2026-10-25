# Hướng dẫn chạy lại và mở rộng thí nghiệm

Mục tiêu bàn giao: một thành viên khác clone đúng commit, tự dựng môi trường, chạy
lại MalMem2022, rồi chạy cùng phương pháp trên nguồn dữ liệu mới và trả lại bằng
chứng có thể kiểm tra. Không yêu cầu phải thu được kết quả tốt hơn.

## 1. Chốt câu hỏi trước khi chạy

| Thí nghiệm | Bằng chứng bổ sung | Giới hạn kết luận |
| --- | --- | --- |
| Cùng CSV, commit, cấu hình, seed trên máy khác | Khả năng tái lập | Không làm tăng đa dạng dữ liệu |
| MalMem, seed 42/43/44 | Biến động theo chia tập và huấn luyện | Vẫn cùng nguồn dữ liệu |
| Dataset độc lập B, huấn luyện lại, seed 42/43/44 | Phương pháp có chạy/cho kết quả nhất quán trên nguồn khác không | Không chứng minh mô hình học từ A chuyển sang B |
| Train A → test B, mô hình cố định | Tổng quát hóa giữa nguồn | Chưa được runner mới triển khai; cần schema/đơn vị tương thích và thiết kế riêng |

Ba nhóm Ransomware/Spyware/Trojan và 15 family trong MalMem là các phân nhóm cùng
benchmark. Dataset tải lại từ một mirror, đổi tên hoặc chia nhỏ MalMem không phải
nguồn độc lập. Dataset PE/network có thể mở rộng đánh giá phương pháp trên dữ liệu
bảng, nhưng không trực tiếp xác nhận các kết luận forensic về RAM.

Với dataset B, ghi nguồn chính thức, phiên bản, khoảng thời gian thu thập, OS,
công cụ trích xuất, đơn vị quan sát và quan hệ với MalMem. Xác nhận có cả benign
và malware, hiểu ý nghĩa nhãn, loại ID/hash/tên file/timestamp/nhãn suy ra khỏi
feature. Nếu nhiều dòng cùng máy, mẫu malware hay phiên thu thập, random holdout
có thể lạc quan: cần thiết kế split theo nhóm/thời gian riêng trước khi tuyên bố
tổng quát hóa. Runner hiện chỉ chia stratified theo dòng sau loại trùng vector.

## 2. Clone và cài môi trường

Luồng hỗ trợ chính: Linux hoặc Ubuntu trong WSL2, Python **3.10**, R **4.1+**.
Môi trường gốc dùng Python 3.10.12 và R 4.1.2. Bắt đầu với 4 CPU threads; thời gian
và RAM phụ thuộc số dòng/cột, không có mức tối thiểu đã đo cho dataset mới.

Clone repository của nhóm và thay `COMMIT_SHA` bằng commit người bàn giao cung cấp:

```bash
git clone https://github.com/nguyenngoc1211/PTMD.git
cd PTMD
git checkout COMMIT_SHA
git rev-parse HEAD
```

Nếu máy Ubuntu chưa có Python/R/công cụ biên dịch:

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip \
  r-base r-base-dev build-essential libcurl4-openssl-dev libssl-dev libxml2-dev
```

Nếu bản Ubuntu không cung cấp `python3.10`, cần cài Python 3.10 bằng trình quản lý
môi trường của máy; không tự sửa phiên bản thư viện để ép cài. Cài Python và R:

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
Rscript r/install_dependencies.R
.venv/bin/python --version
Rscript --version
.venv/bin/python -m pip check
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -m unittest discover -s tests -v
```

`requirements.txt` ghim dependencies trực tiếp. `requirements-lock.txt` là bản ghi
môi trường gốc, có cả gói phụ thuộc theo nền tảng; không mặc định dùng thay file
requirements chính. ARF được ghim theo commit trong `r/install_dependencies.R`.
Các gói R còn lại lấy từ CRAN, chưa có lockfile R đầy đủ: lưu phiên bản thực tế
để giải thích khác biệt giữa máy. Cài lần đầu cần Internet; chạy pipeline từ CSV
đã có và dependencies đã cài không cần tải thêm dữ liệu.

## 3. Tái lập MalMem trước

Lấy CSV theo [data/README.md](../data/README.md). Chạy từ gốc repository.

Kiểm tra dữ liệu, cả ba split và tạo bản chụp mà chưa huấn luyện:

```bash
.venv/bin/python python/run_dataset_experiments.py \
  --data data/Obfuscated-MalMem2022.csv \
  --manifest data/manifests/malmem2022.json \
  --output results/reproduction/malmem2022-preflight \
  --seeds 42,43,44 --prepare-only
```

Chạy thử toàn bộ luồng Python → R → CFI → báo cáo → verification:

```bash
.venv/bin/python python/run_dataset_experiments.py \
  --data data/Obfuscated-MalMem2022.csv \
  --manifest data/manifests/malmem2022.json \
  --output results/reproduction/malmem2022-smoke \
  --seeds 42 --mode smoke --threads 2
```

`smoke` vẫn huấn luyện XGBoost trên train, nhưng giảm ARF xuống tối đa 200 dòng,
test importance 50 dòng, 2 draws, 2 permutations, 5 cây ARF, min node 10. Chỉ dùng
để kiểm tra cài đặt; không trộn vào kết quả nghiên cứu. Dataset nhiều feature có
thể vẫn mất thời gian, hoặc coverage thấp vì tập ARF nhỏ.

Chạy kết quả chính:

```bash
.venv/bin/python python/run_dataset_experiments.py \
  --data data/Obfuscated-MalMem2022.csv \
  --manifest data/manifests/malmem2022.json \
  --output results/reproduction/malmem2022-full \
  --seeds 42,43,44 --mode full --threads 4
```

`full` dùng cấu hình gốc: split 80/20, ARF tối đa 3.000 dòng train, importance tối
đa 1.500 dòng test, 5 draws, 30 permutations. XGBoost giữ 300 cây, depth 6,
learning rate 0.08. Dataset nhỏ hơn giới hạn dùng số dòng thực tế. Mỗi seed huấn
luyện lại độc lập; chạy tuần tự. Không sửa `config.json` gốc.

Runner từ chối thư mục output đã tồn tại để tránh trộn kết quả. `--prepare-only`
không tự tiếp tục thành full: chọn output mới cho lần chạy thật. Nếu chạy hỏng,
giữ log và chọn tên mới, ví dụ `malmem2022-full-v2`; hiện chưa có resume cho runner
này. `run_family_experiments.py --resume` là tính năng riêng của pipeline family.

## 4. Chuẩn bị dataset mới

```bash
cp data/manifests/new_dataset.example.json data/manifests/dataset_b.json
```

Sửa file JSON theo dữ liệu thực tế:

| Trường | Nội dung |
| --- | --- |
| `dataset_id` | Tên ngắn chỉ gồm chữ thường, số, `_`, `-` |
| `source` | URL chính thức, phiên bản và trích dẫn |
| `collection_context` | Thời gian/OS/công cụ/đơn vị quan sát; giải thích độc lập với A |
| `label_column` | Tên cột nhãn gốc |
| `label_map` | Từng giá trị nhãn CSV → `Benign` hoặc `Malware` |
| `feature_columns` | Danh sách đầy đủ, có thứ tự, chỉ các cột số được phép dùng |
| `category_column` | Không bắt buộc; metadata nhóm/family, không làm feature |
| `case_features` | 1–7 feature có trong danh sách trên, chọn trước khi xem kết quả test |
| `expected_sha256` | Không bắt buộc; nên thêm SHA-256 CSV để khóa bản dữ liệu của nhóm |

Nhãn số phải dùng khóa chuỗi trong JSON, ví dụ `"0": "Benign"`; phải kiểm tra
dataset thật dùng 0/1 theo chiều nào. Nhãn được bỏ khoảng trắng hai đầu; phân biệt
hoa/thường. Nhãn lạ/thiếu gây lỗi, không tự chuyển thành benign. Nhiều loại malware
có thể ánh xạ về `Malware` khi mục tiêu đã chốt là nhị phân. Không dùng cách đó để
tuyên bố kết quả phân loại đa lớp.

CSV phải có header duy nhất, ít nhất hai feature số và đủ mẫu cho hai lớp trong
split. Cột ngoài danh sách feature bị bỏ khỏi đầu vào mô hình và được ghi trong
`provenance.json`. Cột feature có chuỗi không phải số gây lỗi; NaN/±Inf được xử lý
bằng median học từ train. Cột hoàn toàn thiếu trên train gây lỗi. Vector trùng
nhưng khác nhãn gây lỗi để kiểm tra nguồn. Categorical/text/raw binaries cần
pipeline trích xuất hoặc encoding riêng, hiện không tự động hỗ trợ.

Nhóm feature được suy ra từ tiền tố trước dấu chấm (ví dụ `handles.nfile` →
`handles`). Nếu schema không có nhóm có ý nghĩa, chỉ dùng CFI chính theo từng
feature; không diễn giải case study theo nhóm như bằng chứng forensic. Không
đổi tên/đổi đơn vị tùy tiện để làm schema trông giống MalMem.

## 5. Chạy dataset B

Sau khi thay các trường minh họa và đặt CSV thật tại `data/dataset_b.csv`:

```bash
.venv/bin/python python/run_dataset_experiments.py \
  --data data/dataset_b.csv --manifest data/manifests/dataset_b.json \
  --output results/reproduction/dataset-b-preflight --prepare-only

.venv/bin/python python/run_dataset_experiments.py \
  --data data/dataset_b.csv --manifest data/manifests/dataset_b.json \
  --output results/reproduction/dataset-b-smoke --seeds 42 --mode smoke

.venv/bin/python python/run_dataset_experiments.py \
  --data data/dataset_b.csv --manifest data/manifests/dataset_b.json \
  --output results/reproduction/dataset-b-full --seeds 42,43,44 --mode full
```

Giữ cùng cấu hình giữa A/B khi có thể. Nếu cần đổi ngân sách ARF hoặc hyperparameter,
ghi thay đổi trước khi đánh giá, chạy lại đối chứng tương ứng và lưu commit mới.
Không chọn seed hoặc cấu hình dựa trên kết quả test thuận lợi.

## 6. Theo dõi, xác minh và đọc kết quả

Trong terminal khác:

```bash
tail -f results/reproduction/dataset-b-full/seed_42/pipeline.log
```

Ctrl+C ở cửa sổ `tail` chỉ dừng xem log. Mỗi output có:

```text
status.json                      # prepared/running/failed/passed; mode; seeds
metrics_by_seed.csv               # chỉ được tổng hợp khi mọi seed thành công
metrics_summary.csv               # mean/std/min/max; std rỗng nếu chỉ 1 seed
seed_42/
  config.json                    # cấu hình của lần chạy
  dataset_manifest.json          # nguồn, nhãn, schema đã khai báo
  provenance.json                # hash CSV nguồn, commit, thứ tự dòng, cột bỏ
  python/  r/  run_all.sh         # snapshot mã nguồn
  data/input.csv                 # CSV chuẩn hóa theo manifest
  pipeline.log
  results/intermediate/          # split, draws, manifest, verification
  results/tables/                 # metrics, MFI/CFI, coverage, diagnostics
  results/figures/
  results/models/
  report/research_results.md
```

Chạy kiểm chứng lại một snapshot bằng Python của môi trường đã cài:

```bash
.venv/bin/python results/reproduction/dataset-b-full/seed_42/python/verify_results.py
```

Đọc `verification.json`: cần `status: passed`; phép kiểm tra tính lại metrics,
CFI top-3 từ model và draws, kiểm tra split, căn hàng Python/R, SHA-256 mã nguồn
và CSV đã chuẩn hóa. `passed` là nhất quán kỹ thuật, không đảm bảo dataset phù hợp
hay mẫu ARF đạt chất lượng. Mỗi báo cáo chỉ mô tả một seed.

Tổng hợp ít nhất: F1, precision, recall, ROC-AUC, accuracy theo seed và mean ± SD;
số dòng/lớp/cột, trùng lặp, nguồn dữ liệu; CFI cùng SE/q-value/coverage; diagnostics
ARF và độ ổn định top-k. SD của ba seed là mô tả biến động, không phải khoảng tin
cậy từ ba dataset độc lập. Không so MFI F1 với CFI log-loss; bảng
`marginal_vs_conditional.csv` dùng cùng log-loss và hàng hợp lệ.

Đối chiếu feature giữa A/B chỉ khi cùng ngữ nghĩa và đơn vị; khác schema thì đối
chiếu hiệu năng/quy trình và chất lượng sampler, không ép so hạng các cột khác
nhau. Báo cáo cả trường hợp giảm hiệu năng, coverage thấp hoặc không tái lập được.

## 7. Gói kết quả gửi lại

Điền [mẫu biên bản](EXPERIMENT_RESULT_TEMPLATE.md), ghi commit, lệnh, môi trường,
thời gian và mọi thay đổi. Lưu môi trường của máy chạy:

```bash
.venv/bin/python -m pip freeze > results/reproduction/dataset-b-full/python-environment.txt
Rscript -e '.libPaths(c("RLibrary", .libPaths())); print(sessionInfo()); print(installed.packages()[,c("Package","Version")])' \
  > results/reproduction/dataset-b-full/r-environment.txt
```

Gói đầy đủ để người khác kiểm chứng lại (có CSV, model và draws; có thể lớn):

```bash
tar --exclude='*/RLibrary' --exclude='*/.cache' --exclude='*/__pycache__' \
  -czf results/dataset-b-full.tar.gz -C results/reproduction dataset-b-full
sha256sum results/dataset-b-full.tar.gz
```

Gửi qua nơi lưu trữ của nhóm phù hợp với quyền chia sẻ dataset. Không commit gói
này vào Git. Có thể gửi riêng bảng/hình/log để đọc, nhưng gói rút gọn không đủ để
tính lại CFI. Symlink RLibrary trong snapshot trỏ vào máy chạy; khi chuyển máy,
dùng môi trường R của clone mới và tạo lại symlink nếu cần chạy lại sampling.

## 8. Lỗi thường gặp

| Lỗi | Cách xử lý |
| --- | --- |
| Không tìm thấy `.venv/bin/python` | Tạo venv, dùng Python 3.10, chạy từ gốc repo |
| Cài R package lỗi compiler | Cài `r-base-dev`, `build-essential`, thư viện hệ thống rồi đọc lỗi đầu tiên |
| Thiếu `arf` | Chạy `Rscript r/install_dependencies.R` ở clone gốc |
| SHA-256 không khớp | Xác minh phiên bản/định dạng dữ liệu, không bỏ qua để nhận là tái lập cùng CSV |
| Thiếu cột/nhãn lạ/feature không phải số | Sửa manifest theo CSV, kiểm tra ý nghĩa nhãn; không tự loại dòng lỗi |
| Không đủ mẫu mỗi lớp/trùng vector khác nhãn | Kiểm tra dataset và mục tiêu; không nhân bản mẫu trước chia tập |
| Output đã tồn tại | Chọn tên output mới, giữ lại log cũ |
| Hết RAM | Giảm `--threads`, chạy seed tuần tự; ghi rõ nếu phải đổi ngân sách sampling |
| Coverage thấp hoặc không có CFI hữu hạn | Kiểm tra phân phối/đặc trưng hiếm, diagnostics và số mẫu ARF; không thay bằng kết quả 0 |

Các module family/LOFO/multiclass trong README vẫn chuyên cho `Category` của
MalMem. Không chạy chúng trên dataset B chỉ vì pipeline nhị phân đã chạy được.
