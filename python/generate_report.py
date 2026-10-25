"""Generate the Vietnamese research report from current, auditable artifacts."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from common import CONFIG, ROOT, setup


def table(frame):
    def fmt(x):
        if pd.isna(x): return 'N/A'
        if isinstance(x,(float,np.floating)): return f'{x:.6g}'
        return str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                       '| '+' | '.join(['---']*len(frame.columns))+' |']+
                      ['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    setup()
    profile=json.loads(Path('results/intermediate/data_profile.json').read_text())
    metadata=json.loads(Path('results/intermediate/arf_metadata.json').read_text())
    read=lambda name:pd.read_csv(f'results/tables/{name}.csv')
    metrics=read('model_metrics'); cfi=read('conditional_feature_importance')
    mfi=read('marginal_feature_importance'); merged=read('marginal_vs_conditional')
    cases=read('conditioning_case_studies'); diagnostics=read('sampling_diagnostics')
    groups=read('forensic_group_summary'); pairs=read('high_corr_pairs')
    saved=joblib.load('results/models/xgboost_model.joblib')
    tn,fp,fn,tp=confusion_matrix(saved['y_test'],saved['y_pred']).ravel()
    significant=cfi[cfi.q_value.lt(CONFIG['alpha']) & cfi.ci_lo.gt(0)]
    top=cfi.head(3).feature.tolist()
    environment=dict(python=platform.python_version(),packages={name:importlib.metadata.version(name)
                     for name in ['numpy','pandas','scipy','scikit-learn','xgboost','matplotlib','seaborn','joblib']},
                     r=metadata['r_version'],arf=metadata['arf_version'])
    files=list(Path('python').glob('*.py'))+list(Path('r').glob('*.R'))+[Path('config.json'),Path('run_all.sh')]
    files += [p for p in (Path('dataset_manifest.json'), Path('provenance.json')) if p.exists()]
    manifest=dict(config=CONFIG,dataset_sha256=profile['sha256'],environment=environment,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})
    Path('results/intermediate/run_manifest.json').write_text(json.dumps(manifest,indent=2))
    main_diag=diagnostics[diagnostics.conditioning_set.eq('All other features')]
    arf_accuracies = metadata.get('discriminator_accuracy', [])
    if isinstance(arf_accuracies, (int, float)):
        arf_accuracies = [arf_accuracies]  # jsonlite auto-unboxes a single ARF iteration.
    arf_accuracy = ', '.join(f'{v:.2%}' for v in arf_accuracies)
    dataset_description = CONFIG.get('dataset_description',
        'CIC-MalMem-2022 — https://www.unb.ca/cic/datasets/malmem-2022.html')
    text=f'''# Phân tích mức độ quan trọng có điều kiện của đặc trưng mã độc trên RAM

Báo cáo được sinh từ kết quả hiện tại bởi `python/generate_report.py`. Các bảng, hình và cấu hình thuộc cùng pipeline. Đây là nghiên cứu thực nghiệm trên benchmark; không phải hệ thống triển khai phát hiện mã độc ngoài thực tế.

## 1. Mục tiêu và phạm vi

Câu hỏi nghiên cứu: với mô hình XGBoost đã huấn luyện, đặc trưng nào còn ảnh hưởng đến chất lượng dự đoán khi việc thay thế đặc trưng có xét sự phụ thuộc với các đặc trưng khác?

Phương pháp chính là lấy mẫu có điều kiện bằng Adversarial Random Forest (ARF) theo hướng cARFi, sau đó tính chênh lệch log-loss trên từng quan sát. MFI theo log-loss là đối chứng chính; MFI theo F1 là kết quả bổ sung. Không huấn luyện malware, không thu thập memory dump mới và không suy luận nhân quả từ feature importance.

## 2. Dữ liệu và chống rò rỉ

Dữ liệu: {dataset_description}. Thí nghiệm dùng CSV tại `{profile['path']}`; fingerprint SHA-256 bên dưới xác định chính xác đầu vào đã chuẩn hóa. Nếu chạy qua `run_dataset_experiments.py`, xem `provenance.json` để truy vết SHA-256 CSV nguồn và ánh xạ nhãn.

| Thuộc tính | Giá trị |
| --- | --- |
| Số mẫu gốc | {profile['samples_raw']} |
| Bản ghi trùng vector đặc trưng đã loại | {profile['duplicates_removed']} |
| Số mẫu sau loại trùng | {profile['samples_total']} |
| Số mẫu train / test | {profile['samples_train']} / {profile['samples_test']} |
| Benign / Malware sau loại trùng | {profile['class_counts']['0']} / {profile['class_counts']['1']} |
| Số đặc trưng / nhóm | {profile['features_numeric']} / {len(profile['feature_groups'])} |
| Giá trị thiếu trước điền median | {profile['missing_values_before_imputation']} |
| Mẫu test trùng vector với train sau xử lý | {profile['train_test_duplicate_overlap']} |

SHA-256: `{profile['sha256']}`.

Loại `Category` và `Class` khỏi đầu vào mô hình. Nhãn 0 là Benign, 1 là Malware. Các vector đặc trưng trùng hệt nhau được giữ một bản trước khi chia stratified holdout với seed {CONFIG['seed']}. Nếu vector giống nhau nhưng nhãn mâu thuẫn, pipeline dừng để kiểm tra. Median chỉ được học từ train; không chuẩn hóa vì mô hình cây không yêu cầu bước này.

Các cột hằng số trên train: {', '.join('`'+f+'`' for f in profile['constant_features'])}. Chúng vẫn nằm trong đầu vào XGBoost để giữ schema. ARF bỏ cột hằng số trong tập con huấn luyện của nó; đây cũng là hạn chế khi giá trị hiếm không xuất hiện trong tập con.

Loại bản ghi trùng không loại được mọi dạng phụ thuộc giữa các dump cùng máy hay cùng họ mã độc. Holdout hiện tại chưa phải phép thử trên họ mã độc chưa từng thấy hoặc trên thời điểm tương lai.

![Phân bố nhãn](../results/figures/01_class_distribution.png)

## 3. Mô hình và kết quả phân loại

XGBoost: 300 cây, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85; seed={CONFIG['seed']}. Không chọn siêu tham số dựa trên test.

{table(metrics[['accuracy','precision','recall','f1','roc_auc']])}

Ma trận nhầm lẫn: TN={tn}, FP={fp}, FN={fn}, TP={tp}. Hiệu năng này chỉ mô tả tập test hiện tại. Accuracy cao không xác nhận chất lượng của mô hình sinh ARF và không chứng minh khả năng tổng quát hóa ra thực tế.

![Ma trận nhầm lẫn](../results/figures/02_confusion_matrix.png)

## 4. Thiết kế MFI và CFI

### 4.1 Phân biệt phép đo

- **MFI F1 bổ sung:** hoán vị một cột trên toàn bộ {profile['samples_test']} mẫu test, {CONFIG['permutation_repeats']} lần. Giá trị là F1 gốc trừ F1 sau hoán vị; error bar là độ lệch chuẩn qua các lần hoán vị, không phải SE.
- **MFI log-loss đối chứng:** hoán vị cột trên {metadata['test_rows']} mẫu test được chọn stratified, {CONFIG['permutation_repeats']} lần; trung bình chênh lệch loss theo từng hàng.
- **CFI log-loss:** thay một cột bằng mẫu ARF có điều kiện, {metadata['repeats']} lần trên mỗi hàng, rồi trung bình loss theo hàng trước khi kiểm định. Mô hình XGBoost được giữ cố định.

Với feature j và hàng i, d_i = trung bình_r[loss(y_i, f(X_i thay cột j bằng mẫu r))] − loss(y_i, f(X_i)). CFI_j = trung bình_i d_i. Loss là binary log-loss; xác suất chuyển sang float64 và chặn tại [1e-15, 1−1e-15] để tránh log(0).

MFI log-loss trong bảng so sánh được tính lại trên đúng các hàng có CFI hợp lệ của từng feature. Không chia trực tiếp CFI cho MFI F1 vì hai đơn vị khác nhau. Nếu coverage khác nhau giữa các feature, thứ hạng vẫn chịu ảnh hưởng bởi tập quan sát khác nhau.

### 4.2 ARF và kiểm soát lấy mẫu

ARF học trên **{metadata['train_rows']} mẫu chỉ từ train**, với {CONFIG['arf_num_trees']} cây và min_node_size={CONFIG['arf_min_node_size']}. FORDE học mật độ trên cùng tập con. Tập test không tham gia học ARF hoặc FORDE.

Theo [tài liệu FORDE](https://bips-hb.github.io/arf/reference/forde.html), `finite_bounds="no"` giữ các biên ngoài vô hạn. Cấu hình này tránh áp biên thực nghiệm của từng lá lên giá trị chưa gặp. Nó cũng cho phép giá trị sinh vượt phạm vi train, nên cần xem diagnostics.

Theo [tài liệu FORGE](https://bips-hb.github.io/arf/reference/forge.html), `nomatch="force"` có thể chuyển sang lấy mẫu ngẫu nhiên khi evidence không có lá phù hợp. Pipeline dùng **`nomatch="na"`**; hàng có bất kỳ replicate không hữu hạn nào bị loại khỏi ước lượng của feature đó và được thống kê rõ. Không thay mẫu thiếu bằng mẫu marginal.

Các chỉ số test_position là zero-based; replicate là 1..{metadata['repeats']}. Python kiểm tra tính duy nhất, độ đầy đủ và sắp lại thứ tự theo khóa; R kiểm tra các giá trị evidence sau FORGE. Sampling chạy song song giữa các feature, tuần tự bên trong mỗi feature, với seed riêng cho từng feature/cấu hình để tái lập kết quả trong cùng môi trường.

Đây là triển khai theo nguyên lý ARF + CPI bằng R/Python, **không tuyên bố tương đương chính xác mọi chi tiết với package cARFi chính thức**. Tài liệu và mã nguồn tham khảo: [cARFi_paper](https://github.com/bips-hb/cARFi_paper), [bài báo ARF](https://proceedings.mlr.press/v206/watson23a.html).

### 4.3 Suy luận thống kê

SE = SD(d_i)/sqrt(n). Kiểm định t một phía xét H0: E[d_i] ≤ 0, H1: E[d_i] > 0. `ci_lo` là cận dưới tin cậy một phía {(1-CONFIG['alpha'])*100:g}%, không phải khoảng hai phía. Điều chỉnh Benjamini–Hochberg trên {len(cfi)} giả thuyết CFI; case studies điều chỉnh riêng trên {len(cases)} cấu hình. `q_value < {CONFIG['alpha']}` là quy tắc báo cáo tín hiệu dương sau điều chỉnh.

SE mô tả biến động giữa các hàng sau trung bình replicate; không bao gồm toàn bộ bất định do huấn luyện lại XGBoost/ARF. Kết quả kiểm định là thăm dò, phụ thuộc giả định về các hàng và chất lượng phân phối có điều kiện. Loss có thể lệch mạnh trên mô hình gần hoàn hảo; p-value nhỏ không thay thế đánh giá độ lớn hiệu ứng.

## 5. Kết quả feature importance

### 5.1 MFI F1 trên toàn test

{table(mfi.head(10))}

![MFI F1](../results/figures/03_marginal_importance_top10.png)

### 5.2 CFI trên tập con test

{table(cfi.head(15)[['feature','conditional_importance','conditional_importance_se','p_value','q_value','ci_lo','n_valid','coverage']])}

Ba feature có ước lượng CFI cao nhất: {', '.join('`'+f+'`' for f in top)}. Có **{len(significant)} / {len(cfi)}** feature có q-value < {CONFIG['alpha']} và cận dưới dương. Xếp hạng cao và ý nghĩa thống kê là hai tiêu chí khác nhau; xem cả độ lớn, SE và coverage.

![CFI](../results/figures/04_conditional_importance_top10.png)

### 5.3 So sánh cùng metric

{table(merged.nsmallest(15,'conditional_rank')[['feature','marginal_logloss','conditional_importance','marginal_logloss_rank','conditional_rank','rank_change','q_value']])}

`rank_change = rank CFI − rank MFI log-loss`; số dương nghĩa là tụt hạng sau conditioning. Thay đổi thứ hạng chỉ gợi ý sự khác nhau giữa hai phép đo. Không tự động chứng minh redundancy, suppression hoặc tương tác nhân quả; còn có sai số lấy mẫu và sai số mô hình mật độ.

![So sánh log-loss](../results/figures/05_marginal_vs_conditional.png)
![So sánh thứ hạng](../results/figures/05_rank_mfi_vs_cfi.png)
![Thay đổi thứ hạng](../results/figures/06_rank_change.png)

## 6. Tương quan và nhóm forensic

Có {len(pairs)} cặp trong toàn bộ {len(cfi)} feature thỏa |Spearman ρ| ≥ 0.8 trên test. Các cột hằng số có tương quan không xác định và không được đưa vào danh sách cặp.

{table(pairs.head(12))}

![Tương quan](../results/figures/07_top_feature_correlation_heatmap.png)

{table(groups)}

`mean_cfi` là trung bình có dấu trên các feature hợp lệ, không chỉ trung bình giá trị dương. Bảng tổng hợp từng feature không phải phép đo importance khi thay thế cả nhóm cùng lúc.

Các nhóm như svcscan, handles và malfind cung cấp cách tổ chức đặc trưng để diễn giải kết quả. Importance không cho biết malware làm số service hay handle tăng hoặc giảm; cần phân tích phân bố theo nhãn hoặc phân tích hiệu ứng riêng để kết luận chiều tác động. CFI thấp cũng không đủ cơ sở gọi một nhóm là vô dụng hoặc khuyến nghị bỏ feature mà chưa kiểm tra mô hình sau khi bỏ.

## 7. Conditioning case studies

{cases.feature.nunique()} feature được định trước trong cấu hình hoặc mã nguồn, mỗi feature xét ba tập điều kiện: tất cả feature khác, cùng nhóm theo tiền tố tên cột trước dấu chấm, và các nhóm khác. Nhóm chỉ có ý nghĩa forensic nếu schema dữ liệu định nghĩa như vậy. Trong mỗi feature, ba phép đo dùng giao của các hàng hợp lệ để giữ cùng tập đánh giá. Trường hợp “all” tái sử dụng chính mẫu của CFI chính, không sinh lại một bộ mẫu ngẫu nhiên khác.

{table(cases[['feature','conditioning_set','importance','se','q_value','n_valid','coverage']])}

![Case studies](../results/figures/08_conditioning_sets.png)

Các kết quả cho thấy độ nhạy của ước lượng khi đổi tập điều kiện. Không diễn giải tỷ lệ giảm CPI thành “phần trăm thông tin được giải thích”, và không giả định CPI phải biến đổi đơn điệu khi thêm điều kiện. Với tập điều kiện một phần, các feature còn lại vẫn giữ giá trị quan sát trong mô hình dự đoán; đây là phép nhiễu từng cột theo tập điều kiện đã chọn.

## 8. Diagnostics và giới hạn

- Tổng số draw CFI chính: {int(main_diag.n_draws.sum())}; không hữu hạn: {int(main_diag.n_missing.sum())}; giá trị âm: {int(main_diag.n_negative.sum())}; ngoài min/max tập con ARF train: {int(main_diag.n_outside_train_range.sum())}.
- Coverage CFI nhỏ nhất/lớn nhất: {cfi.coverage.min():.2%} / {cfi.coverage.max():.2%}. Thiếu mẫu có thể gây selection bias; khi coverage thấp, CFI chỉ mô tả phần dữ liệu có thể lấy mẫu.
- Giá trị sinh ngoài train không mặc nhiên sai, nhưng giá trị âm đối với số đếm forensic là dấu hiệu hạn chế mô hình mật độ. Pipeline ghi nhận, không tự ý cắt giá trị vì thao tác đó sẽ thay phân phối lấy mẫu. Xem toàn bộ `sampling_diagnostics.csv`.
- Độ chính xác discriminator ARF theo vòng huấn luyện: {arf_accuracy}. Mức còn cao hơn 50% cho thấy dữ liệu sinh vẫn có thể phân biệt với dữ liệu thật; không được xem ARF là bộ lấy mẫu điều kiện đã được xác nhận hoàn toàn.
- Chưa xác nhận đầy đủ chất lượng phân phối có điều kiện trên dữ liệu này. ARF có thể chưa học hết phụ thuộc, nhất là feature hiếm, ràng buộc số nguyên và quan hệ xác định giữa các tỷ lệ.
- Một seed, một holdout, một mô hình; chưa đo độ ổn định qua nhiều seed hay trên benchmark độc lập.
- Các bản ghi gần giống nhau và nguồn dump liên quan vẫn có thể xuất hiện ở hai tập sau loại trùng chính xác.
- Số mẫu ARF và số replicate hữu hạn; SE không bao trùm sai số do mô hình sinh.

## 9. Kết luận

Pipeline hoàn chỉnh cho phép huấn luyện XGBoost, đo MFI và CFI, so sánh cùng log-loss trên các hàng tương ứng, phân tích conditioning sets và lưu diagnostics. Kết quả chính của lần chạy này là danh sách feature, độ lớn hiệu ứng và bất định trong các bảng ở trên; chúng không phải bằng chứng nhân quả hoặc cam kết phát hiện malware ngoài thực tế.

Hướng phát triển tiếp theo: kiểm tra chất lượng conditional sampler, lặp nhiều seed, đánh giá theo nhóm nguồn dump/họ malware và đo importance cả nhóm để kiểm tra các giả thuyết redundancy.

## 10. Tái lập và sản phẩm

Chạy `bash run_all.sh` từ thư mục project. Cấu hình:

```json
{json.dumps(CONFIG,indent=2)}
```

Môi trường đã chạy:

```json
{json.dumps(environment,indent=2)}
```

- `results/intermediate/run_manifest.json`: cấu hình, phiên bản và SHA-256 mã nguồn.
- `results/intermediate/data_profile.json`: kiểm tra dữ liệu và chia tập.
- `results/intermediate/verification.json`: kết quả xác minh artifact sau khi pipeline kết thúc.
- `results/tables/`: toàn bộ metrics, importance, p/q-value, coverage, diagnostics và tương quan.
- `results/figures/`: hình đã tái sinh từ kết quả hiện tại.
- `results/models/`: XGBoost và ARF/FORDE của lần chạy.
- `tests/test_pipeline.py`: kiểm thử các lỗi hồi quy về dữ liệu, loss, thống kê và alignment.

Kiểm tra nhanh: `.venv/bin/python -m unittest discover -s tests -v` và `.venv/bin/python python/verify_results.py`.
'''
    Path('report').mkdir(exist_ok=True)
    Path('report/research_results.md').write_text(text)
    print('Generated report/research_results.md and run_manifest.json')


if __name__ == '__main__': main()
