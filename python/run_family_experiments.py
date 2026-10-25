"""Run the existing XGBoost + MFI + cARFi pipeline for each malware type."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import ROOT, CONFIG
from generate_report import table

FAMILIES = ('Ransomware', 'Spyware', 'Trojan')


def select_family(frame, family):
    if family not in FAMILIES:
        raise ValueError(f'Unknown malware type: {family}')
    classes = frame.Class.str.strip().str.lower()
    types = frame.Category.str.split('-').str[0]
    mask = (classes.eq('benign') & types.eq('Benign')) | (classes.eq('malware') & types.eq(family))
    selected = frame.loc[mask].copy()
    if set(selected.Class.str.lower()) != {'benign', 'malware'}:
        raise ValueError('Both classes are required')
    return selected


def prepare_run(frame, family, base):
    work = base / family.lower()
    for directory in ('python', 'r', 'data', 'report'):
        (work / directory).mkdir(parents=True, exist_ok=True)
    # Snapshot the unchanged pipeline so hard-coded relative outputs are isolated.
    for pattern in ('python/*.py', 'r/*.R', 'run_all.sh'):
        for source in ROOT.glob(pattern):
            shutil.copy2(source, work / source.relative_to(ROOT))
    library = work / 'RLibrary'
    if not library.exists():
        library.symlink_to(ROOT / 'RLibrary', target_is_directory=True)
    selected = select_family(frame, family)
    selected.to_csv(work / CONFIG['data'], index=False)
    (work / 'config.json').write_text(json.dumps(CONFIG, indent=2))
    (work / 'selection.json').write_text(json.dumps(dict(
        malware_type=family, source=str(ROOT / CONFIG['data']),
        source_sha256=hashlib.sha256((ROOT / CONFIG['data']).read_bytes()).hexdigest(),
        original_source_rows=selected.index.tolist(),
        raw_counts=selected.Class.value_counts().to_dict(),
        categories=selected.Category.str.split('-').str[:2].str.join('-').value_counts().to_dict(),
    ), indent=2))
    return work


def write_milestone_reports(base, metrics, importance):
    """Record completed milestones from verified artifacts, without rerunning training."""
    destination = ROOT / 'report/family_milestones'
    destination.mkdir(parents=True, exist_ok=True)
    data_rows, verification_rows, diagnostic_rows = [], [], []
    for family in FAMILIES:
        work = base / family.lower()
        profile = json.loads((work / 'results/intermediate/data_profile.json').read_text())
        receipt = json.loads((work / 'results/intermediate/verification.json').read_text())
        data_rows.append(dict(family=family, raw=profile['samples_raw'],
                             duplicates_removed=profile['duplicates_removed'],
                             benign=profile['class_counts']['0'], malware=profile['class_counts']['1'],
                             train=profile['samples_train'], test=profile['samples_test'],
                             duplicate_overlap=profile['train_test_duplicate_overlap']))
        verification_rows.append(dict(family=family, status=receipt['status'], checks=receipt['n_checks'],
                                      coverage_min=receipt['coverage_min'], coverage_max=receipt['coverage_max']))
        diag = pd.read_csv(work / 'results/tables/sampling_diagnostics.csv')
        diag = diag[diag.conditioning_set.eq('All other features')]
        diagnostic_rows.append(dict(family=family, draws=int(diag.n_draws.sum()),
                                    missing=int(diag.n_missing.sum()), negative=int(diag.n_negative.sum()),
                                    outside_train_range=int(diag.n_outside_train_range.sum())))
    data = pd.DataFrame(data_rows)
    data.to_csv(base / 'data_summary.csv', index=False)
    common_note = ('Báo cáo cột mốc đã hoàn thành, được tổng hợp từ artifact của cả ba bài toán. '
                   'Các mốc được ghi lại sau khi lượt chạy kết thúc; không phải các lần huấn luyện mới.\n\n')
    source_links = '\n'.join(
        f'- [{family}: kết quả kiểm chứng](../../results/family_experiments/{family.lower()}/results/intermediate/verification.json)'
        for family in FAMILIES)
    top = pd.concat([importance[importance.family.eq(family)].nsmallest(3, 'conditional_rank') for family in FAMILIES])
    top = top[['family','feature','conditional_importance','conditional_importance_se','q_value','coverage']]
    reports = {
        '01_data.md': '# Mốc 1 — Chuẩn bị dữ liệu\n\n' + common_note
            + 'Phạm vi: Ransomware, Spyware và Trojan, mỗi nhóm ghép với benign. Không trộn các nhóm malware trong cùng mô hình; chưa tách 15 family con. Nhãn Class/Category không nằm trong 55 feature.\n\n'
            + table(data) + '\n\n'
            + 'Loại trùng vector trước khi chia stratified train/test 80/20; median chỉ học từ train. Cả ba bài toán có 0 vector trùng giữa train và test sau tiền xử lý. Giữ toàn bộ benign, không undersampling. Các tập benign của ba bài toán có thể trùng nhau; ba thí nghiệm không phải ba mẫu nghiên cứu độc lập.\n\n'
            + 'Kết luận: dữ liệu đã sẵn sàng cho ba bài toán nhị phân. Exact deduplication không loại được mọi bản ghi gần giống hoặc dump có liên hệ.\n\n'
            + '[Bảng dữ liệu](../../results/family_experiments/data_summary.csv)\n',
        '02_models.md': '# Mốc 2 — Huấn luyện và đánh giá\n\n' + common_note
            + 'Mỗi nhóm được huấn luyện một XGBoost riêng, 300 cây, depth 6, learning rate 0.08, seed 42; giữ cấu hình mô hình gốc.\n\n'
            + table(metrics[['family','n_train','n_test','accuracy','precision','recall','f1','roc_auc','tn','fp','fn','tp']]) + '\n\n'
            + 'FP: benign bị dự đoán là malware. FN: malware bị bỏ sót. Ransomware không sai mẫu nào; Spyware và Trojan mỗi nhóm bỏ sót một mẫu, không báo nhầm benign.\n\n'
            + 'Kết luận: ba mô hình đạt F1 rất cao trên holdout này. Không dùng chênh lệch rất nhỏ giữa chúng để kết luận nhóm nào dễ phát hiện hơn, vì tỷ lệ lớp và tập test khác nhau. Chưa đánh giá trên nguồn thu thập hoặc family con chưa từng thấy.\n\n'
            + '[Metrics đầy đủ](../../results/family_experiments/metrics_summary.csv)\n',
        '03_importance.md': '# Mốc 3 — Feature importance\n\n' + common_note
            + 'Đã tính importance cho 55 feature × 3 mô hình. MFI F1: mức giảm F1 khi hoán vị trên toàn test, 30 lần. MFI log-loss: mức tăng log-loss khi hoán vị. CFI: mức tăng log-loss khi thay feature bằng mẫu có điều kiện trên các feature còn lại, 5 draws mỗi dòng. Không so trực tiếp giá trị MFI F1 với CFI.\n\n'
            + 'Top-3 CFI từng bài toán:\n\n' + table(top) + '\n\n'
            + 'Ransomware: `svcscan.process_services` dẫn đầu; hai feature dịch vụ kế tiếp chưa đạt q < 0,05. Spyware: `svcscan.nservices` và `svcscan.process_services` có ảnh hưởng nổi bật và đạt q < 0,05. Trojan: `svcscan.process_services` dẫn đầu; `svcscan.nservices` cũng đạt q < 0,05, còn `callbacks.ncallbacks` đứng thứ hai nhưng chưa đạt ngưỡng.\n\n'
            + 'Kết luận: feature dẫn đầu của cả ba mô hình đều thuộc `svcscan`. Đây là kết quả theo từng feature, chưa phải phép đo importance của toàn nhóm. Top-k không đồng nghĩa mọi feature trong đó có ý nghĩa thống kê; importance không thể hiện chiều tăng/giảm của giá trị feature hay quan hệ nhân quả.\n\n'
            + '[Toàn bộ MFI và CFI](../../results/family_experiments/feature_importance_all.csv)\n\n'
            + '![Top-10 CFI](../../results/family_experiments/cfi_top10_comparison.png)\n',
        '04_verification_conclusions.md': '# Mốc 4 — Kiểm chứng và kết luận\n\n' + common_note
            + table(pd.DataFrame(verification_rows)) + '\n\n'
            + 'Mỗi bài toán vượt qua 20 kiểm tra: tách train/test, không trùng vector, nhất quán dữ liệu Python/R, căn chỉnh conditional draws, BH q-values, tính lại metrics từ model và top-3 CFI từ draws. Tổng hợp còn kiểm tra chỉ có đúng nhóm malware được chọn và nhãn khớp dòng nguồn.\n\n'
            + 'Diagnostics của conditional sampler:\n\n' + table(pd.DataFrame(diagnostic_rows)) + '\n\n'
            + 'Giá trị âm và ngoài khoảng train có thể trùng nhau. Coverage cao không đảm bảo mẫu sinh hợp lý; cần kiểm tra thêm phân phối có điều kiện. Các giá trị âm ở feature đếm là hạn chế thực tế của sampler, vì vậy CFI cần được đọc cùng MFI và diagnostics.\n\n'
            + 'Kết luận cuối: đã hoàn thành ba mô hình riêng và bảng importance của từng nhóm. Đặc trưng dịch vụ dẫn đầu lặp lại; chưa thể coi đây là dấu hiệu riêng của ransomware, spyware hoặc trojan. Kết quả thuộc một seed và benchmark này, không phải cam kết hiệu năng ngoài thực tế.\n\n'
            + source_links + '\n\n[Báo cáo tổng hợp](../family_results.md)\n',
    }
    for name, content in reports.items():
        (destination / name).write_text(content)
    index = '# Báo cáo theo cột mốc — từng nhóm malware + benign\n\n' + common_note
    for name, content in reports.items():
        title = content.splitlines()[0].removeprefix('# ')
        index += f'- [{title}]({name}) — hoàn thành.\n'
    index += '\n[Báo cáo tổng hợp và kết luận](../family_results.md)\n'
    (destination / 'README.md').write_text(index)


def summarize(base):
    metrics, importances, sections, leaders = [], [], [], []
    for family in FAMILIES:
        work = base / family.lower()
        receipt = json.loads((work / 'results/intermediate/verification.json').read_text())
        assert receipt['status'] == 'passed'
        profile = json.loads((work / 'results/intermediate/data_profile.json').read_text())
        saved = joblib.load(work / 'results/models/xgboost_model.joblib')
        split = joblib.load(work / 'results/intermediate/train_test_split.joblib')
        selected = pd.read_csv(work / CONFIG['data'])
        assert len(select_family(selected, family)) == len(selected)
        for part in ('train', 'test'):
            original = selected.iloc[split['source_rows'][split[f'{part}_indices']]]
            np.testing.assert_array_equal(original.Class.eq('Malware').astype(int), split[f'y_{part}'])
        tn, fp, fn, tp = confusion_matrix(saved['y_test'], saved['y_pred'], labels=[0, 1]).ravel()
        row = pd.read_csv(work / 'results/tables/model_metrics.csv').iloc[0].to_dict()
        row.update(family=family, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   benign_after_dedup=profile['class_counts']['0'], malware_after_dedup=profile['class_counts']['1'],
                   duplicates_removed=profile['duplicates_removed'])
        metrics.append(row)
        importance = pd.read_csv(work / 'results/tables/marginal_vs_conditional.csv')
        importance.insert(0, 'family', family)
        importances.append(importance)
        top = importance.sort_values('conditional_importance', ascending=False).head(10)
        significant = importance[importance.q_value.lt(CONFIG['alpha']) & importance.ci_lo.gt(0)]
        supported_top = top.head(3)[top.head(3).feature.isin(significant.feature)]
        leaders.append(dict(family=family, top_feature=top.iloc[0].feature,
                            cfi=top.iloc[0].conditional_importance, q_value=top.iloc[0].q_value,
                            significant_features=len(significant)))
        mfi = importance.sort_values('marginal_importance_mean', ascending=False).head(3)
        diag = pd.read_csv(work / 'results/tables/sampling_diagnostics.csv')
        diag = diag[diag.conditioning_set.eq('All other features')]
        sections.append(f'''## {family} + Benign

Sau loại trùng còn {profile['class_counts']['0']:,} benign và {profile['class_counts']['1']:,} {family}; loại {profile['duplicates_removed']:,} dòng trùng. Train/test: {row['n_train']:.0f}/{row['n_test']:.0f}.

F1 = {row['f1']:.6f}; recall = {row['recall']:.6f}; precision = {row['precision']:.6f}. Nhầm {fp} benign thành malware và bỏ sót {fn} malware trên test.

Top-10 CFI (độ tăng log-loss; càng lớn mô hình càng phụ thuộc vào feature khi đã xét các feature khác):

{table(top[['feature','conditional_importance','conditional_importance_se','q_value','coverage','marginal_logloss','marginal_importance_mean']])}

Theo MFI F1, ba feature đứng đầu là {', '.join('`'+f+'`' for f in mfi.feature)}. Theo CFI, ba feature đứng đầu là {', '.join('`'+f+'`' for f in top.head(3).feature)}. Có {len(significant)}/{len(importance)} feature có CFI dương, cận dưới một phía > 0 và q < {CONFIG['alpha']} sau BH trong bài toán này.

Trong top-3 CFI, các feature đạt ngưỡng trên là {', '.join('`'+f+'`' for f in supported_top.feature) or 'không có'}. Thứ hạng cao không tự động đồng nghĩa với ý nghĩa thống kê; ý nghĩa thống kê cũng không đảm bảo hiệu ứng lớn.

Kết luận: mô hình phân biệt {family} với benign chủ yếu dựa trên các feature dẫn đầu ở bảng trên trong phép đo CFI. Thứ hạng MFI khác CFI có thể phản ánh thông tin dùng chung giữa các feature và sự khác nhau của phép thay thế; không đủ để khẳng định cơ chế hoạt động của malware. Coverage CFI từ {importance.coverage.min():.2%} đến {importance.coverage.max():.2%}; sampler tạo {int(diag.n_negative.sum()):,} giá trị âm và {int(diag.n_outside_train_range.sum()):,} giá trị ngoài khoảng train (hai loại có thể trùng nhau), nên cần xem đây là bằng chứng phụ thuộc vào chất lượng sampler.

[Báo cáo đầy đủ](../results/family_experiments/{family.lower()}/report/research_results.md) · [Toàn bộ importance](../results/family_experiments/{family.lower()}/results/tables/marginal_vs_conditional.csv)
''')
    metrics = pd.DataFrame(metrics)
    importance = pd.concat(importances, ignore_index=True)
    metrics.to_csv(base / 'metrics_summary.csv', index=False)
    importance.to_csv(base / 'feature_importance_all.csv', index=False)
    ranks = importance.pivot(index='feature', columns='family', values='conditional_rank')
    ranks.to_csv(base / 'conditional_rank_comparison.csv')
    top_sets = [set(group.nsmallest(10, 'conditional_rank').feature) for _, group in importance.groupby('family')]
    common = sorted(set.intersection(*top_sets))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, family in zip(axes, FAMILIES):
        top = importance[importance.family.eq(family)].nlargest(10, 'conditional_importance').iloc[::-1]
        ax.barh(top.feature, top.conditional_importance, xerr=top.conditional_importance_se)
        ax.set(title=f'{family} vs Benign', xlabel='CFI: log-loss increase (± SE)')
    fig.tight_layout()
    fig.savefig(base / 'cfi_top10_comparison.png', dpi=160)
    plt.close(fig)
    report = '''# XGBoost và feature importance theo từng nhóm malware

Chạy riêng ba mô hình nhị phân: Ransomware + Benign, Spyware + Benign, Trojan + Benign. Đây là ba **nhóm malware cấp cao** theo ví dụ yêu cầu; chưa tách 15 family con như Conti, Emotet, Gator. Mỗi mô hình chỉ thấy một nhóm malware và benign; Class/Category không được dùng làm feature.

Giữ cấu hình pipeline gốc: 55 feature, loại trùng vector trước split stratified 80/20, median học trên train, XGBoost 300 cây/depth 6/learning rate 0.08/seed 42. Giữ toàn bộ benign, không cân bằng bằng cách bỏ bớt dữ liệu. MFI F1 dùng toàn bộ test với 30 permutations; MFI log-loss và CFI dùng 1.500 dòng test, ARF học trên 3.000 dòng train, 5 conditional draws. MFI log-loss trong bảng được tính trên đúng các dòng hợp lệ của CFI cho từng feature. SE của CFI đo biến thiên giữa các dòng sau khi trung bình draws; không bao gồm bất định do train lại mô hình/sampler.

Các bài toán có tỷ lệ lớp và tập test riêng, vì vậy không dùng chênh lệch nhỏ về accuracy/importance giữa chúng để kết luận nhóm nào dễ phát hiện hơn. Đây là holdout ngẫu nhiên, không phải đánh giá trên family con hoặc nguồn thu thập chưa từng thấy.

## Hiệu năng

''' + table(metrics[['family','n_train','n_test','accuracy','precision','recall','f1','roc_auc','fp','fn']]) + '\n\n' + '\n'.join(sections)
    report += '\n## So sánh và kết luận chung\n\n'
    report += table(pd.DataFrame(leaders)) + '\n\n'
    if all(row['top_feature'].startswith('svcscan.') for row in leaders):
        report += 'Trong cả ba mô hình, feature có CFI lớn nhất đều thuộc nhóm `svcscan` (đặc trưng về dịch vụ). Vì vậy, nhóm dịch vụ là điểm nổi bật nhất lặp lại ở cả ba bài toán theo phép đo này. Đây là nhận xét về feature đứng đầu; chưa đo importance của cả nhóm khi thay thế đồng thời.\n\n'
    report += 'Các feature cùng nằm trong top-10 CFI của cả ba bài toán: ' + ', '.join(f'`{f}`' for f in common) + '.\n\n'
    report += 'Các feature chung được xếp hạng cao lặp lại trong benchmark; điều này chưa đồng nghĩa tất cả đều có CFI dương có ý nghĩa thống kê hoặc là dấu hiệu đặc thù của từng nhóm. Để chứng minh tính đặc thù cần thêm bài toán phân biệt các nhóm malware với nhau hoặc kiểm định trên family con chưa thấy. CFI nhỏ không chứng minh feature vô ích: tín hiệu có thể được các feature tương quan thay thế. Importance không phải quan hệ nhân quả hay phần trăm đóng góp. Kết quả hiện tại dùng một seed và chưa chứng minh khả năng tổng quát hóa ngoài benchmark.\n\n'
    report += '![Top-10 CFI từng nhóm](../results/family_experiments/cfi_top10_comparison.png)\n\n'
    report += 'Đã kiểm tra nhãn/nguồn dòng theo từng nhóm, không trùng vector train/test, tính lại metrics từ model lưu và tính lại top-3 CFI từ conditional draws cho cả ba mô hình.\n'
    report += '\n[Báo cáo riêng theo bốn cột mốc](family_milestones/README.md).\n'
    (ROOT / 'report/family_results.md').write_text(report)
    write_milestone_reports(base, metrics, importance)
    (base / 'verification.json').write_text(json.dumps(dict(status='passed', families=list(FAMILIES),
        checks=['Existing full pipeline verification for each type', 'Only selected malware type plus benign',
                'Saved split labels match selected source rows']), indent=2))
    print(metrics[['family','f1','roc_auc','fp','fn']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary-only', action='store_true', help='Summarize completed, verified runs')
    parser.add_argument('--resume', action='store_true', help='Verify and reuse completed runs; restart incomplete ones')
    args = parser.parse_args()
    base = ROOT / 'results/family_experiments'
    if not args.summary_only:
        frame = pd.read_csv(ROOT / CONFIG['data'])
        for family in FAMILIES:
            existing = base / family.lower()
            receipt = existing / 'results/intermediate/verification.json'
            if args.resume and receipt.exists():
                selection = json.loads((existing / 'selection.json').read_text())
                same_data = selection['source_sha256'] == hashlib.sha256((ROOT / CONFIG['data']).read_bytes()).hexdigest()
                same_config = json.loads((existing / 'config.json').read_text()) == CONFIG
                if same_data and same_config:
                    subprocess.run([sys.executable, str(existing / 'python/verify_results.py')],
                                   cwd=existing, check=True, stdout=subprocess.DEVNULL)
                    print(f'Reusing verified {family}.', flush=True)
                    continue
            if receipt.exists():
                receipt.unlink()  # A new run must not inherit an old success receipt.
            work = prepare_run(frame, family, base)
            print(f'Running {family}; log: {work / "pipeline.log"}', flush=True)
            env = dict(os.environ, PYTHON=sys.executable)
            with (work / 'pipeline.log').open('w') as log:
                subprocess.run(['bash', str(work / 'run_all.sh')], cwd=work, env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'{family} verified.', flush=True)
    summarize(base)


if __name__ == '__main__':
    main()
