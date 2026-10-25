# Đưa mã nguồn lên GitHub và bàn giao

Repository của nhóm: [nguyenngoc1211/PTMD](https://github.com/nguyenngoc1211/PTMD).
Người nhận bàn giao bắt đầu từ [hướng dẫn chạy](REPRODUCIBILITY_VI.md); các bước
khởi tạo/push bên dưới dành cho người quản lý repository.

Repository cần chứa code, cấu hình, manifest dữ liệu, hướng dẫn và tests. CSV,
dependencies cài tại máy và output lớn đã được `.gitignore` loại khỏi Git. Các
báo cáo trong `report/` là artifact cục bộ, không có sẵn trong clone mới.

## 1. Chuẩn bị repository đích

Tạo repository trống trên GitHub, ví dụ `PTMD`; chọn private nếu đây là nội dung
nghiên cứu nội bộ. Không tạo README/.gitignore từ giao diện vì project đã có.
Thêm thành viên nhóm quyền đọc hoặc cộng tác theo nhu cầu. Dùng SSH hoặc đăng nhập
HTTPS qua trình quản lý credential của Git; không đặt token trong URL remote,
file cấu hình, tài liệu hoặc lệnh được lưu trong repo.

## 2. Kiểm tra trước lần push đầu

Từ gốc project:

```bash
git status --short
git remote -v
git check-ignore data/Obfuscated-MalMem2022.csv .venv RLibrary results report
MPLCONFIGDIR="$PWD/.cache/matplotlib" .venv/bin/python -m unittest discover -s tests -v
bash -n run_all.sh
```

Stage đúng phần bàn giao, kiểm tra rồi tạo commit:

```bash
git add .gitignore .github README.md config.json run_all.sh \
  requirements.txt requirements-lock.txt requirements_R.txt \
  python r tests docs data/README.md data/manifests
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
git commit -m "Prepare reproducible malware feature experiments and dataset handoff"
git branch -M main
```

Danh sách staged không được có CSV, `.venv/`, `RLibrary/`, `results/`, `.env` hoặc
credential. `research_report/` chứa ghi chú nghiên cứu cục bộ, không nằm trong
danh sách stage trên; nếu nhóm muốn công bố, đọc lại nội dung và quyết định riêng.
Chưa tự gán giấy phép mã nguồn: quyền phân phối code/dataset cần theo quyết định
của nhóm và nguồn tương ứng.

Thay URL thật và push (nếu chưa có remote):

```bash
git remote add origin https://github.com/nguyenngoc1211/PTMD.git
git push -u origin main
git rev-parse HEAD
```

Nếu đã có `origin`, xem lại URL bằng `git remote -v`; chỉ dùng `git remote set-url`
khi đã xác định đích đúng. Nếu remote có lịch sử riêng, fetch và xem khác biệt trước;
không force-push để bỏ qua lỗi. Không cần push CSV để người khác chạy được: gửi
nguồn tải/phiên bản/hash trong manifest.

## 3. Thông tin bàn giao cho thành viên

Gửi URL repo, commit SHA đã chốt, quyền truy cập dataset B (nếu chưa công khai),
deadline và đường dẫn gửi artifact. Có thể dùng nội dung sau:

> Nhờ bạn kiểm chứng pipeline XGBoost + ARF/CFI tại repo
> https://github.com/nguyenngoc1211/PTMD, commit `<SHA>`.
> Đọc `docs/REPRODUCIBILITY_VI.md`, cài môi trường và chạy tests. Chạy MalMem2022
> trước (smoke rồi full seed 42,43,44), sau đó lập manifest cho dataset B từ nguồn
> độc lập và chạy cùng cấu hình. Gửi lại biên bản theo
> `docs/EXPERIMENT_RESULT_TEMPLATE.md`, bảng tổng hợp, báo cáo từng seed,
> verification, log, phiên bản môi trường và link gói artifact. Ghi cả lỗi hoặc
> kết quả thấp; không đổi cấu hình theo test để làm đẹp kết quả. Chưa gọi đây là
> kiểm tra train MalMem → test dataset B vì mỗi dataset được train riêng.

Người chạy nên dùng thư mục output mới và ghi máy/ngày chạy trong biên bản. Nếu
sửa code/schema, tạo branch và commit riêng, ghi SHA mới; không gộp kết quả của
hai phiên bản thành một bảng mà thiếu chú thích.

## 4. Tiêu chí nghiệm thu

- Clone đúng commit; tests Python qua; cài và ghi phiên bản R thực tế.
- MalMem đúng hash; dataset B có mô tả nguồn độc lập và danh sách feature đã rà soát.
- Mọi seed full có `verification.json` với `status: passed`, không nhầm với smoke.
- Đủ metrics, CFI/coverage/diagnostics, cấu hình, manifest và log cho cả A/B.
- Kết luận phân biệt tái lập, lặp phương pháp trên dataset khác và chuyển mô hình giữa nguồn.

GitHub Actions kiểm tra Python khi push/PR. Dấu xanh CI không thay thế kết quả chạy
R, kiểm tra dataset thật hay đánh giá giá trị khoa học của thí nghiệm.
