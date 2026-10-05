"""Sinh lại Thuyet_minh_nhiem_vu_TV2.pdf (ở gốc repo) từ nội dung khai trong file này.

Sửa nội dung thuyết minh thì sửa các hằng số ở phần "Nội dung" rồi chạy lại:

    pip install reportlab          # chỉ script này cần, pipeline dữ liệu không cần
    python Golub_DeTai10/TV2/tao_thuyet_minh_pdf.py

Font DejaVu Sans lấy từ thư mục dữ liệu của matplotlib (có sẵn trong môi trường dự án) nên
chạy được trên mọi hệ điều hành và hiển thị đủ dấu tiếng Việt. Không dùng DejaVu Sans Mono:
bản đó thiếu nhiều chữ Việt có hai dấu (ẫ, ố, ả, ấ…) và in ra ô vuông.

Căn cứ nội dung: Ke_hoach_trien_khai_project_Golub.pdf — mục 5.1 (Bảng 5.2), 5.2 (công việc
1.1), 5.3 (công việc 2.4), Bảng 2.4, 2.5, 2.6, 2.9 và mục 3.5.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle,
)

OUT = Path(__file__).resolve().parents[2] / "Thuyet_minh_nhiem_vu_TV2.pdf"
HEADER = "DỰ ÁN GOLUB ET AL. (1999) — THUYẾT MINH NHIỆM VỤ TV2"
FOOTER = "Tài liệu lưu hành nội bộ nhóm · Bản sửa 06/10/2026"

# ---------------------------------------------------------------------------
# Nội dung
# ---------------------------------------------------------------------------
TITLE = "THUYẾT MINH NHIỆM VỤ THÀNH VIÊN 2 (TV2)"
SUBTITLE = "Dự án: Phân loại bạch cầu cấp ALL/AML từ dữ liệu biểu hiện gen (Golub et al., 1999)"
INFO = [("Phân công", "Thành viên 2 (TV2)"),
        ("Phạm vi", "Công việc 1.1 (GĐ1) và 2.4 (GĐ2)"),
        ("Thời gian", "Tuần 1 – Tuần 7")]
OVERVIEW = (
    "<b>TỔNG QUAN VAI TRÒ:</b> Theo Bảng 5.2 của kế hoạch, <b>TV2</b> là người dựng <b>hạ tầng dữ "
    "liệu</b> ở Giai đoạn 1 (công việc 1.1: nạp, ghép dữ liệu và kiểm tra chất lượng), rồi dựng tiếp "
    "<b>hạ tầng đánh giá dùng chung</b> ở Giai đoạn 2 (công việc 2.4: khung đánh giá và thí nghiệm "
    "<b>selection bias</b>, giả thuyết H3). Đầu ra của 1.1 là đầu vào của mọi công việc khác; hàm "
    "đánh giá của 2.4 được TV1 và TV3 dùng từ tuần 5."
)

GD1_GOAL = ("Đưa bộ dữ liệu Golub từ Kaggle và metadata OpenIntro vào dạng dùng được cho cả nhóm, "
            "đáng tin cậy và tái lập được.")
GD1_TASKS = [
    ("Quản trị dữ liệu thô",
     "Tải 3 file Kaggle (data_set_ALL_AML_train.csv, data_set_ALL_AML_independent.csv, actual.csv) và "
     "golub.csv của OpenIntro vào data/raw/; ghi data/raw/SOURCES.md gồm <b>link, ngày tải và MD5</b> "
     "từng file để chứng minh dữ liệu gốc không bị thay đổi."),
    ("Thiết lập hạ tầng dự án",
     "Dựng repo GitHub, cấu trúc thư mục theo mục 2.2, environment.yml <b>cố định phiên bản</b> Python và "
     "thư viện (mục 2.7), Makefile (make data, make test…), src/config.py."),
    ("Module nạp và ghép dữ liệu (src/load.py)",
     "Theo đúng 7 bước mục 2.1.2: ghép theo <b>mã bệnh nhân (sample_id), không bao giờ theo thứ tự "
     "dòng</b>; đọc OpenIntro chỉ 6 cột mô tả (usecols); left join metadata vào bảng mẫu với "
     "validate=\"one_to_one\"; kiểm tra sau ghép (đủ 72 mẫu, không mẫu nào thiếu metadata, class khớp "
     "subtype 72/72, tissue.mf khớp tissue + sex). Xuất 4 file tầng standardized (expression, calls, "
     "samples, genes) và hàm <b>load_golub()</b> cho cả nhóm."),
    ("Module kiểm tra chất lượng (src/quality.py)",
     "Kiểm tra đủ <b>4 mức của Bảng 2.9</b>: Value (NaN, giá trị &lt; 100 và &gt; 16000), Value-set (probe "
     "AFFX, nhiều probe cho cùng một gen, metadata ngoài danh mục), Record (mẫu trùng lặp; mẫu bất "
     "thường qua tương quan giữa các mẫu và boxplot theo mẫu → <b>gắn cờ qc_outlier, không xóa</b>), "
     "Relation (nhãn Kaggle khớp subtype OpenIntro, phân bố Bảng 2.4). Ghi tầng cleansed và "
     "quality_report.md; có kiểm tra FAIL thì dừng pipeline."),
    ("Kiểm thử tự động và tài liệu",
     "tests/test_load.py kiểm tra <b>toàn bộ Bảng 2.4</b> (kích thước, class, split × class, split × "
     "tissue, nguồn mẫu train/test, subtype, khớp nhãn 72/72); docs/data_dictionary.md; soạn mục "
     "2.1–2.5 và 2.7 của báo cáo cuối kỳ."),
]
GD1_MILESTONES = [
    ("T7 10/10", "Hạn hoàn thành công việc 1.1", "Cả nhóm"),
    ("CN 11/10", "Giao load_golub() bản tạm", "TV3 (công việc 1.2)"),
    ("CN 18/10", "Giao samples.parquet chính thức", "TV1 (công việc 1.3)"),
    ("CN 01/11", "Trình bày 10 phút, chạy demo nghiệm thu; nộp docs/handover/1.1.md", "Trưởng nhóm (TV1)"),
]
GD1_PRODUCTS = [
    ("Mã nguồn xử lý", "src/load.py\nsrc/quality.py", "Nạp/ghép dữ liệu; kiểm tra chất lượng 4 mức"),
    ("Mã nguồn kiểm thử", "tests/test_load.py", "Kiểm tra toàn bộ Bảng 2.4 sau khi ghép"),
    ("Tầng standardized", "data/standardized/*.parquet",
     "expression, calls, samples, genes — cùng cấu trúc, chưa đánh giá chất lượng"),
    ("Tầng cleansed", "data/cleansed/*",
     "Đã kiểm tra chất lượng; giá trị biểu hiện giữ nguyên, samples thêm cột qc_outlier. Tầng sẵn sàng "
     "cho phân tích là curated (TV3)"),
    ("Báo cáo chất lượng", "data/cleansed/quality_report.md",
     "Sinh tự động bằng make quality; không đưa lên Git (data/ nằm trong .gitignore)"),
    ("Tài liệu kỹ thuật", "docs/data_dictionary.md\ndocs/decisions.md", "Từ điển dữ liệu; nhật ký quyết định"),
    ("Báo cáo", "Mục 2.1–2.5, 2.7", "Nguồn dữ liệu, lưu trữ, mô hình dữ liệu, pipeline, chất lượng, quản trị"),
]
GD1_DEMO = """$ make data && pytest tests/test_load.py
>>> from src.load import load_golub
>>> X, samples = load_golub(layer='cleansed')
>>> X.shape, samples.shape
((72, 7129), (72, 7))        # 6 cột mô tả + cột cờ qc_outlier
>>> samples['class'].value_counts()                   # ALL 47, AML 25
>>> pd.crosstab(samples['split'], samples['class'])   # train 27/11, test 20/14
Trích quality_report.md:
[PASS] 72 mẫu, 7129 probe, không trùng sample_id
[PASS] Nhãn Kaggle khớp subtype OpenIntro: 72/72
[INFO] Số giá trị < 100: … Số giá trị > 16000: … Probe AFFX: …
[INFO] Mẫu gắn cờ qc_outlier: … (lý do: …)"""

GD2_GOAL = ("Đảm bảo mọi con số hiệu năng là trung thực và chứng minh bằng thực nghiệm tác hại của "
            "đánh giá sai (giả thuyết H3).")
GD2_TASKS = [
    ("Khung đánh giá dùng chung (src/evaluate.py)",
     "Hàm <b>evaluate(pipeline, X, y, scheme, seeds)</b> cho 4 sơ đồ của mục 3.5.1: original_split "
     "(38/34), loocv (trên 38 mẫu train, chọn gen lại ở mỗi fold, chỉ số tính trên dự đoán gộp), "
     "nested_cv (stratified lặp lại, vòng trong chọn k và siêu tham số) và wrong_cv (chọn gen trên toàn "
     "bộ dữ liệu rồi mới CV — chỉ để minh họa). Số fold, số lần lặp và seed đọc từ "
     "config/experiment_config.yaml của TV1. Kết quả trả về gồm bal_acc_mean, bal_acc_sd, auc_mean, "
     "per_fold, config (và predictions để kiểm định, vẽ ROC)."),
    ("Chỉ số và kiểm định thống kê",
     "Balanced accuracy (chỉ số chính), AUC-ROC, sensitivity và specificity (<b>AML là lớp dương</b>); "
     "khoảng tin cậy Wilson 95% cho accuracy trên tập test; trung bình ± độ lệch chuẩn qua các lần lặp "
     "CV; kiểm định McNemar cho cặp mô hình trên tập test."),
    ("Thí nghiệm selection bias (H3)",
     "Bảng <b>2 × 2</b>: nhãn thật / nhãn hoán vị ngẫu nhiên (100 lần) × CV đúng (chọn gen trong "
     "pipeline) / CV sai (chọn gen trước CV). Với nhãn hoán vị, CV đúng phải quanh 0.50; CV sai cao bất "
     "thường là bằng chứng H3. Vẽ hình F10 (violin/strip plot, đường tham chiếu 0.5)."),
    ("Đánh giá mô hình cuối",
     "Vẽ đường cong ROC và ma trận nhầm lẫn cho mô hình cuối được nhóm thống nhất chọn (hình F11)."),
    ("Rà soát rò rỉ dữ liệu",
     "Review code của TV1, TV3, TV4: mọi bước học từ dữ liệu (lọc gen, chuẩn hóa, chọn gen) phải nằm "
     "trong Pipeline và chỉ fit trên phần huấn luyện của từng fold; không bước nào nhìn thấy fold "
     "kiểm tra hay tập test độc lập."),
]
GD2_MILESTONES = [
    ("Đầu tuần 5", "Giao evaluate.py bản 1 (hàm evaluate dùng chung)", "TV1 (2.1), TV3 (2.3)"),
    ("Tuần 5", "Nhận config/experiment_config.yaml; giao định dạng file số liệu F10/F11", "TV1 / TV6"),
    ("Tuần 5–7", "Review rò rỉ dữ liệu khi có code", "TV1, TV3, TV4"),
    ("Tuần 6–7", "Giao số liệu thật cho F10, F11; viết mục 3.5", "TV6, trưởng nhóm"),
]
GD2_PRODUCTS = [
    ("Khung đánh giá", "src/evaluate.py",
     "evaluate(), wilson_ci(), mcnemar_test(), selection_bias_experiment()"),
    ("Mã nguồn kiểm thử", "tests/test_evaluate.py", "Kiểm tra 4 sơ đồ, Wilson, McNemar trên dữ liệu tổng hợp"),
    ("Notebook thí nghiệm", "notebooks/08_evaluation.ipynb", "Thí nghiệm selection bias và kiểm định McNemar"),
    ("Hình F10", "results/figures/report/\nF10_selection_bias_report.png\n"
                 "results/figures/slides/\nF10_selection_bias_slide.png",
     "Phân phối balanced accuracy: CV đúng vs CV sai, nhãn thật vs hoán vị"),
    ("Hình F11", "results/figures/report/\nF11_final_model_roc_cm_report.png\n"
                 "results/figures/slides/\nF11_final_model_roc_cm_slide.png",
     "ROC và ma trận nhầm lẫn của mô hình cuối"),
    ("Bảng tổng hợp chỉ số", "results/metrics/summary_*.csv", "Đưa thẳng vào báo cáo (Bảng 2.6)"),
    ("Báo cáo", "Mục 3.5", "Đánh giá mô hình và thí nghiệm selection bias"),
]
GD2_DEMO = """>>> from src.evaluate import evaluate
>>> res = evaluate(pipe, X, y, scheme='nested_cv', seeds=range(10))
>>> res.keys()
dict_keys(['bal_acc_mean', 'bal_acc_sd', 'auc_mean', 'per_fold', 'config', 'predictions'])"""
# Bảng selection bias của demo nghiệm thu; cột cuối là kỳ vọng ghi trong kế hoạch.
GD2_BIAS_TABLE = [
    ("Thật", "Đúng", "… ± …", "cao"),
    ("Thật", "Sai", "… ± …", "cao, có thể cao hơn CV đúng"),
    ("Hoán vị", "Đúng", "… ± …", "xấp xỉ 0.50"),
    ("Hoán vị", "Sai", "… ± …", "cao bất thường (bằng chứng H3)"),
]

SUMMARY = [
    ("Giai đoạn 1\n(Tuần 1–4)", "1.1. Nạp, ghép dữ liệu và kiểm tra chất lượng",
     "• src/load.py, src/quality.py\n• tests/test_load.py\n• data/standardized/, data/cleansed/\n"
     "• docs/data_dictionary.md\n• Báo cáo mục 2.1–2.5, 2.7",
     "Dữ liệu đáng tin cậy, truy vết được về file gốc, tái lập bằng một lệnh cho cả nhóm."),
    ("Giai đoạn 2\n(Tuần 5–7)", "2.4. Khung đánh giá và thí nghiệm selection bias",
     "• src/evaluate.py\n• notebooks/08_evaluation.ipynb\n• Hình F10, F11\n• Bảng tổng hợp chỉ số\n"
     "• Báo cáo mục 3.5",
     "Mọi con số hiệu năng trung thực; chứng minh H3; rà soát chống rò rỉ dữ liệu."),
]

# ---------------------------------------------------------------------------
# Trình bày
# ---------------------------------------------------------------------------
NAVY = colors.HexColor("#1B2A4A")
BLUE = colors.HexColor("#2E75B6")
GREY = colors.HexColor("#7F8C8D")
LIGHT = colors.HexColor("#F2F5F9")
RULE = colors.HexColor("#D5DBE3")


def _register_fonts() -> None:
    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(ttf / file)))
    pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                  italic="DejaVu", boldItalic="DejaVu-Bold")


def _styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("body", fontName="DejaVu", fontSize=8.6, leading=12.2, alignment=TA_JUSTIFY)
    return {
        "body": base,
        "title": ParagraphStyle("title", base, fontName="DejaVu-Bold", fontSize=17, leading=21, textColor=NAVY),
        "subtitle": ParagraphStyle("subtitle", base, fontSize=9.5, leading=13, textColor=BLUE),
        "h1": ParagraphStyle("h1", base, fontName="DejaVu-Bold", fontSize=12, leading=15.5, textColor=NAVY,
                             spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", base, fontName="DejaVu-Bold", fontSize=9.6, leading=13, textColor=BLUE,
                             spaceBefore=8, spaceAfter=4),
        "bullet": ParagraphStyle("bullet", base, spaceAfter=4),
        "cell": ParagraphStyle("cell", base, fontSize=7.6, leading=10, alignment=0),
        "cellb": ParagraphStyle("cellb", base, fontName="DejaVu-Bold", fontSize=7.6, leading=10, alignment=0),
        "head": ParagraphStyle("head", base, fontName="DejaVu-Bold", fontSize=7.8, leading=10,
                               textColor=colors.white, alignment=0),
        "code": ParagraphStyle("code", fontName="DejaVu", fontSize=7.4, leading=9.8, textColor=NAVY),
    }


def _decorate(canvas, doc) -> None:
    w, h = letter
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 6 * mm, w, 6 * mm, stroke=0, fill=1)
    canvas.setFont("DejaVu", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(doc.leftMargin, h - 12 * mm, HEADER)
    canvas.setStrokeColor(RULE)
    canvas.line(doc.leftMargin, h - 14 * mm, w - doc.rightMargin, h - 14 * mm)
    canvas.line(doc.leftMargin, 14 * mm, w - doc.rightMargin, 14 * mm)
    canvas.drawString(doc.leftMargin, 10 * mm, FOOTER)
    canvas.drawRightString(w - doc.rightMargin, 10 * mm, f"Trang {doc.page}")
    canvas.restoreState()


def _table(rows, header, widths, st, bold_col=1) -> Table:
    def cell(text, col):
        return Paragraph(str(text).replace("\n", "<br/>"), st["cellb"] if col == bold_col else st["cell"])
    data = [[Paragraph(h, st["head"]) for h in header]]
    data += [[cell(v, i) for i, v in enumerate(r)] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _box(flowable, width, border=BLUE, background=LIGHT) -> Table:
    t = Table([[flowable]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("LINEBEFORE", (0, 0), (0, -1), 3, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def _section(story, st, width, title, goal, tasks, milestones, products, products_title, demo, demo_title,
             demo_extra=None):
    story.append(Paragraph(title, st["h1"]))
    story.append(Paragraph(f"<b>Mục tiêu:</b> {goal}", st["body"]))
    story.append(Paragraph("Các đầu việc chi tiết:", st["h2"]))
    for lead, text in tasks:
        story.append(Paragraph(f"• <b>{lead}:</b> {text}", st["bullet"]))
    story.append(KeepTogether([
        Paragraph("Mốc phối hợp (theo kế hoạch):", st["h2"]),
        _table(milestones, ["Mốc", "Nội dung", "Người nhận / liên quan"],
               [width * 0.14, width * 0.56, width * 0.30], st, bold_col=0),
    ]))
    story.append(KeepTogether([
        Paragraph(products_title, st["h2"]),
        _table(products, ["Hạng mục", "Tên file / Sản phẩm", "Mô tả / Vị trí"],
               [width * 0.20, width * 0.40, width * 0.40], st),
    ]))
    block = [Paragraph(demo_title, st["h2"]), _box(Preformatted(demo, st["code"]), width, border=NAVY)]
    if demo_extra is not None:
        block += [Spacer(1, 6), demo_extra]
    story.append(KeepTogether(block))


def build(out: Path = OUT) -> Path:
    _register_fonts()
    st = _styles()
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=22 * mm, bottomMargin=20 * mm, title=TITLE, author="TV2",
                            subject="Thuyết minh nhiệm vụ TV2 — dự án Golub")
    width = doc.width
    story = [Spacer(1, 6), Paragraph(TITLE, st["title"]), Spacer(1, 3), Paragraph(SUBTITLE, st["subtitle"]),
             Spacer(1, 8)]
    info = Table([[Paragraph(f"<b>{k}:</b> {v}", st["cell"]) for k, v in INFO]], colWidths=[width / 3] * 3)
    info.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                              ("LINEABOVE", (0, 0), (-1, 0), 2, NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                              ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [info, Spacer(1, 10), _box(Paragraph(OVERVIEW, st["body"]), width)]

    _section(story, st, width,
             "1. GIAI ĐOẠN 1 — CÔNG VIỆC 1.1: NẠP, GHÉP DỮ LIỆU VÀ KIỂM TRA CHẤT LƯỢNG "
             "(TUẦN 1–4, DEADLINE 10/10)",
             GD1_GOAL, GD1_TASKS, GD1_MILESTONES, GD1_PRODUCTS, "Sản phẩm bàn giao Giai đoạn 1:",
             GD1_DEMO, "Demo nghiệm thu (mục 5.2 của kế hoạch; số liệu phải khớp Bảng 2.4):")
    _section(story, st, width,
             "2. GIAI ĐOẠN 2 — CÔNG VIỆC 2.4: KHUNG ĐÁNH GIÁ VÀ THÍ NGHIỆM SELECTION BIAS (TUẦN 5–7)",
             GD2_GOAL, GD2_TASKS, GD2_MILESTONES, GD2_PRODUCTS, "Sản phẩm bàn giao Giai đoạn 2:",
             GD2_DEMO, "Demo nghiệm thu (mục 5.3 của kế hoạch):",
             demo_extra=_table(GD2_BIAS_TABLE, ["Nhãn", "Cách CV", "Balanced accuracy (TB ± SD)", "Kỳ vọng"],
                               [width * 0.15, width * 0.15, width * 0.30, width * 0.40], st, bold_col=0))

    story.append(KeepTogether([
        Paragraph("3. BẢNG TỔNG HỢP TOÀN BỘ ĐẦU VIỆC CỦA TV2", st["h1"]),
        _table(SUMMARY, ["Giai đoạn", "Nhiệm vụ chính", "Sản phẩm cụ thể", "Mục đích & ý nghĩa"],
               [width * 0.14, width * 0.24, width * 0.32, width * 0.30], st),
    ]))
    doc.build(story, onFirstPage=_decorate, onLaterPages=_decorate)
    return out


if __name__ == "__main__":
    print(f"Đã ghi {build()}")
