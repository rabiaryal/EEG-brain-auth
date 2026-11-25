#!/usr/bin/env python
"""Build the 2-page summary PDF (results table + 4 figures with captions).

All numbers are read from the results JSON files via make_figures.collect().
Usage: python scripts/make_summary_pdf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import adv_at, collect  # noqa: E402

OUT_PDF = ROOT / "report" / "Rabi_Aryal_EEG_Authentication_Summary.pdf"
FIG = ROOT / "results" / "final"
AUTHOR = "Rabi Aryal"
REPO = "github.com/rabiaryal/ECG-brain-auth"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9, leading=12)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10)
CAP = ParagraphStyle("c", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#333333"))
H1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=17, leading=21, alignment=0, spaceAfter=2)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Helvetica-Bold", fontSize=11, leading=14, spaceBefore=8, spaceAfter=3)


def p(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def build_rows(S: dict) -> list[str]:
    b, c = S["baseline"]["models"], S["cnn"]
    a = S["attacks"]["attacks"]
    d = S["defenses"]["adversarial_training"]
    cr = S["defenses"]["challenge_response"]
    pgd = lambda rows, e: p(adv_at(rows, e)["acceptance_rate"])
    return [
        " / ".join(p(x, 1) for x in (b["logreg"]["identification_accuracy"], b["svm"]["identification_accuracy"], c["identification_accuracy"])),
        " / ".join(p(x, 2) for x in (b["logreg"]["eer"], b["svm"]["eer"], c["eer"])),
        p(a["impostor_zero_effort"]["far_at_threshold"], 2),
        p(a["replay"]["acceptance_rate_test_recordings"]),
        f"{pgd(a['adversarial']['pgd'], 0.05)} / {pgd(a['adversarial']['pgd'], 0.1)}",
        f"{pgd(d['attacks_after']['pgd'], 0.05)} / {pgd(d['attacks_after']['pgd'], 0.1)}",
        f"{p(d['clean_before']['eer'], 2)} → {p(d['clean_after']['eer'], 2)}",
        " / ".join(p(cr["replay_success_by_library_size"][str(k)]) for k in (1, 2, 3, 4)),
    ]


def fig(name: str, width_cm: float) -> Image:
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(FIG / name)).getSize()
    return Image(str(FIG / name), width=width_cm * cm, height=width_cm * cm * h / w)


def main() -> None:
    sets = collect()
    s20, s106 = sets["20 subjects"], sets["all screened subjects"]
    n106 = s106["stats"]["n_subjects"]
    tr, te = s106["stats"]["n_train_clips"], s106["stats"]["n_test_clips"]
    val_frac = 0.15  # configs/*.yaml: models.cnn.val_fraction

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT_PDF), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.3 * cm, title="EEG-Based User Authentication: Summary",
                            author=AUTHOR)
    W = A4[0] - 3.6 * cm
    st = []
    st.append(Paragraph("EEG-Based User Authentication: Attacks and Defenses", H1))
    st.append(Paragraph(f"{AUTHOR} &nbsp;|&nbsp; Code and full results: {REPO}", SMALL))
    st.append(Paragraph("Goal and setup", H2))
    st.append(Paragraph(
        f"<b>Goal:</b> check whether 2-second EEG clips can verify a person's identity, and how easily the verifier can be "
        f"fooled by impostors, replayed recordings and adversarial perturbations. "
        f"<b>Data:</b> public PhysioNet EEGMMIDB (64 channels, 160 Hz), {n106} subjects "
        f"(3 of 109 excluded: runs not at 160 Hz), band-pass 1–40 Hz, 2 s clips. "
        f"<b>Split (by recording run, never by clip):</b> train = runs 1, 3, 4, 5 ({tr:,} clips); "
        f"test = runs 2, 6, 7 ({te:,} clips); validation = {int(val_frac * 100)}% of the training clips, held out for early "
        f"stopping, so test runs are never used for tuning. Models: logistic regression and SVM on band-power features, "
        f"and a small 1D CNN.", BODY))
    st.append(Paragraph("Main results (test runs, unseen during training)", H2))

    labels = ["Metric", f"{20} subjects", f"{n106} subjects"]
    names = [
        "Identification accuracy (LogReg / SVM / CNN)",
        "Verification EER, lower is better (LogReg / SVM / CNN)",
        "Random impostor accepted (at EER threshold)",
        "Replay of a genuine recording accepted",
        "Adversarial (PGD) attack accepted, ε = 0.05 / 0.1",
        "Same attack after adversarial training",
        "Clean EER: before → after adversarial training",
        "Replay success with challenge-response (1 / 2 / 3 / 4 task types recorded)",
    ]
    r20, r106 = build_rows(s20), build_rows(s106)
    data = [labels] + [[Paragraph(n, SMALL), Paragraph(a, SMALL), Paragraph(f"<b>{b}</b>", SMALL)]
                       for n, a, b in zip(names, r20, r106)]
    data[0] = [Paragraph(f"<b>{x}</b>", SMALL) for x in labels]
    t = Table(data, colWidths=[W * 0.42, W * 0.27, W * 0.31], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf5")),
        ("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#f4f8fc")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2cc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    st.append(t)
    st.append(Spacer(1, 6))
    st.append(Paragraph(
        "<b>Reading the table.</b> Plain band-power logistic regression is the best verifier (about 2% EER even with "
        f"{n106} people); the CNN does not beat it. Replaying a genuine recording is accepted almost every time, and a small "
        "adversarial perturbation (ε = 0.1 is about 14% of the signal size) gets an impostor accepted about 94% of the time. "
        "Adversarial training lowers that only to about 87% and raises the clean error. The challenge-response check is a "
        "simulation on the dataset's task labels, with a weak task classifier (46% accuracy on four tasks).", BODY))

    # ---- page 2 ----
    from reportlab.platypus import PageBreak

    st.append(PageBreak())
    st.append(Paragraph("Figures", H2))
    figs = [
        ("fig_models.png", 14.2, "<b>Figure 1.</b> Model comparison: identification accuracy and verification EER for logistic "
         "regression, SVM and CNN on 20 and 106 subjects. Band-power logistic regression is best; accuracy drops a few points with more subjects."),
        ("fig_epsilon_curve.png", 14.2, "<b>Figure 2.</b> Attack success as the perturbation grows: the share of targeted FGSM/PGD "
         "impostor attempts accepted rises from the zero-effort level to about 95% at ε = 0.1 (shaded band: 95% interval)."),
        ("fig_before_after.png", 14.2, "<b>Figure 3.</b> Before vs after adversarial training: attack success at three perturbation "
         "sizes. It helps at large ε but costs clean accuracy and does not help at the smallest ε on 106 subjects."),
    ]
    for name, w, cap in figs:
        st.append(KeepTogether([fig(name, w), Paragraph(cap, CAP), Spacer(1, 5)]))

    lim = [
        Paragraph("Limitations and next steps", ParagraphStyle("h", parent=H2, spaceBefore=0)),
        Paragraph("<b>Limitations.</b> All recordings come from a single visit, so train and test runs share the same visit; "
                  "real logins happen days apart and error rates would likely be higher. Lab-grade EEG only, one random seed, "
                  "digital white-box attacks, and a simulated challenge-response.", SMALL),
        Spacer(1, 3),
        Paragraph("<b>Across days (BED, 21 subjects, 14-channel headset, 3 sessions a week apart).</b> Within one session the error is "
                  "1-8%, but on a different day it is 29-39% (50% is chance). Enrolling on two sessions lowered it in all 12 cases "
                  "(best about 19-23%); per-session alignment, eyes-open/closed and alpha-only features did not help. "
                  "<b>Next steps.</b> More subjects and sessions, adaptive template updates, a pairwise verifier, and attacks on BED.", SMALL),
    ]
    row = Table([[[fig("fig_replay_challenge.png", 7.3),
                   Paragraph("<b>Figure 4.</b> Challenge-response vs replay: replay success falls from 100% to about 10–40% "
                             "when the attacker has recorded only some of the 4 task types (simulation).", CAP)], lim]],
                colWidths=[7.6 * cm, W - 7.6 * cm])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (1, 0), (1, 0), 0), ("LEFTPADDING", (1, 0), (1, 0), 10)]))
    st.append(row)
    doc.build(st)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
