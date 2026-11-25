# BED results (21 subjects, run 'v1')

EER in %, lower is better, chance is 50%. `±` = std over seeds/splits (deterministic models have none).

Tuned on session-1 training data only: {"LogReg (band power)": {"C": 1.0}, "Riemannian (tangent space + LogReg)": {"C": 1.0}, "Riemannian template (cosine)": {}, "EEGNet-style CNN": {"n_filters": 32, "dropout": 0.3}}


## S1+S2 Headline: same session vs 1-week vs 2-week gap (mean EER %)

| group | LogReg (band power) | Riemannian (tangent space + LogReg) | Riemannian template (cosine) | EEGNet-style CNN |
|---|---|---|---|---|
| same session | 2.3 | 1.2 | 5.8 | 7.7 |
| 1-week gap | 34.5 | 35.6 | 34.0 | 29.3 |
| 2-week gap | 38.9 | 38.4 | 36.2 | 33.6 |


### S2 every enrol -> test pair (EER %)

| setting | LogReg (band power) | Riemannian (tangent space + LogReg) | Riemannian template (cosine) | EEGNet-style CNN |
|---|---|---|---|---|
| s1 -> s2 | 35.7 | 38.8 | 40.4 | 35.3 ± 1.3 |
| s1 -> s3 | 40.5 | 39.6 | 35.8 | 35.0 ± 1.4 |
| s2 -> s1 | 40.6 | 42.2 | 38.2 | 31.4 ± 1.6 |
| s2 -> s3 | 33.6 | 27.9 | 24.0 | 23.4 ± 1.6 |
| s3 -> s1 | 37.3 | 37.3 | 36.5 | 32.3 ± 1.4 |
| s3 -> s2 | 27.8 | 33.4 | 33.5 | 27.2 ± 0.4 |


## S3 Single vs multi-session enrolment, same test session (mean EER %)

single = average over the two possible one-session enrolments.

| test | model | single | multi | change |
|---|---|---|---|---|
| s1 | EEGNet-style CNN | 31.8 | 27.8 | -4.0 |
| s1 | LogReg (band power) | 39.0 | 34.3 | -4.7 |
| s1 | Riemannian (tangent space + LogReg) | 39.7 | 36.7 | -3.0 |
| s1 | Riemannian template (cosine) | 37.4 | 36.9 | -0.5 |
| s2 | EEGNet-style CNN | 31.3 | 19.2 | -12.1 |
| s2 | LogReg (band power) | 31.8 | 24.0 | -7.7 |
| s2 | Riemannian (tangent space + LogReg) | 36.1 | 29.2 | -6.9 |
| s2 | Riemannian template (cosine) | 37.0 | 33.3 | -3.7 |
| s3 | EEGNet-style CNN | 29.2 | 22.1 | -7.0 |
| s3 | LogReg (band power) | 37.1 | 32.4 | -4.7 |
| s3 | Riemannian (tangent space + LogReg) | 33.7 | 32.1 | -1.7 |
| s3 | Riemannian template (cosine) | 29.9 | 26.6 | -3.3 |


## S4 Enrolment size: s1 + n clips per user from s2, tested on s3 (2 s per clip)

| setting | LogReg (band power) | Riemannian (tangent space + LogReg) | Riemannian template (cosine) |
|---|---|---|---|
| s1 + 0 clips/user of s2 -> s3 | 40.5 | 39.6 | 35.8 |
| s1 + 5 clips/user of s2 -> s3 | 35.2 | 34.9 | 35.3 |
| s1 + 10 clips/user of s2 -> s3 | 34.6 | 34.0 | 35.0 |
| s1 + 20 clips/user of s2 -> s3 | 33.2 | 32.3 | 33.7 |
| s1 + 40 clips/user of s2 -> s3 | 32.7 | 31.3 | 31.7 |
| s1 + 80 clips/user of s2 -> s3 | 32.4 | 31.0 | 28.1 |


## S5 Session normalisation (none: no session normalisation; fz: z-score band-power features per session; ea: Euclidean alignment (whiten each session by its mean covariance) + feature z-score; riem: Riemannian recentering (whiten by the Riemannian mean covariance) + feature z-score)

| setting | test | LogReg (band power) | Riemannian (tangent space + LogReg) | Riemannian template (cosine) |
|---|---|---|---|---|
| none, single (s1) | s2 | 35.7 | 38.8 | 40.4 |
| none, single (s1) | s3 | 40.5 | 39.6 | 35.8 |
| none, multi (s1+s2) | s3 | 32.4 | 32.1 | 26.6 |
| fz, single (s1) | s2 | 34.2 | 38.8 | 40.4 |
| fz, single (s1) | s3 | 39.4 | 39.6 | 35.8 |
| fz, multi (s1+s2) | s3 | 33.6 | 32.1 | 26.6 |
| ea, single (s1) | s2 | 34.2 | 40.6 | 37.7 |
| ea, single (s1) | s3 | 39.4 | 42.8 | 41.3 |
| ea, multi (s1+s2) | s3 | 33.6 | 38.7 | 37.5 |
| riem, single (s1) | s2 | 34.2 | 36.8 | 37.8 |
| riem, single (s1) | s3 | 39.4 | 38.9 | 35.4 |
| riem, multi (s1+s2) | s3 | 33.6 | 30.9 | 27.1 |


## S6 Resting condition

| setting | test | LogReg (band power) | Riemannian (tangent space + LogReg) | Riemannian template (cosine) |
|---|---|---|---|---|
| both, single (s1) | s2 | 35.7 | 38.8 | 40.4 |
| both, single (s1) | s3 | 40.5 | 39.6 | 35.8 |
| both, multi (s1+s2) | s3 | 32.4 | 32.1 | 26.6 |
| eyes_closed, single (s1) | s2 | 35.7 | 38.8 | 42.9 |
| eyes_closed, single (s1) | s3 | 37.4 | 37.3 | 35.2 |
| eyes_closed, multi (s1+s2) | s3 | 33.2 | 31.8 | 27.4 |
| eyes_open, single (s1) | s2 | 34.8 | 38.7 | 39.5 |
| eyes_open, single (s1) | s3 | 40.2 | 39.9 | 34.9 |
| eyes_open, multi (s1+s2) | s3 | 31.9 | 32.1 | 26.6 |


## S7 Score fusion over k consecutive 2 s clips

k = clips averaged per login attempt

| setting | test | LogReg (band power) k=1 | LogReg (band power) k=3 | LogReg (band power) k=5 | LogReg (band power) k=10 | Riemannian (tangent space + LogReg) k=1 | Riemannian (tangent space + LogReg) k=3 | Riemannian (tangent space + LogReg) k=5 | Riemannian (tangent space + LogReg) k=10 | Riemannian template (cosine) k=1 | Riemannian template (cosine) k=3 | Riemannian template (cosine) k=5 | Riemannian template (cosine) k=10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single (s1) | s2 | 35.7 | 34.2 | 33.8 | 34.9 | 38.8 | 38.0 | 37.5 | 36.8 | 40.4 | 39.4 | 37.9 | 37.7 |
| single (s1) | s3 | 40.5 | 41.1 | 40.7 | 40.9 | 39.6 | 39.3 | 38.3 | 38.4 | 35.8 | 34.5 | 33.8 | 34.1 |
| multi (s1+s2) | s3 | 32.4 | 30.3 | 29.0 | 29.3 | 32.1 | 30.2 | 29.2 | 29.8 | 26.6 | 25.0 | 24.3 | 23.1 |


## S8 Band-power feature set (LogReg)

| setting | test | LogReg (band power) |
|---|---|---|
| all, single (s1) | s2 | 35.7 |
| all, single (s1) | s3 | 40.5 |
| all, multi (s1+s2) | s3 | 32.4 |
| alpha, single (s1) | s2 | 41.8 |
| alpha, single (s1) | s3 | 43.2 |
| alpha, multi (s1+s2) | s3 | 39.1 |
| theta_alpha, single (s1) | s2 | 41.3 |
| theta_alpha, single (s1) | s3 | 42.2 |
| theta_alpha, multi (s1+s2) | s3 | 39.8 |
| alpha_beta, single (s1) | s2 | 38.0 |
| alpha_beta, single (s1) | s3 | 41.5 |
| alpha_beta, multi (s1+s2) | s3 | 35.2 |


## S9 Open-set: impostors are people never enrolled

| setting | Riemannian template (cosine) |
|---|---|
| single (s1) -> s2 | 41.1 ± 2.4 |
| single (s1) -> s3 | 35.1 ± 3.1 |
| multi (s1+s2) -> s3 | 26.9 ± 1.8 |


## S10 Improvement ladder, tested on session 3 (cumulative, steps fixed in advance)

| setting | LogReg (band power) | Riemannian (tangent space + LogReg) | Riemannian template (cosine) | EEGNet-style CNN |
|---|---|---|---|---|
| 0 baseline: enrol s1 | 40.5 | 39.6 | 35.8 | 33.9 ± 0.9 |
| 1 + per-session alignment (EA) | 39.4 | 42.8 | 41.3 | 34.7 ± 1.0 |
| 2 + enrol on s1+s2 | 33.6 | 38.7 | 37.5 | 33.4 ± 0.7 |
| 3 + fuse 5 clips per attempt | 31.0 | 37.1 | 36.7 | 32.1 ± 0.9 |


## Figures
- fig1_time_gap.png
- fig2_multisession.png
- fig3_enrol_size.png
- fig4_fusion.png
- fig5_ladder.png
