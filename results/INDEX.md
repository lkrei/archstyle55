# Индекс артефактов (`runs_res/aggregate/`)

Сюда попадает всё, что считалось «на CPU» уже после 9 supervised прогонов
на Kaggle и двух zero-shot baseline'ов. Цифры консистентны между `.csv`,
`.md` и графиками.

## 1. Главная таблица

* `summary_table.csv`, `summary_table.md` — 14 строк (9 supervised + 2
  ансамбля + 3 zero-shot), accuracy, macro F1, balanced accuracy,
  bootstrap 95% CI (1000 ресемплов), top-5 для zero-shot/ансамблей.

| Top строки (для удобства) | acc | F1 | top-5 |
|---|---|---|---|
| ensemble_top3_uniform                  | **0.7852** | **0.7799** | 0.9685 |
| ensemble_top3_weighted                 | 0.7833 | 0.7781 | 0.9665 |
| efficientnet_v2_s_seed42_abl54 (54 cls) | 0.7786 | 0.7738 | — |
| efficientnet_v2_s_seed42  | 0.7623 | 0.7582 | — |
| dinov2_vitb14_linear      | 0.7467 | 0.7374 | — |
| convnext_small            | 0.7419 | 0.7384 | — |
| siglip_b16_224 (zero-shot) | 0.4308 | 0.4044 | 0.7699 |

* `pairwise_mcnemar.csv` — попарный McNemar по предсказаниям (91 пара).
* `per_class_recall.csv` — recall × модели по 55 классам.

## 2. Графики

`figures/`:
* `bar_models_acc_f1.png` — горизонтальные бары acc + F1 + 95% CI.
* `confusion_<model>.png` для топ-3 (uniform ensemble / weighted /
  efficientnet_v2_s).
* `per_class_recall_top5.png` — heatmap 55×5.
* `worst_classes_top5.png` — bar 10 самых сложных классов.

## 2а. Графики датасета (глава 2 ВКР)

`figures/`:
* `dataset_class_sizes.png` — горизонтальный bar-chart 55 классов
  с медианой / min / max и подсветкой шумного класса
  «Late 20th century Moscow architecture» (см. главу 5.4).
* `dataset_split_stack.png` — stacked bar train/val/test по 55 классам;
  доказательство сбалансированности stratified split'а.
* `dataset_image_dims.png` — hexbin width × height (4 000 sample) +
  гистограмма aspect ratio; ландшафт ≈ 50%, портрет ≈ 22%, square ≈ 28%.

## 2б. Графики обучения (глава 3-4 ВКР)

`figures/`:
* `lr_schedule.png` — cosine + linear warmup (10%) на синтетике, head
  в 10× выше backbone — соответствует training настройкам.
* `training_curves_top5.png` — четырёхпанельный субплот для V2-S,
  EfficientNet-B3, ConvNeXt-S, DINOv2-linear, ViT-B/16:
  train loss, val loss, val acc, LR(head). Best epochs:
  V2-S 35/35, B3 35/35, ConvNeXt 30/30, DINOv2 13/20 (раннее плато),
  ViT 30/30. Прямой ответ комиссии «уточнить, сколько эпох было».

## 3. Калибровка (temperature scaling)

`calibration/`:
* `calibration_summary.csv` — `T`, ECE/MCE до и после, `n_eval`.
* `reliability_<model>.png` — pre/post bar-overlay для топ-5 supervised
  + обоих CLIP/SigLIP.
* `calibration_<model>.json` — численные значения.

Ключевой негативный результат: для CLIP/SigLIP `T ≈ 0.012` (logits ×~80),
pre-ECE 0.41 → post-ECE 0.04. Прямые softmax-вероятности CLIP в этом
домене недоверчивые и требуют recalibration.

## 4. Ансамбли (soft-voting)

`ensembles/`:
* `top3_uniform/`   — uniform 1/3 после temperature scaling.
* `top3_weighted/`  — simplex grid (step 0.1) на 50/50 stratified split.
* `ensemble_summary.csv` — компактная сравнительная таблица.

## 5. Эмбеддинги (DINOv2-base, frozen)

`embeddings/`:
* `embeddings.npz` — `(3138, 768)` фичи + лейблы (старый, сохранён для
  совместимости).
* `embeddings_train.npz`, `embeddings_val.npz`, `embeddings_test.npz` —
  features + labels + paths (используется hybrid-классификатором).
* `centroid_cosine.npy` / `centroid_cosine.png` — 55×55 cosine-карта
  центроидов классов.
* `embedding_projection.png` — UMAP 2D (с цифровыми метками классов).
* `embedding_class_legend.txt` — расшифровка цифр → имя класса.

## 5в. Stacking-гибрид (probs(top-3) ⊕ attrs)

`hybrid_stacking/`:
* `stacking_summary.csv/.md/.json` — 6 вариантов: single V2-S, uniform
  top-3 ensemble, attrs-only, stacker(probs), stacker(probs+attrs),
  stacker(V2-S+attrs).
* `predictions.npz` — y_true и предсказания каждого варианта на 50%
  eval-половине.
* `figures/hybrid_stacking_summary.png` — barchart по 3 метрикам.

Главный вывод: при 50/50 split test (1 569 train HistGBM) **uniform
top-3 = 0.7782** оптимален; stacker(probs+attrs) = 0.7718 —
не значимо хуже (McNemar p = 0.48), stacker(V2-S+attrs) = 0.7240
**значимо хуже** одиночной V2-S (p = 0.0005, переобучение на маленьком
train). Подтверждает: добавление tabular-сигнала вредит, если
secondary classifier не получает достаточно данных.

## 5а. Гибрид (DINOv2 ⊕ SegFormer-attrs)

`hybrid/`:
* `hybrid_summary.csv/.md/.json` — три варианта (`attr_only`, `emb_only`,
  `hybrid`), HistGradientBoosting, train+val (17 740) → test (3 138).
* `hybrid_summary.png` (в `figures/`) — компактный barchart acc / F1 /
  bal-acc по трём вариантам.
* `predictions.npz` — y_true + три предсказания, для McNemar.
* `test_confusion.npy`, `test_report_hybrid.json` — для гибридного варианта.

Цифры: attr_only 0.2814, emb_only 0.7301, **hybrid 0.7371**
(+0.70 п.п. acc к emb_only, McNemar p = 0.12 — прирост *не* значим
на $\alpha = 0.05$; зафиксировано в главе 5 ВКР как honest finding).

## 5б. Post-hoc ablation шумного класса

`posthoc_class_drop.csv/.md` — пересчёт всех моделей на test без класса
«Late 20th century Moscow architecture»: согласованный +1.3 п.п. acc и
+1.5 п.п. F1 у всех топ-моделей.

## 5б+. Полная ablation: V2-S переобучение на 54 классах

Папка `runs_res/_unpacked/run_efficientnet_v2_s_seed42_abl54/` (35 epoch,
seed 42, идентичные гиперпараметры). Полный обзор и таблицы —
`runs_res/aggregate/ablation_v2s/`:

* `ablation_summary.json` — три цифры на одном тесте:
  - V2-S 55-class on 55-class test:        **acc 0.7623**;
  - V2-S 55-class on test без drop class:  **acc 0.7709** (post-hoc);
  - V2-S 54-class on 54-class test:        **acc 0.7786**.

  Чистый прирост от переобучения (одинаковая база сравнения): **+0.78 п.п.**
  Это означает, что вынос шумной категории помогает не только за счёт
  снятия её собственного 0.5%-ного recall'а, но и **позволяет модели
  лучше различать соседние классы**.

* `ablation_per_class.md` — таблица 54 классов: recall full → recall abl,
  отсортированная по дельте. Топ-выигрыши: Stalinist (+0.119),
  Constructivist (+0.100), Neoclassical (+0.100), Chicago school (+0.098),
  Naryshkin Baroque (+0.085), Brutalist (+0.083) — это все категории,
  чьи здания исторически смешивались с постсоветской московской
  эклектикой. Регрессии (Prairie School −0.067, Renaissance −0.065,
  Deconstructivism −0.059) затрагивают и так сильные классы и в среднем
  компенсированы выигрышами.

* `ablation_per_class_delta.png` — диверг-бар на 54 категории.
* `figures/test_confusion_abl54.png` — матрица 54×54 для retrained V2-S.
* `figures/training_curves_abl54.png` — loss/acc по 35 эпохам.

В сводной таблице `summary_table.md` строка
`efficientnet_v2_s_seed42_abl54` (acc 0.7786, F1 0.7738, bal_acc 0.7752)
идёт сразу за обоими ансамблями; замечание: ансамбли остались на
55-class тесте — они напрямую не сопоставимы с abl54.

## 6. Галерея ошибок

`errors/<model>/`:
* `error_gallery.png` — 24 кадра, для которых модель уверена и не права.
* `top_error_pairs.json` — топ-20 пар «true → pred» с количествами.
* Для слайда 12 ВКР используется `errors/top3_uniform/error_gallery.png`.

## 6а. XAI showcase (Grad-CAM++ для CNN)

`figures/xai_showcase_efficientnet_v2_s.png` — 3 правильных + 3 ошибочных
кадра с overlay Grad-CAM++; прямая иллюстрация для слайда 12 ВКР
по комментарию комиссии «примеры неверной классификации». Запуск:
``python -m pipeline.reports.xai_showcase ...`` (см. блок «6а» в
`SLIDES_OUTLINE.md`).

## 6аа. ViT XAI showcase (Attention Rollout + Chefer relevance)

`figures/xai_showcase_vit_b16.png` — 3 правильных + 3 ошибочных кадра
ViT-B/16 с двумя картами для каждого: attention rollout (class-agnostic)
и Chefer relevance (class-specific). Class-specific версия заметно
резче и более интерпретируема — её и стоит использовать в защите.
Запуск: `python -m pipeline.reports.xai_showcase_vit ...`.

`figures/xai_showcase_dinov2_vitb14_linear.png` — то же для DINOv2
ViT-B/14 (linear probe). XAI собирается через
``transformers.Dinov2Model`` (тот же checkpoint что в torch.hub) +
linear head из обученного weight'а. Хорошо видно, что у self-
supervised DINOv2 attention распределено **существенно более
равномерно**, чем у supervised ViT-B/16 — типичная особенность
DINO/iBOT loss; class-specific сигнал лучше показывать через
Chefer-relevance, а не raw rollout. Запуск:
``python -m pipeline.reports.xai_showcase_dinov2 ...``

## 6б+. Per-class delta grid (4 модели одновременно)

`figures/perclass_delta_grid.png` + `perclass_delta_grid.json` — сводная
сетка 2×2 для V2-S, B3, ConvNeXt-S и uniform top-3 ensemble. Видно,
что Stalinist / Russian eclectic / Constructivist стабильно входят в
топ-5 «выигрывающих» классов через все четыре модели → эффект
обусловлен таксономией шумного класса, а не конкретным backbone.

## 6б. Per-class delta (full vs ablation)

`figures/perclass_delta_v2s.png` + `perclass_delta_v2s.json` — топ-12
классов с приростом recall у V2-S после удаления шумной категории
«Late 20th century Moscow architecture».
Показывает, что больше всего выигрывают Stalinist, Russian eclectic,
Constructivist (т.е. реально «свои» предсказания возвращаются).

## 6в. Calibration overview

`figures/calibration_overview.png` — горизонтальный bar-chart ECE
до/после temperature scaling для всех 8 моделей (5 supervised + 3
zero-shot варианта). Сразу виден отдельный кейс CLIP/SigLIP
(T = 0.012, ECE 0.41 → 0.04).

## 7. Compute cost

* `compute_cost.json` — params (M), GFLOPs, inference ms (CPU).
* `compute_cost_table.csv`/`.md` — модель + ресурсы + acc/F1, отсортирован
  по acc (DINOv2 заполнен из карточки модели).
* `compute_cost_bubble.png` — pareto-плот acc vs GFLOPs (диаметр = params).

## 8. Что лежит **вне** репозитория

Сырые per-run артефакты (`run_*/test_metrics.json`, `test_logits.npz`,
`test_confusion.npy`, `test_report.json`) и checkpoint'ы публикуются на
Hugging Face Hub: `kkkaredaw/archstyle55-backbones`. Локальные
`data_splits_local.json` (с абсолютными путями к датасету) в репо не
выкладываются по соображениям воспроизводимости — нужный split-манифест
лежит в `pipeline/results/splits/manifest.csv`.

## 9. Глава «Limitations»

`pipeline/reports/limitations.md` (в коде) — заполнена реальными числами,
включая баланс датасета 199/720, топ-пары ошибок, поведение SigLIP с
неверным паддингом и поведение Swin-V2-T с дефолтными гиперпараметрами.
