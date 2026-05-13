| вариант | acc | macro F1 | bal. acc | n_features | n_train |
| --- | --- | --- | --- | --- | --- |
| single_v2s_argmax | 0.7527 | 0.7480 | 0.7498 | 55 | 0 |
| ensemble_top3_uniform_argmax | 0.7782 | 0.7721 | 0.7724 | 165 | 0 |
| attrs_only | 0.1268 | 0.1107 | 0.1170 | 26 | 1569 |
| stacker_probs_top3 | 0.7661 | 0.7587 | 0.7617 | 165 | 1569 |
| stacker_probs_top3_attrs | 0.7718 | 0.7643 | 0.7673 | 191 | 1569 |
| stacker_v2s_attrs | 0.7240 | 0.7121 | 0.7167 | 81 | 1569 |

McNemar (на той же 50% eval-выборке):
* `v2s_vs_ensemble`: chi2=14.349, p=0.0002, n01=73, n10=33
* `ensemble_vs_stacker_pt`: chi2=1.964, p=0.1611, n01=73, n10=92
* `ensemble_vs_stacker_pt_atr`: chi2=0.506, p=0.4768, n01=75, n10=85
* `stacker_pt_vs_pt_atr`: chi2=1.488, p=0.2225, n01=26, n10=17
* `v2s_vs_stacker_v2s_atr`: chi2=12.176, p=0.0005, n01=57, n10=102
