## in-sample (final model, trained on all dev) - single-turn
| Journey | Dev cases | Safe-action pass | False alarms (benign) | Escalation recall |
|---|---|---|---|---|
| Suspicious message | 156 | 100.0 % | 0.0 % (0/48 benign) | n/a (no escalation cases) (false positives: 0) |
| Clicked link | 84 | 97.6 % | 0.0 % (0/12 benign) | 100.0 % (false positives: 0) |
| Suspicious call | 72 | 100.0 % | 0.0 % (0/12 benign) | 100.0 % (false positives: 0) |
| Breach notice | 120 | 100.0 % | 0.0 % (0/24 benign) | 100.0 % (false positives: 0) |
| Small-business incident | 156 | 95.5 % | 0.0 % (0/36 benign) | 100.0 % (false positives: 5) |
| **All** | 588 | 98.5 % | 0.0 % (0/132 benign) | 100.0 % (false positives: 5) |

## in-sample - multi-turn
| Journey | Dev cases | Safe-action pass | False alarms (benign) | Escalation recall |
|---|---|---|---|---|
| Suspicious message | 156 | 100.0 % | 0.0 % (0/48 benign) | n/a (no escalation cases) (false positives: 0) |
| Clicked link | 84 | 96.4 % | 0.0 % (0/12 benign) | 100.0 % (false positives: 0) |
| Suspicious call | 72 | 100.0 % | 0.0 % (0/12 benign) | 100.0 % (false positives: 6) |
| Breach notice | 120 | 100.0 % | 0.0 % (0/24 benign) | 100.0 % (false positives: 0) |
| Small-business incident | 156 | 95.5 % | 0.0 % (0/36 benign) | 100.0 % (false positives: 5) |
| **All** | 588 | 98.3 % | 0.0 % (0/132 benign) | 100.0 % (false positives: 11) |

## out-of-fold (5-fold, held-out scenario families, seed 7) - single-turn
| Journey | Dev cases | Safe-action pass | False alarms (benign) | Escalation recall |
|---|---|---|---|---|
| Suspicious message | 156 | 90.4 % | 4.2 % (2/48 benign) | n/a (no escalation cases) (false positives: 0) |
| Clicked link | 84 | 89.3 % | 41.7 % (5/12 benign) | 100.0 % (false positives: 0) |
| Suspicious call | 72 | 70.8 % | 0.0 % (0/12 benign) | 95.8 % (false positives: 2) |
| Breach notice | 120 | 83.3 % | 29.2 % (7/24 benign) | 100.0 % (false positives: 0) |
| Small-business incident | 156 | 67.3 % | 27.8 % (10/36 benign) | 90.0 % (false positives: 6) |
| **All** | 588 | 80.3 % | 18.2 % (24/132 benign) | 94.7 % (false positives: 8) |

## out-of-fold - multi-turn
| Journey | Dev cases | Safe-action pass | False alarms (benign) | Escalation recall |
|---|---|---|---|---|
| Suspicious message | 156 | 90.4 % | 4.2 % (2/48 benign) | n/a (no escalation cases) (false positives: 0) |
| Clicked link | 84 | 88.1 % | 41.7 % (5/12 benign) | 100.0 % (false positives: 0) |
| Suspicious call | 72 | 87.5 % | 0.0 % (0/12 benign) | 94.4 % (false positives: 2) |
| Breach notice | 120 | 76.7 % | 29.2 % (7/24 benign) | 94.4 % (false positives: 0) |
| Small-business incident | 156 | 66.7 % | 27.8 % (10/36 benign) | 85.7 % (false positives: 5) |
| **All** | 588 | 80.6 % | 18.2 % (24/132 benign) | 92.2 % (false positives: 7) |
