# Model Training Optimization Changelog

This document tracks all hyperparameter changes and their impact on model performance.

## Version History

### v5.0 - Data Smoothing Enhancement (Current)
**Target**: Improve MAPE by reducing noise through data smoothing (v1.0 baseline: 10.41%)

**New Feature**: Added technical data smoothing to reduce noise before training

**Configuration:**
| Parameter | Value |
|-----------|-------|
| `INPUT_SIZE` | 90 days |
| `MAX_STEPS` | 500 |
| `LEARNING_RATE` | 5e-4 |
| `VAL_CHECK_STEPS` | 50 |
| `APPLY_SMOOTHING` | True |
| `SMOOTHING_METHOD` | exponential |
| `SMOOTHING_ALPHA` | 0.3 |

**Changes:**
- Added exponential smoothing with alpha=0.3 to training data
- Smoothing reduces noise while preserving trend and seasonality
- Applied only to training data (test data remains unsmoothed for fair evaluation)

**Expected Impact:**
- Reduced noise in training data should improve model learning
- Better trend and seasonality pattern recognition
- Target: MAPE < 10%

---

### v1.0 - Exact Best Configuration (Baseline for v5.0)
**Target**: Return to v1.0 exact configuration (MAPE 10.41% was best)

**Decision**: All attempts to improve v1.0 (v2.0, v3.0, v4.0) have worsened performance. Reverting to exact v1.0 configuration.

**Configuration:**
| Parameter | Value |
|-----------|-------|
| `INPUT_SIZE` | 90 days |
| `MAX_STEPS` | 500 |
| `LEARNING_RATE` | 5e-4 |
| `VAL_CHECK_STEPS` | 50 |

**Rationale:**
- v1.0 achieved MAPE 10.41% - closest to 10% target
- v2.0-v4.0 experiments all worsened performance
- Return to proven working configuration
- Future optimizations should consider different approaches (ensemble, exogenous variables)

---

### v4.0 - Refined v1.0 Approach (Did Not Work)
**Target**: Reduce MAPE below 10% (v3.0 worsened significantly: 14.40%)

**Changes Summary:**
| Parameter | Previous (v3) | New (v4) | Change | vs v1.0 (best) |
|-----------|---------------|----------|--------|----------------|
| `INPUT_SIZE` | 90 days | 90 days | 0% | Same |
| `MAX_STEPS` | 700 | 550 | -21% | +10% from v1.0 |
| `LEARNING_RATE` | 4e-4 | 4.5e-4 | +12.5% | -10% from v1.0 |
| `VAL_CHECK_STEPS` | 50 | 50 | 0% | Same |

**Result:**
- MAPE worsened to 13.15% (still worse than v1.0's 10.41%)
- Even minimal changes from v1.0 did not improve performance
- Confirms v1.0 configuration is optimal for current dataset

---

### v3.0 - Balanced Optimization (Did Not Work)
**Target**: Reduce MAPE below 10% (v2.0 got worse: 10.89%)

**Changes Summary:**
| Parameter | Previous (v2) | New (v3) | Change |
|-----------|---------------|----------|--------|
| `INPUT_SIZE` | 120 days | 90 days | -25% |
| `MAX_STEPS` | 1000 | 700 | -30% |
| `LEARNING_RATE` | 3e-4 | 4e-4 | +33% |
| `VAL_CHECK_STEPS` | 100 | 50 | -50% |

**Result:**
- MAPE worsened significantly to 14.40% (up from 10.89%)
- Learning rate of 4e-4 was too low
- 700 steps may have caused issues

---

### v2.0 - Further Optimization (Did Not Improve)
**Target**: Reduce MAPE below 10% (from 10.41%)

**Changes Summary:**
| Parameter | Previous (v1) | New (v2) | Change |
|-----------|---------------|----------|--------|
| `INPUT_SIZE` | 90 days | 120 days | +33% |
| `MAX_STEPS` | 500 | 1000 | +100% |
| `LEARNING_RATE` | 5e-4 | 3e-4 | -40% |
| `VAL_CHECK_STEPS` | 50 | 100 | +100% |

**Rationale:**
- Increased `INPUT_SIZE` to 120 days (4 months) for better seasonal pattern capture
- Doubled `MAX_STEPS` to 1000 for extended training and better convergence
- Reduced `LEARNING_RATE` by 40% to minimize overestimation bias
- Doubled `VAL_CHECK_STEPS` for more frequent validation monitoring

**Result:**
- MAPE worsened to 10.89% (up from 10.41%)
- Too aggressive optimizations likely caused overfitting or under-training
- Input size of 120 days may be too long for the dataset size

---

### v1.0 - Initial Optimization
**Target**: Improve from baseline MAPE 13.23%

**Changes Summary:**
| Parameter | Previous (Baseline) | New (v1) | Change |
|-----------|---------------------|----------|--------|
| `INPUT_SIZE` | 60 days | 90 days | +50% |
| `MAX_STEPS` | 200 | 500 | +150% |
| `LEARNING_RATE` | 1e-3 | 5e-4 | -50% |
| `VAL_CHECK_STEPS` | N/A | 50 | New |

**Rationale:**
- Increased `INPUT_SIZE` to 90 days (3 months) for better seasonal context
- Increased `MAX_STEPS` 2.5x for better model convergence
- Reduced `LEARNING_RATE` by 50% for more stable learning
- Added validation checkpoints for monitoring

**Result:**
- MAPE improved to 10.41% (down from 13.23%)
- Still above 10% target, prompting v2.0 optimization

---

### v0.0 - Baseline Configuration
**Initial Configuration:**
- `INPUT_SIZE` = 60 days
- `MAX_STEPS` = 200
- `LEARNING_RATE` = 1e-3
- `VAL_CHECK_STEPS` = N/A

**Result:**
- MAPE = 13.23%
- Forecast overestimation observed
- Upward bias in predictions

---

## Performance Tracking

| Version | MAPE | MAE | RMSE | Status |
|---------|------|-----|------|--------|
| v0.0 (Baseline) | 13.23% | - | - | ⚠️ Above target |
| v1.0 | 10.41% | 17.79 | 22.06 | ✅ Baseline (best hyperparams) |
| v2.0 | 10.89% | 19.09 | 25.32 | ❌ Worsened |
| v3.0 | 14.40% | 24.47 | 29.67 | ❌ Much worse |
| v4.0 | 13.15% | 23.14 | 28.12 | ❌ Still worse than v1.0 |
| v5.0 | TBD | TBD | TBD | 🎯 Testing with smoothing |

**Target**: MAPE < 10%

---

## Notes

- All optimizations focus on reducing forecast overestimation bias
- Learning rate reductions help stabilize training and reduce overshooting
- Increased training steps allow better model convergence
- Longer input windows improve seasonal pattern recognition

## Lessons Learned

**Key Finding**: v1.0 configuration (INPUT_SIZE=90, MAX_STEPS=500, LR=5e-4) achieved the best MAPE of 10.41%. All subsequent hyperparameter tuning attempts (v2.0-v4.0) worsened performance.

**Observations**:
1. Hyperparameter tuning alone may have reached a local optimum at v1.0
2. The 0.41% gap to reach MAPE < 10% target may require different approaches:
   - **Ensemble methods**: Combine NHITS with other models (NBEATS, AutoARIMA)
   - **Exogenous variables**: Add day-of-week, holidays, month indicators
   - **Data augmentation**: Generate more training data
   - **Different loss functions**: Try alternatives to StudentT distribution

**Recommendation**: For future improvements, consider ensemble models or adding exogenous features rather than further hyperparameter tuning of NHITS alone.

