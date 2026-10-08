# ♟️ Chess Outcome Prediction

**Machine Learning · Data Science · Orange3**

> Multi-classifier + regression pipeline on 20,058 Lichess games — predicting game outcome and estimating player ELO from in-game features.

---

## 🎯 Problem Statement

Given a chess game's metadata (player ratings, opening, time control, termination type), can we:
1. **Classify** the outcome — White win / Black win / Draw?
2. **Regress** a player's ELO rating from observable match statistics?

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Lichess via Kaggle](https://www.kaggle.com/datasets/datasnaek/chess) |
| Total games | **20,058** |
| After filtering | ~18,500 usable rows |
| Target classes | White win · Black win · Draw |

---

## 🔧 Feature Engineering

| Feature | Type | Notes |
|---|---|---|
| `white_rating` | Numeric | ELO of White player |
| `black_rating` | Numeric | ELO of Black player |
| `rating_diff` | Engineered | WhiteElo − BlackElo |
| `opening_ply` | Numeric | Opening length (depth of theory) |
| `opening_category` | Categorical | 130+ openings → 5 grouped categories |
| `eco` | Categorical | Opening family: A / B / C / D / E |
| `time_control` | Categorical | bullet / blitz / rapid / classical |
| `turns` | Numeric | Total half-moves in game |

**Preprocessing note:** Rare openings (< threshold) were grouped into an "other" category to reduce cardinality and improve generalization.

---

## 🤖 Models & Results

### Classification — Predict Outcome (White / Black / Draw)

| Model | AUC | CA | F1 | Prec | Recall | MCC |
|---|---|---|---|---|---|---|
| **Neural Network** | **0.680** | **0.640** | 0.638 | 0.636 | 0.640 | 0.289 |
| SVM | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| Naive Bayes | — | — | — | — | — | — |

*Neural Network architecture: 200→100→50→20 neurons · ReLU · Adam · α=0.01 · max 500 iterations*

**Confusion Matrix insight:**

| Actual \ Predicted | Draw | Mate | Outoftime | Resign |
|---|---|---|---|---|
| Draw | 308 | — | — | — |
| Mate | — | 1890 | — | — |
| Resign | — | — | — | 3451 |

→ **53.9%** of games end by **resignation**, **29.9%** by **checkmate**.

---

### Regression — Predict Player ELO Rating

| Model | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| **Linear Regression** | **0.961** | **51.1** | 42.5 | 0.027 |
| Neural Network | 0.944 | 60.8 | 45.3 | 0.029 |

**Key finding:** ELO rating is highly predictable from observable game statistics (R²=0.961). The model explains 96.1% of variance with ~±51 ELO point error — strong enough to infer player skill level without a rating system.

---

## 🔍 Statistical Insights

- **ELO distribution:** Mean ≈ 1600, σ ≈ 291 — normally distributed, matching real Lichess population
- **Strongest correlation:** `black_rating` ↔ Black win direction = **+0.63**
- **White advantage confirmed:** White's first-move advantage is statistically visible in the data
- **Sicilian defense** dominates as the most common opening, consistent with competitive theory

---

## ⚙️ Pipeline Architecture

```
CSV Input
  └─ Select Columns (remove metadata/IDs)
       └─ Edit Domain (group rare openings → "other", 5 ECO categories)
            └─ Formula (compute rating_diff = white_rating − black_rating)
                 └─ Merge Data
                      └─ Select Rows (filter nulls)
                           ├─ Test & Score ──→ Confusion Matrix
                           │                └─ Venn Diagram
                           │                └─ Predictions Table
                           └─ [Regression branch]
                                └─ Linear Regression
                                └─ Neural Network Regression
                                     └─ Cross Validation
```

---

## 📁 Files

| File | Description |
|---|---|
| `chess_outcome_prediction.ows` | Development Orange3 workflow |
| `chess_outcome_prediction_final.ows` | Final submission workflow |
| `מבוא_למדעי_הנתונים_שחמט.pdf` | Full project report (Hebrew) |

---

## 🚀 How to Run

1. Install [Orange3](https://orangedatamining.com/)
2. Download the dataset: [Kaggle — Chess Game Dataset](https://www.kaggle.com/datasets/datasnaek/chess)
3. Open `chess_outcome_prediction_final.ows`
4. Load the CSV in the **File** widget
5. Run → inspect **Confusion Matrix**, **Distributions**, and **Scatter Plot** widgets

---

## 💡 Limitations & Future Work

- Classification accuracy (~64%) reflects the inherent unpredictability of chess — even simple patterns like ratings explain only so much at the game level
- **Missing features:** move sequences, centipawn loss, time-per-move, endgame type
- **Next step:** Apply sequence models (LSTM / Transformer) on move-by-move PGN data for deeper pattern extraction

---

*Data Science project · 2024*
