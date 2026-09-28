# ♟️ Chess Outcome Prediction

**Machine Learning · Orange3 · Data Science**

Predict chess game outcome (White win / Black win / Draw) using a multi-classifier comparison pipeline.

## Models Compared

| Model | Tool |
|---|---|
| SVM | Orange3 |
| Random Forest | Orange3 |
| Neural Network | Orange3 |
| Naive Bayes | Orange3 |

## Features

- **ELO ratings** — WhiteElo, BlackElo
- **Engineered feature** — RatingDiff (White − Black)
- **Opening type** — 130+ openings, rare ones grouped as "other"
- **ECO code** — opening family (A–E)
- **Time control** — bullet / blitz / rapid / classical
- **Termination** — resignation, checkmate, timeout

## Pipeline

```
CSV → Select Columns → Edit Domain (opening categorization)
    → Formula (rating diff) → Merge → Filter
    → Test & Score → Confusion Matrix · Venn Diagram · Predictions
```

## Files

| File | Description |
|---|---|
| `chess_outcome_prediction.ows` | Main Orange3 workflow |
| `chess_outcome_prediction_final.ows` | Final submission version |

## How to Run

1. Install [Orange3](https://orangedatamining.com/)
2. Open `chess_outcome_prediction_final.ows`
3. Load your chess dataset (CSV with Lichess/Chess.com export format)
4. Run → view results in Confusion Matrix and Venn Diagram widgets

---

*Data Science course project · Ariel University*
