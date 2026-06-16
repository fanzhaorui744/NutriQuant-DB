import numpy as np

def calculate_metrics(predictions, targets):
    """
    Calculates MAE, RMSE, MAPE for weight/calorie estimation.
    predictions: list or array of predicted values
    targets: list or array of ground truth values
    """
    preds = np.array(predictions, dtype=np.float64)
    targs = np.array(targets, dtype=np.float64)

    valid = np.isfinite(preds) & np.isfinite(targs)
    preds = preds[valid]
    targs = targs[valid]

    if preds.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE": float("nan")}

    mae = float(np.mean(np.abs(preds - targs)))
    rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))

    denom_mask = targs > 1e-6
    if np.any(denom_mask):
        mape = float(np.mean(np.abs((preds[denom_mask] - targs[denom_mask]) / targs[denom_mask])) * 100)
    else:
        mape = float("nan")
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

def consistency_analysis(measurements):
    """
    Calculates Coefficient of Variation (CV) for repeated measurements of the same object.
    measurements: list of lists, where each inner list contains repeated measures for one object.
    """
    cvs = []
    for measures in measurements:
        if len(measures) < 2:
            continue
        mean = np.mean(measures)
        std = np.std(measures)
        if mean == 0:
            cvs.append(0)
        else:
            cvs.append((std / mean) * 100)
            
    avg_cv = np.mean(cvs) if cvs else 0
    return avg_cv

if __name__ == "__main__":
    # Example usage
    # Simulated data
    true_weights = [200, 350, 150, 400]
    pred_weights = [210, 340, 160, 380]
    
    print("Error Analysis:")
    metrics = calculate_metrics(pred_weights, true_weights)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    # Consistency (e.g. same food photographed 3 times)
    repeated_measures = [
        [210, 215, 205], # Object 1
        [340, 335, 345], # Object 2
    ]
    cv = consistency_analysis(repeated_measures)
    print(f"\nConsistency Analysis (CV): {cv:.2f}%")
