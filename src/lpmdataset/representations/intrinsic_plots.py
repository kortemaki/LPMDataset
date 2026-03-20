import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.lpmdataset.representations.heatmap import *
import matplotlib 
matplotlib.use("Agg")

# =========================================================
# CONFIG
# =========================================================

SCREEN_W = 1200
SCREEN_H = 900


# =========================================================
# LOAD FILES
# =========================================================

def load_result_files(results_root):

    files = []

    for root, _, fs in os.walk(results_root):
        for f in fs:
            if f.endswith(".csv"):
                files.append(os.path.join(root, f))

    return list(set(files))


# =========================================================
# TRAJECTORY METRICS
# =========================================================

def compute_metrics(df):

    required = ["pred_x","pred_y","gold_x","gold_y"]
    if not all(c in df.columns for c in required):
        return None

    pred = df[["pred_x","pred_y"]].values.astype(np.float32)
    gold = df[["gold_x","gold_y"]].values.astype(np.float32)

    N = min(len(pred), len(gold))
    if N < 2:
        return None

    pred = pred[:N]
    gold = gold[:N]

    pred[:,0] /= SCREEN_W
    pred[:,1] /= SCREEN_H
    gold[:,0] /= SCREEN_W
    gold[:,1] /= SCREEN_H

    dist = np.linalg.norm(pred - gold, axis=1)

    ad = np.mean(dist)
    rmse = np.sqrt(np.mean(dist**2))

    bins = 20
    ph,_,_ = np.histogram2d(pred[:,0], pred[:,1], bins=bins, range=[[0,1],[0,1]])
    gh,_,_ = np.histogram2d(gold[:,0], gold[:,1], bins=bins, range=[[0,1],[0,1]])

    ph /= (ph.sum()+1e-8)
    gh /= (gh.sum()+1e-8)

    iou = np.minimum(ph,gh).sum() / (np.maximum(ph,gh).sum()+1e-8)

    return ad, rmse, iou


# =========================================================
# MOVEMENT METRICS
# =========================================================

def compute_movement_metrics(files, threshold=1.0):

    all_gold, all_pred = [], []

    for f in files:

        df = pd.read_csv(f)

        required = ["pred_x","pred_y","gold_x","gold_y"]
        if not all(c in df.columns for c in required):
            print(f"Skipping (bad format): {f}")
            continue

        if len(df) < 2:
            continue

        # force float (important for OCR)
        df = df.astype({
            "pred_x": float, "pred_y": float,
            "gold_x": float, "gold_y": float
        })

        gold_dx = np.diff(df["gold_x"], prepend=df["gold_x"][0])
        gold_dy = np.diff(df["gold_y"], prepend=df["gold_y"][0])

        pred_dx = np.diff(df["pred_x"], prepend=df["pred_x"][0])
        pred_dy = np.diff(df["pred_y"], prepend=df["pred_y"][0])

        gold_move = (np.sqrt(gold_dx**2 + gold_dy**2) > threshold).astype(int)
        pred_move = (np.sqrt(pred_dx**2 + pred_dy**2) > threshold).astype(int)

        all_gold.append(gold_move)
        all_pred.append(pred_move)

    if len(all_gold) == 0:
        print("No valid movement data.")
        return {"TP":0,"TN":0,"FP":0,"FN":0,
                "precision":0,"recall":0,"f1":0,"accuracy":0}

    all_gold = np.concatenate(all_gold)
    all_pred = np.concatenate(all_pred)

    TP = np.sum((all_pred==1)&(all_gold==1))
    TN = np.sum((all_pred==0)&(all_gold==0))
    FP = np.sum((all_pred==1)&(all_gold==0))
    FN = np.sum((all_pred==0)&(all_gold==1))

    precision = TP/(TP+FP+1e-8)
    recall    = TP/(TP+FN+1e-8)
    f1        = 2*precision*recall/(precision+recall+1e-8)
    accuracy  = (TP+TN)/(TP+TN+FP+FN+1e-8)

    print("\n--- Movement Metrics ---")
    print(f"TP:{TP} TN:{TN} FP:{FP} FN:{FN}")
    print(f"P:{precision:.3f} R:{recall:.3f} F1:{f1:.3f} Acc:{accuracy:.3f}")

    return {"TP":TP,"TN":TN,"FP":FP,"FN":FN,
            "precision":precision,"recall":recall,
            "f1":f1,"accuracy":accuracy}


# =========================================================
# PLOTS
# =========================================================

def plot_confusion_matrix(metrics, model_name):

    cm = np.array([[metrics["TN"],metrics["FP"]],
                   [metrics["FN"],metrics["TP"]]])

    plt.figure(figsize=(6,5))
    im = plt.imshow(cm)

    plt.xticks([0,1],["No Move","Move"])
    plt.yticks([0,1],["No Move","Move"])

    for i in range(2):
        for j in range(2):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center")

    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar(im)
    plt.savefig("plot.png")
    plt.close()


def plot_metric_bars(metrics, model_name):

    names = ["Precision","Recall","F1","Accuracy"]
    vals = [metrics[k] for k in ["precision","recall","f1","accuracy"]]

    plt.figure(figsize=(6,4))
    bars = plt.bar(names, vals)

    for b,v in zip(bars,vals):
        plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha='center')

    plt.ylim(0,1.05)
    plt.title(f"{model_name} — Movement Metrics")
    plt.ylabel("Score")
    plt.savefig("plot.png")
    plt.close()



def plot_combined_heatmap(root, model_name, num_slides=5):

    files = load_result_files(root)
    if len(files)==0: return

    files = np.random.choice(files, min(num_slides,len(files)), replace=False)

    gx,gy,px,py=[],[],[],[]

    for f in files:
        df = pd.read_csv(f)
        gx.extend(df["gold_x"]/SCREEN_W)
        gy.extend(df["gold_y"]/SCREEN_H)
        px.extend(df["pred_x"]/SCREEN_W)
        py.extend(df["pred_y"]/SCREEN_H)

    fig,ax = plt.subplots(1,2,figsize=(10,5))

    h1=ax[0].hist2d(gx,gy,bins=40)
    ax[0].set_title(f"{model_name} — Gold")
    plt.colorbar(h1[3],ax=ax[0])

    h2=ax[1].hist2d(px,py,bins=40)
    ax[1].set_title(f"{model_name} — Predicted")
    plt.colorbar(h2[3],ax=ax[1])

    plt.suptitle(f"{model_name} — Spatial Distribution")
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

def spatial_metric_and_visualization(root, model_name, num_slides=5, bins=32):

    files = load_result_files(root)
    if len(files) == 0:
        print("No files found")
        return []

    files = np.random.choice(files, min(num_slides, len(files)), replace=False)

    scores = []

    rows = len(files)
    fig, axes = plt.subplots(rows, 2, figsize=(8, 4*rows))

    if rows == 1:
        axes = np.array([axes])

    for i, f in enumerate(files):

        df = pd.read_csv(f)

        required = ["pred_x","pred_y","gold_x","gold_y"]
        if not all(c in df.columns for c in required):
            continue

        # -------- GOLD --------
        df_gold = pd.DataFrame({
            "x": df["gold_x"],
            "y": df["gold_y"],
            "timestamp": np.arange(len(df))
        })

        hm_gold = HeatMap(df_gold)
        hm_gold.upsample()
        hist_g, _, _ = hm_gold.low_res(bins=bins)

        # -------- PRED --------
        df_pred = pd.DataFrame({
            "x": df["pred_x"],
            "y": df["pred_y"],
            "timestamp": np.arange(len(df))
        })

        hm_pred = HeatMap(df_pred)
        hm_pred.upsample()
        hist_p, _, _ = hm_pred.low_res(bins=bins)

        # -------- METRIC (Histogram Intersection) --------
        hist_g = hist_g / (hist_g.sum() + 1e-8)
        hist_p = hist_p / (hist_p.sum() + 1e-8)

        score = np.minimum(hist_g, hist_p).sum()
        scores.append(score)

        # -------- PLOT --------
        ax_g = axes[i, 0]
        ax_p = axes[i, 1]

        ax_g.imshow(hist_g.T, origin='lower', aspect='auto')
        ax_g.set_title(f"Slide {i+1} — Gold")
        ax_g.axis('off')

        ax_p.imshow(hist_p.T, origin='lower', aspect='auto')
        ax_p.set_title(f"Slide {i+1} — Pred\nScore: {score:.3f}")
        ax_p.axis('off')

    plt.suptitle(f"{model_name} — Spatial Metric (Intrinsic) + Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

    print(f"\nSpatial Metric (Histogram Intersection) — Mean: {np.mean(scores):.4f}")

    return scores
# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(root, name):

    print(f"\n===== {name} =====")

    files = load_result_files(root)
    if len(files)==0:
        print("No files found")
        return

    ads,rmses,ious=[],[],[]

    for f in files:
        df = pd.read_csv(f)
        res = compute_metrics(df)
        if res is None: continue
        ad,rmse,iou = res
        ads.append(ad); rmses.append(rmse); ious.append(iou)

    print(f"AD:{np.mean(ads):.4f} RMSE:{np.mean(rmses):.4f} IoU:{np.mean(ious):.4f}")

    metrics = compute_movement_metrics(files)

    plot_confusion_matrix(metrics, name)
    plot_metric_bars(metrics, name)
    plot_combined_heatmap(root, name)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    BASE="results"

    MODELS={
        "Mouse Only": os.path.join(BASE,"mouse_only"),
        "OCR Only": os.path.join(BASE,"ocr_only"),
        "OCR + ASR": os.path.join(BASE,"ocr_asr")
    }

    for name,root in MODELS.items():

        if not os.path.exists(root):
            print(f"Skipping {name}")
            continue

        evaluate_model(root,name)
        spatial_metric_and_visualization(root,name)