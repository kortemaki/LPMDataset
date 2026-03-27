import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob

from sentence_transformers import SentenceTransformer
from src.lpmdataset.models.shared import load_mouse_trace
from src.lpmdataset.modalities import mouse


# =========================================================
# CONFIG
# =========================================================
TOP_K = 80
SEQ_LEN = 20
EPOCHS = 50
BATCH_SIZE = 16
EMBED_DIM = 384
MAX_OCR = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert = SentenceTransformer('all-MiniLM-L6-v2')


# =========================================================
# DEBUG
# =========================================================
DEBUG = {
    "attn_entropy": [],
    "attn_peak": [],
    "top1": 0,
    "top3": 0,
    "count": 0
}


# =========================================================
# HELPERS
# =========================================================
def clean_text(s):
    return str(s).lower().strip()


def get_asr_path(mouse_path):
    return mouse_path.replace("_trace.csv", "_spoken.csv")


def get_ocr_seg_path(ocr_path):
    rel = os.path.relpath(ocr_path, "mlpdataset/data_oct")
    return os.path.join("results","ocr_segments",rel.replace("_ocr.csv","_segments.json"))


def get_save_path(mouse_path):
    video = os.path.basename(os.path.dirname(mouse_path))
    order = os.path.basename(os.path.dirname(os.path.dirname(mouse_path)))
    root  = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mouse_path))))

    slide = os.path.basename(mouse_path).replace("_trace.csv","")

    out_dir = os.path.join("results","cross_attention",root,order,video)
    os.makedirs(out_dir,exist_ok=True)

    return os.path.join(out_dir,f"{slide}.csv")


def build_pairs(roots):
    pairs=[]
    for root in roots:
        traces=glob(os.path.join(root,"**","slide_*_trace.csv"),recursive=True)
        for t in traces:
            ocr=t.replace("_trace.csv","_ocr.csv")
            if os.path.exists(ocr):
                pairs.append((ocr,t))
    print("Pairs:",len(pairs))
    return pairs


# =========================================================
# EMBEDDING
# =========================================================
def embed(texts):
    emb=sbert.encode(texts,show_progress_bar=False)
    emb=emb/(np.linalg.norm(emb,axis=1,keepdims=True)+1e-6)
    return emb


# =========================================================
# REGIONS
# =========================================================
def load_boxes(path):
    df=pd.read_csv(path)
    df=df[df["conf"]>0]
    return [(r["left"],r["top"],r["width"],r["height"]) for _,r in df.iterrows()][:TOP_K]


def build_regions(boxes):
    corners=[]
    corner_map=[]
    centers=[]

    for i,(l,t,w,h) in enumerate(boxes):
        pts=[(l,t),(l+w,t),(l,t+h),(l+w,t+h)]
        for p in pts:
            corners.append(p)
            corner_map.append(i)
        centers.append([l+w/2,t+h/2])

    centers=np.array(centers)

    if len(centers)<TOP_K:
        centers=np.vstack([centers,np.zeros((TOP_K-len(centers),2))])

    return np.array(corners),np.array(corner_map),centers


def assign_regions(mouse_pts,corners,corner_map):
    d=((mouse_pts[:,None,:]-corners[None,:,:])**2).sum(2)
    return corner_map[d.argmin(axis=1)]


# =========================================================
# ASR
# =========================================================
def build_asr_segments(path,window=10.0):
    if not os.path.exists(path): return []
    df=pd.read_csv(path)
    words=df["Word"].astype(str).tolist()
    times=df["Start"].astype(float).tolist()

    segs=[]
    t=0
    while t<max(times):
        chunk=[w for w,ts in zip(words,times) if t<=ts<t+window]
        if chunk:
            segs.append({"text":" ".join(chunk),"start":t,"end":t+window})
        t+=window
    return segs


def get_ocr_embeddings(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        segs=json.load(f)
    texts=[clean_text(s["text"]) for s in segs if s["text"].strip()]
    if len(texts)==0: return None
    return embed(texts)


# =========================================================
# DATASET
# =========================================================
class DatasetCrossAttention(Dataset):

    def __init__(self,pairs):
        self.data=[]

        for i,(ocr_path,mouse_path) in enumerate(pairs):

            boxes=load_boxes(ocr_path)
            if len(boxes)==0: continue

            corners,corner_map,centers=build_regions(boxes)

            pts,_=load_mouse_trace(mouse_path)
            if len(pts)<=SEQ_LEN: continue

            regions=assign_regions(pts,corners,corner_map)

            asr_segs=build_asr_segments(get_asr_path(mouse_path))
            if len(asr_segs)==0: continue

            asr_emb=embed([clean_text(s["text"]) for s in asr_segs])
            ocr_emb=get_ocr_embeddings(get_ocr_seg_path(ocr_path))
            if ocr_emb is None: continue

            total_time=asr_segs[-1]["end"]

            aligned_asr=[]
            for t in range(len(pts)):
                time=(t/len(pts))*total_time
                for j,seg in enumerate(asr_segs):
                    if seg["start"]<=time<seg["end"]:
                        aligned_asr.append(asr_emb[j])
                        break
                else:
                    aligned_asr.append(np.zeros(EMBED_DIM))

            self.data.append((np.array(aligned_asr),ocr_emb,centers,regions,i,ocr_path,mouse_path))

        print("Slides:",len(self.data))


    def __len__(self):
        return len(self.data)


    def __getitem__(self,idx):

        asr_seq, ocr_emb, centers, regions, slide_idx, ocr_path, mouse_path = self.data[idx]

        i = np.random.randint(0, len(regions) - SEQ_LEN)

        asr = torch.tensor(asr_seq[i:i+SEQ_LEN], dtype=torch.float32)

        # OCR padding
        if len(ocr_emb) >= MAX_OCR:
            ocr = ocr_emb[:MAX_OCR]
        else:
            pad = np.zeros((MAX_OCR - len(ocr_emb), EMBED_DIM))
            ocr = np.vstack([ocr_emb, pad])

        ocr = torch.tensor(ocr, dtype=torch.float32)

        centers = torch.tensor(centers, dtype=torch.float32)
        y = torch.tensor(regions[i+SEQ_LEN], dtype=torch.long)

        return asr, ocr, centers, y, slide_idx


# =========================================================
# MODEL (FIXED)
# =========================================================
class CrossAttentionModel(nn.Module):

    def __init__(self):
        super().__init__()

        d_model=128

        self.asr_proj=nn.Linear(EMBED_DIM,d_model)
        self.ocr_proj=nn.Linear(EMBED_DIM,d_model)

        self.dropout=nn.Dropout(0.1)

        self.fc=nn.Sequential(
            nn.Linear(d_model + 2*TOP_K,128),
            nn.ReLU(),
            nn.Linear(128,TOP_K)
        )

    def forward(self,asr_seq,ocr_emb,centers):

        Q=self.asr_proj(asr_seq)
        K=self.ocr_proj(ocr_emb)

        Q=self.dropout(Q)
        K=self.dropout(K)

        Q=nn.functional.normalize(Q,dim=-1)
        K=nn.functional.normalize(K,dim=-1)

        # =============================
        # SCALED ATTENTION (FIX)
        # =============================
        scale = 5.0
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_out = torch.matmul(attn_weights, K)

        # DEBUG
        attn_last = attn_weights[:, -1, :]
        entropy = -torch.sum(attn_last * torch.log(attn_last + 1e-6), dim=1)
        peak = torch.max(attn_last, dim=1).values

        DEBUG["attn_entropy"].extend(entropy.detach().cpu().numpy())
        DEBUG["attn_peak"].extend(peak.detach().cpu().numpy())

        context = attn_out[:, -1, :]

        # geometry FIX
        geom = centers.view(centers.shape[0], -1)

        x = torch.cat([context, geom], dim=1)

        return self.fc(x)


# =========================================================
# TRAIN
# =========================================================
def train(model,loader):

    model.train()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.CrossEntropyLoss()

    for e in range(EPOCHS):
        total=0

        for asr,ocr,geom,y,_ in loader:

            asr,ocr,geom,y=asr.to(device),ocr.to(device),geom.to(device),y.to(device)

            opt.zero_grad()
            out=model(asr,ocr,geom)
            loss=loss_fn(out,y)

            loss.backward()
            opt.step()

            total+=loss.item()

        print(f"Epoch {e+1}: {total/len(loader):.4f}")


# =========================================================
# EVAL
# =========================================================
def evaluate(model,dataset):

    model.eval()

    for idx in range(len(dataset.data)):

        asr_seq, ocr_emb, centers, regions, slide_idx, ocr_path, mouse_path = dataset.data[idx]

        asr = torch.tensor(asr_seq, dtype=torch.float32).to(device)

        if len(ocr_emb) >= MAX_OCR:
            ocr = ocr_emb[:MAX_OCR]
        else:
            pad = np.zeros((MAX_OCR - len(ocr_emb), EMBED_DIM))
            ocr = np.vstack([ocr_emb, pad])

        ocr = torch.tensor(ocr, dtype=torch.float32).to(device)
        centers = torch.tensor(centers, dtype=torch.float32).to(device)

        pts,_=load_mouse_trace(mouse_path)

        T=min(len(pts),len(regions),len(asr_seq))

        preds=[]

        with torch.no_grad():
            for t in range(T-SEQ_LEN):

                seq=asr[t:t+SEQ_LEN].unsqueeze(0)
                out=model(seq,ocr.unsqueeze(0),centers.unsqueeze(0))

                pred=out.argmax(1).item()
                preds.append(pred)

                gt=regions[t+SEQ_LEN]

                if pred==gt:
                    DEBUG["top1"]+=1

                if pred in np.argsort(out.cpu().numpy()[0])[-3:]:
                    DEBUG["top3"]+=1

                DEBUG["count"]+=1

        pred_coords=np.array([centers.cpu().numpy()[p] for p in preds])

        gt_df=mouse.load_trace_data(mouse_path)

        N=min(len(pred_coords),len(gt_df))
        pred_coords=pred_coords[:N]
        gt_coords=gt_df[["x","y"]].values[:N]

        out_path=get_save_path(mouse_path)

        pd.DataFrame({
            "time":np.arange(N),
            "pred_x":pred_coords[:,0],
            "pred_y":pred_coords[:,1],
            "gold_x":gt_coords[:,0],
            "gold_y":gt_coords[:,1]
        }).to_csv(out_path,index=False)

        print("Saved →",out_path)


# =========================================================
# DEBUG PRINT
# =========================================================
def print_debug():

    print("\n===== DEBUG METRICS =====")
    print(f"Attention Entropy: {np.mean(DEBUG['attn_entropy']):.4f}")
    print(f"Attention Peak:    {np.mean(DEBUG['attn_peak']):.4f}")

    if DEBUG["count"]>0:
        print(f"\nTop-1 Align: {DEBUG['top1']/DEBUG['count']:.4f}")
        print(f"Top-3 Align: {DEBUG['top3']/DEBUG['count']:.4f}")


# =========================================================
# MAIN
# =========================================================
if __name__=="__main__":

    train_pairs=build_pairs( ["mlpdataset/data_oct/anat-1" ,
"mlpdataset/data_oct/anat-2",
"mlpdataset/data_oct/bio-1",
"mlpdataset/data_oct/bio-3",
"mlpdataset/data_oct/bio-4",
"mlpdataset/data_oct/dental",
"mlpdataset/data_oct/psy-1",
"mlpdataset/data_oct/psy-2"]  )
    test_pairs=build_pairs(["mlpdataset/data_oct/ml-1",
             "mlpdataset/data_oct/speaking"
             ])

    train_ds=DatasetCrossAttention(train_pairs)
    test_ds=DatasetCrossAttention(test_pairs)

    loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)

    model=CrossAttentionModel().to(device)

    if not os.path.exists("ocr_semantic_asr_FULL_model.pth"):
            train(model,loader)
            torch.save(model.state_dict(),"ocr_semantic_asr_FULL_model.pth")
    else:
            model.load_state_dict(torch.load("ocr_semantic_asr_FULL_model.pth",map_location=torch.device('cpu')))

    
    evaluate(model,test_ds)
    print_debug()