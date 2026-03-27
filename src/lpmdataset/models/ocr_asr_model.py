import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from src.lpmdataset.models.shared import *
from src.lpmdataset.modalities import mouse
from src.lpmdataset.representations.heatmap import HeatMap

# =========================================================
# CONFIG
# =========================================================
TOP_K_BOXES = 80
TEXT_DIM = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# OCR + REGION
# =========================================================
def load_top_ocr_boxes(path, K=TOP_K_BOXES):
    boxes=[]
    with open(path,"r",encoding="utf-8-sig",errors="replace") as f:
        reader=csv.reader(f)
        header=[h.strip().lower() for h in next(reader)]

        if "conf" not in header:
            return None

        li,ti,wi,hi,ci = [header.index(x) for x in ["left","top","width","height","conf"]]

        for row in reader:
            try:
                l,t,w,h,c = float(row[li]),float(row[ti]),float(row[wi]),float(row[hi]),float(row[ci])
            except:
                continue

            if c<=0 or w<8 or h<8 or w*h<100:
                continue

            boxes.append(((w*h)*(c/100.0),(l,t,w,h)))

    if not boxes:
        return None

    boxes.sort(reverse=True)
    return [b for _,b in boxes[:K]]


def build_regions(boxes):
    corners=[]
    corner_to_box=[]
    centers=[]

    for i,(l,t,w,h) in enumerate(boxes):
        pts=[(l,t),(l+w,t),(l,t+h),(l+w,t+h)]
        for p in pts:
            corners.append(p)
            corner_to_box.append(i)
        centers.append([l+w/2,t+h/2])

    corners=np.array(corners)
    corner_to_box=np.array(corner_to_box)
    centers=np.array(centers)

    if len(centers)<TOP_K_BOXES:
        centers=np.vstack([centers,np.zeros((TOP_K_BOXES-len(centers),2))])

    return corners,corner_to_box,centers


def assign_regions(mouse_pts,corners,corner_to_box):
    d=((mouse_pts[:,None,:]-corners[None,:,:])**2).sum(2)
    return corner_to_box[d.argmin(axis=1)]


# =========================================================
# ASR
# =========================================================
def build_vocab(pairs,max_vocab=TEXT_DIM):
    counter=Counter()
    for _,m in pairs:
        p=m.replace("_trace.csv","_spoken.csv")
        if not os.path.exists(p): continue
        df=pd.read_csv(p)
        counter.update(df["Word"].astype(str).str.lower().tolist())

    vocab=[w for w,_ in counter.most_common(max_vocab)]
    return {w:i for i,w in enumerate(vocab)}


def asr_to_vector(path,word_to_idx):
    vec=np.zeros(len(word_to_idx))
    if not os.path.exists(path):
        return vec

    df=pd.read_csv(path)
    for w in df["Word"].astype(str).str.lower():
        if w in word_to_idx:
            vec[word_to_idx[w]]+=1

    if vec.sum()>0:
        vec/=vec.sum()

    return vec


# =========================================================
# DATASET (LAZY)
# =========================================================
class OCRASRDataset(Dataset):

    def __init__(self,pairs,word_to_idx):

        self.data=[]
        print("Building dataset...")

        for i,(ocr_path,mouse_path) in enumerate(pairs):

            boxes=load_top_ocr_boxes(ocr_path)
            if boxes is None: continue

            corners,corner_to_box,centers=build_regions(boxes)

            pts,_=load_mouse_trace(mouse_path)
            if len(pts)<=SEQ_LEN: continue

            regions=assign_regions(pts,corners,corner_to_box)

            text_vec=asr_to_vector(
                mouse_path.replace("_trace.csv","_spoken.csv"),
                word_to_idx
            )

            centers_norm=centers.copy()
            centers_norm[:,0]/=(centers[:,0].max()+1e-6)
            centers_norm[:,1]/=(centers[:,1].max()+1e-6)

            geom=centers_norm.flatten()

            self.data.append((regions,geom,text_vec,centers,i))

        print("Slides:",len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):

        regions,geom,text_vec,centers,slide_idx=self.data[idx]

        i=np.random.randint(0,len(regions)-SEQ_LEN)

        x=np.concatenate([
            np.eye(TOP_K_BOXES)[regions[i:i+SEQ_LEN]],
            np.repeat(geom[None,:],SEQ_LEN,0),
            np.repeat(text_vec[None,:],SEQ_LEN,0)
        ],1)

        y=regions[i+SEQ_LEN]

        return (
            torch.tensor(x,dtype=torch.float32),
            torch.tensor(y,dtype=torch.long),
            torch.tensor(centers,dtype=torch.float32),
            slide_idx
        )


# =========================================================
# MODEL
# =========================================================
class OCRASRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(3*TOP_K_BOXES+TEXT_DIM,64,batch_first=True)
        self.fc=nn.Linear(64,TOP_K_BOXES)

    def forward(self,x):
        o,_=self.lstm(x)
        return self.fc(o[:,-1])


# =========================================================
# TRAIN
# =========================================================
def train_region(model,loader):

    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    loss_fn=nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        total_loss=0

        for x,y,_,_ in loader:

            x=x.to(device)
            y=y.to(device)

            optimizer.zero_grad()
            logits=model(x)
            loss=loss_fn(logits,y)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss/len(loader):.4f}")


# =========================================================
# EVALUATION (SAVE LOGIC UNCHANGED)
# =========================================================
def evaluate_multimodal(model,dataset,test_pairs,idx=0):

    model.eval()

    x,_,centers,slide_idx=dataset[idx]
    x = x.to(device)

    ocr_path,mouse_path=test_pairs[slide_idx]

    # ===== PRECOMPUTE ONCE (FIXED) =====
    pts,_ = load_mouse_trace(mouse_path)
    boxes = load_top_ocr_boxes(ocr_path)
    corners,corner_to_box,_ = build_regions(boxes)
    regions_full = assign_regions(pts,corners,corner_to_box)

    steps = len(regions_full) - SEQ_LEN

    seq=x.unsqueeze(0)
    preds=[]

    with torch.no_grad():

        for step in range(steps):

            r=model(seq).argmax(dim=1).item()
            preds.append(r)

            next_real = regions_full[SEQ_LEN + step]

            onehot=torch.zeros(1,1,TOP_K_BOXES).to(device)
            onehot[0,0,next_real]=1

            geom=seq[:,-1,TOP_K_BOXES:3*TOP_K_BOXES].unsqueeze(1)
            text=seq[:,-1,3*TOP_K_BOXES:].unsqueeze(1)

            seq=torch.cat([seq[:,1:],torch.cat([onehot,geom,text],2)],1)

    pred_coords=np.array([centers.numpy()[r] for r in preds])

    pred_df=pd.DataFrame(pred_coords,columns=["x","y"])
    gt_df=mouse.load_trace_data(mouse_path)

    N=min(len(pred_df),len(gt_df))
    pred_df=pred_df.iloc[:N]
    gt_df=gt_df.iloc[:N]

    # ===== SAVE (UNCHANGED) =====
    video_folder=os.path.basename(os.path.dirname(mouse_path))
    order_folder=os.path.basename(os.path.dirname(os.path.dirname(mouse_path)))
    root_folder=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(mouse_path))))

    slide_name=os.path.basename(mouse_path).replace("_trace.csv","")

    out_dir=os.path.join("results","ocr_asr",root_folder,order_folder,video_folder)
    os.makedirs(out_dir,exist_ok=True)

    out_path=os.path.join(out_dir,f"{slide_name}.csv")

    save_df=pd.DataFrame({
        "time":np.arange(N),
        "pred_x":pred_df["x"].values,
        "pred_y":pred_df["y"].values,
        "gold_x":gt_df["x"].values,
        "gold_y":gt_df["y"].values
    })

    save_df.to_csv(out_path,index=False)
    print("Saved →",out_path)


# =========================================================
# MAIN
# =========================================================
if __name__=="__main__":

    train_pairs=build_slide_pairs_recursive(TRAIN_ROOT)
    test_pairs=build_slide_pairs_recursive(TEST_ROOT)

    vocab=build_vocab(train_pairs)

    train_ds=OCRASRDataset(train_pairs,vocab)
    test_ds=OCRASRDataset(test_pairs,vocab)

    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)

    model=OCRASRModel().to(device)

    if not os.path.exists("ocr_asr_FULL_model.pth"):
        train_region(model,train_loader)
        torch.save(model.state_dict(),"ocr_asr_FULL_model.pth")
    else:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load("ocr_asr_FULL_model.pth"))
        else:   
            model.load_state_dict(torch.load("ocr_asr_FULL_model.pth",map_location=torch.device('cpu')))

    processed=set()

    for i in range(len(test_ds)):
        slide_idx=test_ds.data[i][4]
        if slide_idx in processed:
            continue
        processed.add(slide_idx)

        evaluate_multimodal(model,test_ds,test_pairs,i)