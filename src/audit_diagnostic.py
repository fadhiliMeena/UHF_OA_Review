#!/usr/bin/env python3
"""Reproduce the documented TF-IDF + logistic-regression screening diagnostic.

The classifier is diagnostic only. It does not define the final 5,898-record map.
For calibration, 'Exclude' is the negative class; Include, Map only and Unclear
are treated as retain-for-further-consideration. Five-fold stratified CV is
reported as ROC AUC. With the archived 350-record audit the AUC rounds to 0.902.
"""
from pathlib import Path
import argparse, json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument("--out",type=Path,default=None); args=ap.parse_args()
    repo=args.repo.resolve(); f=repo/"data/manual/audit_350.csv"; df=pd.read_csv(f)
    decision=df["Manual_Audit_Decision"].fillna(df.get("Audit_Decision","")).astype(str).str.strip()
    y=(decision!="Exclude").astype(int)
    text=(df["Title"].fillna("")+" "+df["Abstract_Screening_Text"].fillna("")).astype(str)
    X=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=20000,stop_words="english",sublinear_tf=True).fit_transform(text)
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=20260913)
    model=LogisticRegression(max_iter=2000,C=2.0)
    scores=cross_val_score(model,X,y,cv=cv,scoring="roc_auc")
    result={"n":int(len(df)),"positive_definition":"Include + Map only + Unclear","negative_definition":"Exclude","fold_auc":[float(x) for x in scores],"mean_auc":float(scores.mean()),"rounded_mean_auc":round(float(scores.mean()),3)}
    out=args.out or repo/"outputs/audit_diagnostic.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
