
import json, copy, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, log_loss, average_precision_score)
warnings.filterwarnings("ignore")
SEED=42; rng=np.random.default_rng(SEED); np.random.seed(SEED)
PAL={0:"#2B5F91",1:"#D94A38"}; sns.set_style("whitegrid")
R={}
# ---------- Stage 1-2: Acquire (synthetic) ----------
N=2000
cats=["Electronics","Fashion","Home & Kitchen","Beauty","Books","Sports"]
cat_p=[.22,.26,.18,.14,.10,.10]
base_price={"Electronics":280,"Fashion":55,"Home & Kitchen":90,"Beauty":35,"Books":18,"Sports":70}
regions=["North","South","East","West","Central"]
pays=["Credit Card","UPI","Debit Card","COD","Wallet"]
devs=["Mobile","Desktop","Tablet"]
df=pd.DataFrame({
 "order_month":rng.choice(np.arange(1,13),N,p=(lambda w:w/w.sum())(np.array([6,6,7,7,7,7,8,8,8,10,13,13],dtype=float))),
 "customer_age":np.clip(rng.normal(34,10,N),18,70).round().astype(int),
 "gender":rng.choice(["F","M"],N),
 "region":rng.choice(regions,N,p=[.2,.25,.2,.2,.15]),
 "category":rng.choice(cats,N,p=cat_p),
 "payment_method":rng.choice(pays,N,p=[.28,.34,.16,.14,.08]),
 "device":rng.choice(devs,N,p=[.62,.30,.08]),
})
df["unit_price"]=[round(float(rng.lognormal(np.log(base_price[c]),0.45)),2) for c in df.category]
df["quantity"]=np.clip(rng.poisson(1.6,N)+1-(df.category=="Electronics")*1,1,8)
df["discount_pct"]=np.clip(rng.normal(12,9,N)+(df.order_month>=11)*8,0,60).round(1)
df["shipping_days"]=np.clip(rng.gamma(3,1.4,N)+(df.region=="Central")*1.5,1,18).round().astype(int)
df["is_prime"]=rng.binomial(1,.38,N)
df.loc[df.is_prime==1,"shipping_days"]=np.clip(df.loc[df.is_prime==1,"shipping_days"]-2,1,18)
df["prior_orders"]=rng.poisson(2.2+1.5*df.is_prime,N)
df["session_minutes"]=np.clip(rng.gamma(2.5,5,N)+(df.device=="Desktop")*3,1,90).round(1)
cat_eff=df.category.map({"Electronics":0,"Fashion":.1,"Home & Kitchen":.0,"Beauty":.25,"Books":.35,"Sports":.05})
lp=(-0.2+0.22*df.prior_orders+0.5*df.is_prime-0.17*(df.shipping_days-4)+0.025*df.discount_pct
    +cat_eff-0.0012*(df.unit_price*df.quantity).clip(upper=600)+0.012*(df.session_minutes-15)
    -0.3*(df.payment_method=="COD")+rng.normal(0,0.9,N))
df["customer_rating"]=np.clip((4.1-0.12*(df.shipping_days-4)+0.3*(lp-lp.mean())/lp.std()+rng.normal(0,.6,N)),1,5).round(1)
df["repeat_purchase"]=(rng.random(N)<1/(1+np.exp(-lp))).astype(int)
# missing values (real-world style)
df.loc[rng.choice(N,38,replace=False),"customer_rating"]=np.nan
df.loc[rng.choice(N,22,replace=False),"session_minutes"]=np.nan
# engineered
df["order_value"]=(df.unit_price*df.quantity*(1-df.discount_pct/100)).round(2)
df["discount_amount"]=(df.unit_price*df.quantity*df.discount_pct/100).round(2)
df["value_per_session_min"]=(df.order_value/df.session_minutes).round(2)
df["late_delivery_flag"]=(df.shipping_days>=7).astype(int)
df["loyalty_score"]=(df.prior_orders*2+df.is_prime*3+(df.customer_rating.fillna(df.customer_rating.median())>=4).astype(int)*2)
df["target"]=df.repeat_purchase
df.insert(0,"order_id",["ORD%05d"%i for i in range(1,N+1)])
df.to_csv("data/ecommerce_sales_synthetic.csv",index=False)
R["shape"]=df.shape; R["n_features"]=df.shape[1]-2
R["dups"]=int(df.duplicated(subset=[c for c in df.columns if c!="order_id"]).sum())
R["missing"]={k:int(v) for k,v in df.isnull().sum().items() if v>0}
R["class_counts"]=df.target.value_counts().sort_index().to_dict()
cont=["customer_age","unit_price","quantity","discount_pct","shipping_days","order_value","session_minutes","customer_rating"]
desc=df[cont].describe().T
desc["skew"]=df[cont].skew(); desc["kurt"]=df[cont].kurt()
R["desc"]=desc.round(2).reset_index().to_dict("records")
# outliers
out={}
for c in ["unit_price","order_value","shipping_days","session_minutes","discount_pct"]:
    q1,q3=df[c].quantile([.25,.75]); i=q3-q1; lo,hi=q1-1.5*i,q3+1.5*i
    out[c]=dict(iqr=round(i,2),lo=round(lo,2),hi=round(hi,2),n=int(((df[c]<lo)|(df[c]>hi)).sum()),mx=float(df[c].max()))
R["outliers"]=out
# ---------- Split ----------
feat=df.drop(columns=["order_id","repeat_purchase","target"])
y=df.target.values
Xtr,Xtmp,ytr,ytmp=train_test_split(feat,y,test_size=.30,stratify=y,random_state=SEED)
Xva,Xte,yva,yte=train_test_split(Xtmp,ytmp,test_size=.50,stratify=ytmp,random_state=SEED)
R["split"]={k:[len(a),int((b==0).sum()),int((b==1).sum())] for k,a,b in [("train",Xtr,ytr),("val",Xva,yva),("test",Xte,yte)]}
for c in ["customer_rating","session_minutes"]:
    med=Xtr[c].median()
    for p in (Xtr,Xva,Xte): p[c]=p[c].fillna(med)
R["impute"]={c:round(float(Xtr[c].median()),2) for c in ["customer_rating","session_minutes"]}
for p in (Xtr,Xva,Xte):
    p["value_per_session_min"]=(p.order_value/p.session_minutes).round(2)
    p["loyalty_score"]=p.prior_orders*2+p.is_prime*3+(p.customer_rating>=4).astype(int)*2
catc=["gender","region","category","payment_method","device"]
Etr=pd.get_dummies(Xtr,columns=catc,drop_first=True)
Eva=pd.get_dummies(Xva,columns=catc,drop_first=True).reindex(columns=Etr.columns,fill_value=0)
Ete=pd.get_dummies(Xte,columns=catc,drop_first=True).reindex(columns=Etr.columns,fill_value=0)
numc=["order_month","customer_age","unit_price","quantity","discount_pct","shipping_days","prior_orders","session_minutes","customer_rating","order_value","discount_amount","value_per_session_min","loyalty_score"]
sc=StandardScaler(); Etr[numc]=sc.fit_transform(Etr[numc]); Eva[numc]=sc.transform(Eva[numc]); Ete[numc]=sc.transform(Ete[numc])
Etr,Eva,Ete=[e.astype(float) for e in (Etr,Eva,Ete)]
R["input_dim"]=Etr.shape[1]
# ---------- Stats tests ----------
t={}
for c in ["order_value","shipping_days","discount_pct","prior_orders","session_minutes"]:
    a=df[df.target==1][c].dropna(); b=df[df.target==0][c].dropna()
    tt=stats.ttest_ind(a,b,equal_var=False); mw=stats.mannwhitneyu(a,b)
    t[c]=dict(m1=round(a.mean(),2),m0=round(b.mean(),2),med1=round(a.median(),2),med0=round(b.median(),2),t=round(tt.statistic,2),p=float(tt.pvalue),mwp=float(mw.pvalue))
R["ttests"]=t
chi={}
for c in ["category","region","payment_method","device","is_prime"]:
    ct=pd.crosstab(df[c],df.target); chi2,p,dof,_=stats.chi2_contingency(ct)
    v=np.sqrt(chi2/(ct.values.sum()*(min(ct.shape)-1)))
    chi[c]=dict(chi2=round(chi2,2),p=float(p),dof=int(dof),v=round(v,3))
R["chi"]=chi
R["rate_by_cat"]=(df.groupby("category").target.mean()*100).round(1).to_dict()
R["rate_by_pay"]=(df.groupby("payment_method").target.mean()*100).round(1).to_dict()
R["rate_prime"]=(df.groupby("is_prime").target.mean()*100).round(1).to_dict()
R["rev_by_cat"]=df.groupby("category").order_value.sum().round(0).sort_values(ascending=False).to_dict()
R["rev_total"]=round(float(df.order_value.sum()),0); R["aov"]=round(float(df.order_value.mean()),2)
R["rev_by_month"]=df.groupby("order_month").order_value.sum().round(0).to_dict()
R["rev_by_region"]=df.groupby("region").order_value.sum().round(0).to_dict()
corr=df[cont+["prior_orders","is_prime","target"]].corr()
R["corr_target"]=corr.target.drop("target").round(2).sort_values().to_dict()
r_late=stats.pearsonr(df.shipping_days,df.customer_rating.fillna(df.customer_rating.median()))
R["ship_rating_r"]=[round(r_late[0],3),float(r_late[1])]
# ---------- Figures ----------
def sv(n): plt.savefig(f"fig/{n}.png",dpi=200,bbox_inches="tight"); plt.close()
# F1 class dist + revenue by category
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
c=df.target.value_counts().sort_index()
b=ax[0].bar(["One-time (0)","Repeat (1)"],c.values,color=[PAL[0],PAL[1]],edgecolor="k")
for r,v in zip(b,c.values): ax[0].text(r.get_x()+r.get_width()/2,v+15,f"{v}\n({v/N*100:.1f}%)",ha="center",fontsize=9,fontweight="bold")
ax[0].set_ylim(0,c.max()*1.2); ax[0].set_ylabel("Orders"); ax[0].set_title("Repeat-Purchase Target Distribution",fontweight="bold")
rv=df.groupby("category").order_value.sum().sort_values()
ax[1].barh(rv.index,rv.values/1000,color="#2B5F91",edgecolor="k")
for i,v in enumerate(rv.values/1000): ax[1].text(v+1,i,f"{v:,.0f}K",va="center",fontsize=9)
ax[1].set_title("Total Revenue by Category (thousands)",fontweight="bold"); ax[1].set_xlim(0,rv.max()/1000*1.15)
plt.tight_layout(); sv("f1")
# F2 order value distribution
fig,ax=plt.subplots(1,2,figsize=(11,4.2))
sns.histplot(data=df,x="order_value",hue="target",palette=PAL,kde=True,bins=45,ax=ax[0],log_scale=True)
ax[0].set_title("Order Value Distribution (log scale) by Repeat Status",fontweight="bold"); ax[0].set_xlabel("Order value (log scale)")
sns.violinplot(data=df,x="target",y="order_value",hue="target",palette=PAL,ax=ax[1],cut=0,legend=False)
ax[1].set_yscale("log"); ax[1].set_xticklabels(["One-time (0)","Repeat (1)"]); ax[1].set_title("Violin Spread of Order Value",fontweight="bold"); ax[1].set_xlabel("")
plt.tight_layout(); sv("f2")
# F3 boxplots
fig,axes=plt.subplots(2,2,figsize=(11,7.5))
for ax,cc in zip(axes.flat,["order_value","shipping_days","discount_pct","session_minutes"]):
    sns.boxplot(data=df,x="target",y=cc,hue="target",palette=PAL,ax=ax,width=.45,fliersize=3,legend=False)
    q1,q3=df[cc].quantile([.25,.75]); ax.axhline(q3+1.5*(q3-q1),color="red",ls=":",label="Upper IQR fence")
    ax.set_title(cc,fontweight="bold"); ax.set_xticklabels(["One-time","Repeat"]); ax.set_xlabel(""); ax.legend(fontsize=8)
plt.tight_layout(); sv("f3")
# F4 category/prime/payment rates
fig,ax=plt.subplots(1,3,figsize=(13,4.2))
rc=df.groupby("category").target.mean().sort_values()*100
ax[0].barh(rc.index,rc.values,color="#D94A38",edgecolor="k"); ax[0].set_title("Repeat Rate by Category (%)",fontweight="bold")
for i,v in enumerate(rc.values): ax[0].text(v+.5,i,f"{v:.1f}",va="center",fontsize=8)
ax[0].set_xlim(0,rc.max()*1.2)
rp=df.groupby("payment_method").target.mean().sort_values()*100
ax[1].barh(rp.index,rp.values,color="#3B8A94",edgecolor="k"); ax[1].set_title("Repeat Rate by Payment Method (%)",fontweight="bold")
for i,v in enumerate(rp.values): ax[1].text(v+.5,i,f"{v:.1f}",va="center",fontsize=8)
ax[1].set_xlim(0,rp.max()*1.2)
rpr=df.groupby("is_prime").target.mean()*100
ax[2].bar(["Non-Prime","Prime"],rpr.values,color=["#2B5F91","#D94A38"],edgecolor="k"); ax[2].set_title("Repeat Rate by Prime Membership (%)",fontweight="bold")
for i,v in enumerate(rpr.values): ax[2].text(i,v+1,f"{v:.1f}",ha="center",fontsize=9)
ax[2].set_ylim(0,100)
plt.tight_layout(); sv("f4")
# F5 correlation
cc=["customer_age","unit_price","quantity","discount_pct","shipping_days","prior_orders","is_prime","session_minutes","customer_rating","order_value","loyalty_score","target"]
cm=df[cc].corr(); mask=np.triu(np.ones_like(cm,dtype=bool))
plt.figure(figsize=(9,7)); sns.heatmap(cm,mask=mask,annot=True,fmt=".2f",cmap="coolwarm",vmin=-.6,vmax=.6,linewidths=.5,annot_kws={"size":8})
plt.title("Pearson Correlation Heatmap",fontweight="bold"); sv("f5")
# F6 monthly revenue + shipping vs rating
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
mr=df.groupby("order_month").order_value.sum()/1000
ax[0].plot(mr.index,mr.values,marker="o",color="#2B5F91",lw=2); ax[0].fill_between(mr.index,mr.values,alpha=.15,color="#2B5F91")
ax[0].set_xticks(range(1,13)); ax[0].set_xlabel("Month"); ax[0].set_ylabel("Revenue (thousands)"); ax[0].set_title("Monthly Revenue Seasonality",fontweight="bold")
tmp=df.assign(r=df.customer_rating)
sns.regplot(data=tmp[tmp.target==0],x="shipping_days",y="r",x_jitter=.25,scatter_kws=dict(alpha=.25,s=14),color=PAL[0],ax=ax[1],label="One-time")
sns.regplot(data=tmp[tmp.target==1],x="shipping_days",y="r",x_jitter=.25,scatter_kws=dict(alpha=.25,s=14),color=PAL[1],ax=ax[1],label="Repeat")
ax[1].set_xlabel("Shipping days"); ax[1].set_ylabel("Customer rating"); ax[1].set_title("Shipping Delay vs Customer Rating",fontweight="bold"); ax[1].legend()
plt.tight_layout(); sv("f6")
# ---------- Models ----------
cfgs={"Config A (Shallow Fast)":dict(h=(32,),a=1e-3,lr=5e-3),
      "Config B (Tuned Deep MLP)":dict(h=(32,16),a=1e-2,lr=1e-3),
      "Config C (Wide Deep MLP)":dict(h=(64,32),a=1e-3,lr=5e-4)}
def train(cfg,epochs=150,patience=15):
    m=MLPClassifier(hidden_layer_sizes=cfg["h"],alpha=cfg["a"],learning_rate_init=cfg["lr"],batch_size=32,random_state=SEED,max_iter=1)
    cls=np.array([0,1]); hist=dict(tl=[],vl=[],ta=[],va=[]); best=(1e9,None,0); bad=0
    for ep in range(1,epochs+1):
        idx=np.random.RandomState(SEED+ep).permutation(len(Etr))
        m.partial_fit(Etr.values[idx],ytr[idx],classes=cls)
        ptr=m.predict_proba(Etr)[:,1]; pva=m.predict_proba(Eva)[:,1]
        hist["tl"].append(log_loss(ytr,ptr)); hist["vl"].append(log_loss(yva,pva))
        hist["ta"].append(accuracy_score(ytr,ptr>.5)); hist["va"].append(accuracy_score(yva,pva>.5))
        if hist["vl"][-1]<best[0]-1e-4: best=(hist["vl"][-1],copy.deepcopy(m),ep); bad=0
        else: bad+=1
        if bad>=patience: break
    return best,hist
res={}
for k,cfg in cfgs.items():
    best,h=train(cfg); mm=best[1]
    pva=mm.predict_proba(Eva)[:,1]
    nparams=sum(w.size for w in mm.coefs_)+sum(b.size for b in mm.intercepts_)
    res[k]=dict(loss=best[0],auc=roc_auc_score(yva,pva),ep=best[2],np=int(nparams),hist=h,model=mm,cfg=cfg)
bk=min(res,key=lambda k:res[k]["loss"]); R["tuning"]={k:dict(h=list(v["cfg"]["h"]),lr=v["cfg"]["lr"],alpha=v["cfg"]["a"],loss=round(v["loss"],4),auc=round(v["auc"],4),ep=v["ep"],np=v["np"]) for k,v in res.items()}
R["best_cfg"]=bk
mlp=res[bk]["model"]; H=res[bk]["hist"]; be=res[bk]["ep"]; R["best_epoch"]=be; R["epochs_run"]=len(H["tl"])
R["best_val_acc"]=round(H["va"][be-1],4); R["best_val_loss"]=round(H["vl"][be-1],4)
R["train_loss0"]=round(H["tl"][0],3); R["train_loss_end"]=round(H["tl"][-1],3); R["train_acc_end"]=round(H["ta"][-1],3)
# curves
ep=np.arange(1,len(H["tl"])+1)
plt.figure(figsize=(8.5,4.6)); plt.plot(ep,H["tl"],lw=2,color="#2B5F91",label="Training Loss"); plt.plot(ep,H["vl"],lw=2,color="#D94A38",label="Validation Loss")
plt.axvline(be,color="green",ls="--",label=f"Best checkpoint (epoch {be}, val loss = {H['vl'][be-1]:.4f})"); plt.scatter([be],[H["vl"][be-1]],color="green",zorder=5)
plt.xlabel("Epoch"); plt.ylabel("Binary cross-entropy"); plt.title("Training and Validation Loss vs Epoch",fontweight="bold"); plt.legend(); plt.grid(alpha=.3); sv("f7")
plt.figure(figsize=(8.5,4.6)); plt.plot(ep,H["ta"],lw=2,color="#2B5F91",label="Training Accuracy"); plt.plot(ep,H["va"],lw=2,color="#D94A38",label="Validation Accuracy")
plt.axvline(be,color="green",ls="--",label=f"Best checkpoint (epoch {be}, val acc = {H['va'][be-1]*100:.1f}%)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training and Validation Accuracy vs Epoch",fontweight="bold"); plt.legend(); plt.grid(alpha=.3); sv("f8")
# test eval
pt=mlp.predict_proba(Ete)[:,1]
fpr,tpr,thr=roc_curve(yte,pt); J=tpr-fpr; jt=thr[J.argmax()]
def ev(p,th=.5):
    pr=(p>=th).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pr).ravel()
    return dict(acc=accuracy_score(yte,pr),prec=precision_score(yte,pr),rec=recall_score(yte,pr),spec=tn/(tn+fp),f1=f1_score(yte,pr),auc=roc_auc_score(yte,p),ap=average_precision_score(yte,p),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
R["mlp"]=ev(pt); R["mlp_youden"]=dict(thr=round(float(jt),3),tpr=round(float(tpr[J.argmax()]),3),fpr=round(float(fpr[J.argmax()]),3))
lr=LogisticRegression(max_iter=2000,C=1.0,random_state=SEED).fit(Etr,ytr); R["lr"]=ev(lr.predict_proba(Ete)[:,1])
cmx=confusion_matrix(yte,(pt>=.5).astype(int)); tn,fp,fn,tp=cmx.ravel(); pct=cmx/cmx.sum()*100
lab=np.array([[f"True Negative (TN)\n{tn}\n({pct[0,0]:.1f}%)",f"False Positive (FP)\n{fp}\n({pct[0,1]:.1f}%)"],[f"False Negative (FN)\n{fn}\n({pct[1,0]:.1f}%)",f"True Positive (TP)\n{tp}\n({pct[1,1]:.1f}%)"]])
plt.figure(figsize=(6.6,5.4)); sns.heatmap(cmx,annot=lab,fmt="",cmap="Blues",cbar=False,linewidths=2,linecolor="k",xticklabels=["Predicted One-time (0)","Predicted Repeat (1)"],yticklabels=["Actual One-time (0)","Actual Repeat (1)"],annot_kws={"size":10,"fontweight":"bold"})
plt.title(f"Confusion Matrix on Unseen Test Set (n = {len(yte)})",fontweight="bold"); sv("f9")
plt.figure(figsize=(6.6,5.4)); plt.plot(fpr,tpr,color="#D94A38",lw=2.4,label=f"MLP (AUC = {R['mlp']['auc']:.4f})")
fl,tl,_=roc_curve(yte,lr.predict_proba(Ete)[:,1]); plt.plot(fl,tl,color="#2B5F91",lw=1.8,label=f"Logistic Regression (AUC = {R['lr']['auc']:.4f})")
plt.plot([0,1],[0,1],"k:",label="Random (AUC = 0.5000)"); plt.scatter([fpr[J.argmax()]],[tpr[J.argmax()]],color="k",zorder=5,label=f"Youden's J cutoff = {jt:.3f}")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves on Unseen Test Set",fontweight="bold"); plt.legend(loc="lower right",fontsize=8); plt.grid(alpha=.3); sv("f10")
pi=permutation_importance(mlp,Ete,yte,scoring="roc_auc",n_repeats=10,random_state=SEED)
o=np.argsort(-pi.importances_mean)[:10]
R["perm"]=[[Etr.columns[i],round(float(pi.importances_mean[i]),5),round(float(pi.importances_std[i]),5)] for i in o]
json.dump(R,open("results.json","w"),indent=1,default=lambda x: x.item() if hasattr(x,"item") else str(x))
print(json.dumps(R,indent=1,default=lambda x: x.item() if hasattr(x,"item") else str(x)))
