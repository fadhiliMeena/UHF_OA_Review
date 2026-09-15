"""
Reproduce the data-driven knowledge-mapping figures for:

Health Financing for Universal Health Coverage in Low- and
Lower-Middle-Income Countries: Potential and Challenges

Inputs
------
../Supplementary/Supplementary_Data_S3_Systematic_Map_Corpus_5898.csv
../Reproducibility/financing_theme_assignments.csv

Outputs
-------
Figure 6: potential/challenge signal percentages by exact financing theme
Figure 7: alluvial theme -> UHC outcome -> signal co-occurrence map
Figure 8: title/abstract term co-occurrence network and density map
Figure 9: financing-theme/UHC-outcome chord diagram
Derived CSV tables underlying every added visualisation

Important
---------
These are descriptive knowledge-development analyses of the 5,898-record
OpenAlex corpus. Categories are non-mutually-exclusive. Counts are
co-occurrences, not effect sizes, causal estimates, or quality-weighted
systematic-review conclusions.
"""
from pathlib import Path
import re, math, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle, Wedge
import networkx as nx
from scipy.ndimage import gaussian_filter

ROOT = Path(os.environ.get('HPP_REPO_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
CORPUS = Path(os.environ.get('HPP_CORPUS', str(ROOT / 'data/frozen/systematic_map_corpus_5898.csv'))).resolve()
THEME_ASSIGN = Path(os.environ.get('HPP_THEME_ASSIGN', str(ROOT / 'data/frozen/financing_theme_assignments.csv'))).resolve()
FIGDIR = Path(os.environ.get('HPP_FIGDIR', str(ROOT / 'outputs/figures'))).resolve()
DERIVED = Path(os.environ.get('HPP_DERIVED', str(ROOT / 'outputs/derived'))).resolve()
FIGDIR.mkdir(exist_ok=True)
DERIVED.mkdir(exist_ok=True)
SEED = 20260914
TIFF_DPI = int(os.getenv("HPP_TIFF_DPI", "300"))
SAVE_VECTOR = os.getenv("HPP_SAVE_VECTOR", "0") == "1"

OUTCOMES = {
    "Coverage/entitlement":[r"\bcoverage\b",r"\benrol+l?ment\b",r"\benrollment\b",r"\bentitlement\b",r"\bbenefit package\b"],
    "Access/utilization":[r"\baccess\b",r"\butili[sz]ation\b",r"\bservice use\b",r"\bhealth(?:care| care) use\b"],
    "Financial protection":[r"\bfinancial (?:risk )?protection\b",r"\bcatastrophic (?:health|medical) (?:expenditure|spending|payment|cost)\b",r"\bout[- ]of[- ]pocket\b",r"\bimpoverish(?:ment|ing)?\b",r"\bfinancial hardship\b"],
    "Equity":[r"\bequit(?:y|able)\b",r"\binequit(?:y|ies|able)\b",r"\binequalit(?:y|ies)\b",r"\bpro[- ]poor\b",r"\bprogressiv(?:e|ity)\b",r"\bregressiv(?:e|ity)\b"],
    "Efficiency/value":[r"\befficien(?:cy|t)\b",r"\bvalue for money\b",r"\bresource allocation\b"],
    "Quality":[r"\bquality of care\b",r"\bservice quality\b",r"\bservice readiness\b"],
    "Fiscal sustainability/resources":[r"\bfiscal space\b",r"\bfiscal sustainab(?:ility|le)\b",r"\bfinancial sustainab(?:ility|le)\b",r"\bresource mobili[sz]ation\b",r"\bgovernment spending\b",r"\bpublic (?:health )?expenditure\b",r"\brevenue mobil"],
    "Pooling/fragmentation":[r"\brisk pooling\b",r"\bpooling\b",r"\bfragment(?:ation|ed)\b",r"\brisk pool\b",r"\bcross[- ]subsid"],
    "Governance/implementation":[r"\bgovernance\b",r"\bstewardship\b",r"\bimplementation\b",r"\badministrative (?:capacity|cost|burden|challenge)\b",r"\baccountability\b",r"\binstitutional capacity\b"],
}

POTENTIAL = {
    "Coverage/access potential":[r"\bexpand(?:ed|ing)? (?:population )?coverage\b",r"\bincreas(?:e|ed|es|ing) (?:healthcare |health care )?(?:access|utili[sz]ation|coverage)\b",r"\bimprov(?:e|ed|es|ing) (?:access|utili[sz]ation|coverage)\b"],
    "Financial-protection potential":[r"\breduc(?:e|ed|es|ing) (?:out[- ]of[- ]pocket|catastrophic|financial burden)",r"\bimprov(?:e|ed|es|ing) financial protection\b",r"\bprotect(?:s|ed|ing)? households?\b"],
    "Equity potential":[r"\bimprov(?:e|ed|es|ing) equit",r"\bmore equitable\b",r"\bprogressive financ",r"\bsubsid(?:y|ies|ized|ised|ization|isation)"],
    "Resource/pooling potential":[r"\bresource mobili[sz]ation\b",r"\brisk sharing\b",r"\bcross[- ]subsid",r"\blarger risk pool",r"\bconsolidat(?:e|ed|ion) (?:risk )?pool"],
    "Efficiency/quality potential":[r"\bimprov(?:e|ed|es|ing) efficien",r"\bimprov(?:e|ed|es|ing) quality\b",r"\bbetter value\b"],
}

CHALLENGES = {
    "Affordability/OOP challenge":[r"\bhigh out[- ]of[- ]pocket\b",r"\bcatastrophic (?:health|medical) (?:expenditure|spending|payment|cost)",r"\bfinancial hardship\b",r"\bfinancial barrier",r"\buser fee"],
    "Coverage/enrolment challenge":[r"\blow (?:insurance )?(?:coverage|enrol+ment)\b",r"\bcoverage gaps?\b",r"\bunder[- ]?enrol",r"\bnon[- ]?enrol",r"\bvoluntary enrol"],
    "Informality/selection challenge":[r"\binformal sector\b",r"\binformal workers?\b",r"\badverse selection\b",r"\brisk selection\b"],
    "Fragmentation/pooling challenge":[r"\bfragment(?:ed|ation)\b",r"\bsmall (?:risk )?pools?\b",r"\bmultiple (?:risk )?pools?\b"],
    "Fiscal/sustainability challenge":[r"\blimited fiscal space\b",r"\bconstrained fiscal space\b",r"\binsufficient (?:public )?fund",r"\bunderfund",r"\bfinancial sustainability\b",r"\bfiscal sustainability\b",r"\bunsustain"],
    "Governance/capacity challenge":[r"\bweak governance\b",r"\badministrative (?:capacity|cost|burden|challenge)",r"\bimplementation (?:challenge|barrier|constraint)",r"\bcorruption\b",r"\bleakage\b"],
    "Donor-dependence challenge":[r"\bdonor depend",r"\baid depend",r"\bexternal depend",r"\bvolatile (?:aid|funding|financing)",r"\bdeclining aid\b"],
    "Supply/provider challenge":[r"\bservice readiness\b",r"\bprovider (?:shortage|behavio|response|incentive)",r"\bsupply[- ]side (?:constraint|barrier)"],
    "Inequity/regressivity challenge":[r"\binequit",r"\bregressive\b",r"\bdisparit",r"\bexclusion\b"],
}

TERM_PATTERNS = {
    "universal health coverage":[r"\buniversal health coverage\b",r"\buhc\b"],
    "health financing":[r"\bhealth financ(?:e|ing)\b",r"\bhealthcare financ(?:e|ing)\b"],
    "financial protection":[r"\bfinancial protection\b"],
    "out-of-pocket payments":[r"\bout[- ]of[- ]pocket\b",r"\boop\b"],
    "catastrophic health expenditure":[r"\bcatastrophic (?:health )?expenditure\b",r"\bcatastrophic spending\b"],
    "impoverishment":[r"\bimpoverish(?:ment|ing)?\b"],
    "equity":[r"\bequit(?:y|able)\b",r"\binequit(?:y|ies|able)\b"],
    "health insurance":[r"\bhealth insurance\b"],
    "social health insurance":[r"\bsocial health insurance\b"],
    "community-based health insurance":[r"\bcommunity[- ]based (?:health )?insurance\b",r"\bcbhi\b"],
    "risk pooling":[r"\brisk pool(?:ing)?\b",r"\bpooling\b"],
    "prepayment":[r"\bpre[- ]?payment\b"],
    "tax financing":[r"\btax(?:ation)? financ(?:e|ing)\b",r"\btax[- ]based\b"],
    "government spending":[r"\bgovernment (?:health )?spending\b",r"\bpublic (?:health )?expenditure\b"],
    "fiscal space":[r"\bfiscal space\b"],
    "fiscal sustainability":[r"\bfiscal sustainab(?:ility|le)\b",r"\bsustainable financing\b"],
    "donor financing":[r"\bdonor financ(?:e|ing)\b",r"\bexternal financ(?:e|ing)\b",r"\bdevelopment assistance for health\b"],
    "user fees":[r"\buser fee(?:s)?\b"],
    "cost sharing":[r"\bcost[- ]sharing\b",r"\bco[- ]?payment(?:s)?\b",r"\bcoinsurance\b"],
    "strategic purchasing":[r"\bstrategic purchasing\b"],
    "provider payment":[r"\bprovider payment\b",r"\bpayment reform\b"],
    "performance-based financing":[r"\bperformance[- ]based financ(?:e|ing)\b",r"\bresults[- ]based financ(?:e|ing)\b",r"\bpbf\b"],
    "governance":[r"\bgovernance\b",r"\bstewardship\b"],
    "access":[r"\baccess\b"],
    "utilization":[r"\butili[sz]ation\b",r"\bservice use\b"],
    "quality of care":[r"\bquality of care\b",r"\bservice quality\b"],
    "efficiency":[r"\befficien(?:cy|t)\b",r"\bvalue for money\b"],
    "primary health care":[r"\bprimary health care\b",r"\bprimary healthcare\b"],
    "informal sector":[r"\binformal sector\b",r"\binformal worker(?:s)?\b"],
    "vulnerable populations":[r"\bvulnerable (?:population|group)s?\b"],
    "rural populations":[r"\brural (?:population|community|household)s?\b"],
    "fragmentation":[r"\bfragment(?:ation|ed)\b"],
    "coverage":[r"\bcoverage\b"],
    "enrolment":[r"\benrol+l?ment\b",r"\benrollment\b"],
    "benefit package":[r"\bbenefit package\b"],
    "health outcomes":[r"\bhealth outcomes?\b"],
    "provider incentives":[r"\bprovider incentive(?:s)?\b"],
    "accountability":[r"\baccountability\b"],
    "implementation":[r"\bimplementation\b"],
}

def compile_map(mapping):
    return {k:re.compile("|".join(f"(?:{p})" for p in v),re.I) for k,v in mapping.items()}

def any_match(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)

def savefig(fig, stem):
    fig.savefig(FIGDIR/f"{stem}.jpg", dpi=150, bbox_inches="tight", pil_kwargs={"quality":95})
    fig.savefig(FIGDIR/f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGDIR/f"{stem}.tif", dpi=TIFF_DPI, bbox_inches="tight", pil_kwargs={"compression":"tiff_lzw"})
    if SAVE_VECTOR:
        fig.savefig(FIGDIR/f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGDIR/f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

# The complete plotting implementation used for the submitted files is
# intentionally kept in this executable script. Run from Reproducibility/:
#     python run_data_driven_knowledge_maps.py
#
# To avoid duplicating several hundred lines in prose documentation, the
# code below is generated from the transparent data files above.
# The package also contains the exact CSV matrices/edge lists underlying
# Figures 7-9, making every ribbon, node and link auditable.

pub = pd.read_csv(CORPUS)
themes = pd.read_csv(THEME_ASSIGN)
pub["text"] = (pub["title"].fillna("")+" "+pub["abstract"].fillna("")).str.lower()

# ----- outcome and signal coding -----
for name,pat in compile_map(OUTCOMES).items():
    pub["O:"+name] = pub["text"].str.contains(pat, regex=True)
for name,pats in POTENTIAL.items():
    pub["P:"+name] = pub["text"].map(lambda x,p=pats:any_match(x,p))
for name,pats in CHALLENGES.items():
    pub["C:"+name] = pub["text"].map(lambda x,p=pats:any_match(x,p))

pcols=["P:"+x for x in POTENTIAL]; ccols=["C:"+x for x in CHALLENGES]
pub["potential_signal"]=pub[pcols].any(axis=1)
pub["challenge_signal"]=pub[ccols].any(axis=1)
pub["signal_class"]=np.select(
    [pub.potential_signal & pub.challenge_signal,pub.potential_signal,pub.challenge_signal],
    ["Both","Potential only","Challenge only"],default="No mapped signal")

signal=[]
for theme,g in themes.groupby("theme"):
    sub=pub[pub.work_id.isin(g.work_id.unique())]
    p=sub.potential_signal; c=sub.challenge_signal
    signal.append({
        "Financing theme":theme,"Records":len(sub),
        "Any potential signal":int(p.sum()),"Any challenge signal":int(c.sum()),
        "Both potential and challenge":int((p&c).sum()),
        "Potential signal %":round(100*p.mean(),1),
        "Challenge signal %":round(100*c.mean(),1),
        "Both %":round(100*(p&c).mean(),1)})
signal=pd.DataFrame(signal).sort_values("Records",ascending=False)
signal_out = Path(os.environ.get('HPP_SIGNAL_OUT', str(ROOT / 'outputs/abstract_potential_challenge_signals.csv'))).resolve(); signal_out.parent.mkdir(parents=True, exist_ok=True); signal.to_csv(signal_out,index=False)

# Figure 6
plot6=signal[signal["Financing theme"]!="Willingness to pay"].sort_values("Records")
y=np.arange(len(plot6)); h=.36
fig,ax=plt.subplots(figsize=(8.2,5.6))
ax.barh(y-h/2,plot6["Potential signal %"],height=h,label="Potential signal")
ax.barh(y+h/2,plot6["Challenge signal %"],height=h,label="Challenge signal")
ax.set_yticks(y,plot6["Financing theme"]); ax.set_xlabel("Theme-specific records containing signal (%)")
ax.legend(frameon=False,ncol=2,loc="lower right"); ax.grid(axis="x",alpha=.2)
fig.tight_layout(); savefig(fig,"Figure_6_Health_Financing_Potential_and_Challenge_Signals")

# ----- theme/outcome pairs and triples -----
long=themes[["work_id","theme"]].merge(pub[["work_id","signal_class"]+["O:"+x for x in OUTCOMES]],on="work_id")
pairs=[]; triples=[]
for theme,g in long.groupby("theme"):
    for outcome in OUTCOMES:
        m=g["O:"+outcome].fillna(False); n=int(m.sum())
        if n:pairs.append({"financing_theme":theme,"uhc_outcome":outcome,"records":n})
        for s in ["Potential only","Challenge only","Both"]:
            ns=int((m & (g.signal_class==s)).sum())
            if ns:triples.append({"financing_theme":theme,"uhc_outcome":outcome,"signal_class":s,"cooccurrence_events":ns})
pairs=pd.DataFrame(pairs); triples=pd.DataFrame(triples)
pairs.to_csv(DERIVED/"Figure_9_Theme_Outcome_Cooccurrence.csv",index=False)
triples.to_csv(DERIVED/"Figure_7_Theme_Outcome_Signal_Triples.csv",index=False)

# Figure 7 alluvial
out_order=triples.groupby("uhc_outcome").cooccurrence_events.sum().sort_values(ascending=False).head(7).index.tolist()
tri=triples[triples.uhc_outcome.isin(out_order)]
L=tri.groupby(["financing_theme","uhc_outcome"],as_index=False).cooccurrence_events.sum()
R=tri.groupby(["uhc_outcome","signal_class"],as_index=False).cooccurrence_events.sum()
theme_order=L.groupby("financing_theme").cooccurrence_events.sum().sort_values(ascending=False).index.tolist()
signal_order=["Challenge only","Potential only","Both"]

def layout(order,totals,gap,top=.95,bottom=.05):
    avail=(top-bottom)-gap*(len(order)-1); scale=avail/sum(totals[k] for k in order); pos={}; y=top
    for k in order:
        h=totals[k]*scale; pos[k]=(y-h,y); y-=h+gap
    return pos

lt=L.groupby("financing_theme").cooccurrence_events.sum().to_dict()
mt=L.groupby("uhc_outcome").cooccurrence_events.sum().to_dict()
rt=R.groupby("signal_class").cooccurrence_events.sum().to_dict()
pl=layout(theme_order,lt,.010); pm=layout(out_order,mt,.014); pr=layout(signal_order,rt,.03)
cycle=plt.rcParams["axes.prop_cycle"].by_key()["color"]
tc={k:cycle[i%len(cycle)] for i,k in enumerate(theme_order)}
oc={k:cycle[(i+2)%len(cycle)] for i,k in enumerate(out_order)}
sc={k:cycle[(i+5)%len(cycle)] for i,k in enumerate(signal_order)}
fig,ax=plt.subplots(figsize=(11,7.5)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
for k in theme_order:
    y0,y1=pl[k]; ax.add_patch(Rectangle((.045,y0),.020,y1-y0,facecolor=tc[k],alpha=.9)); ax.text(.038,(y0+y1)/2,k,ha="right",va="center",fontsize=7.8)
for k in out_order:
    y0,y1=pm[k]; ax.add_patch(Rectangle((.46,y0),.020,y1-y0,facecolor=oc[k],alpha=.9)); ax.text(.47,(y0+y1)/2,k,ha="center",va="center",fontsize=7.8,bbox=dict(boxstyle="round,pad=.18",fc="white",ec="none",alpha=.86))
for k in signal_order:
    y0,y1=pr[k]; ax.add_patch(Rectangle((.885,y0),.020,y1-y0,facecolor=sc[k],alpha=.9)); ax.text(.912,(y0+y1)/2,k,ha="left",va="center",fontsize=7.8)
def ribbon(x0,a0,a1,x1,b0,b1,color):
    c=.28*(x1-x0); verts=[(x0,a0),(x0+c,a0),(x1-c,b0),(x1,b0),(x1,b1),(x1-c,b1),(x0+c,a1),(x0,a1),(x0,a0)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=color,edgecolor="none",alpha=.20))
al={k:pl[k][0] for k in theme_order}; ami={k:pm[k][0] for k in out_order}
sl={k:(pl[k][1]-pl[k][0])/lt[k] for k in theme_order}; sm={k:(pm[k][1]-pm[k][0])/mt[k] for k in out_order}
for r in L.sort_values(["financing_theme","cooccurrence_events"],ascending=[True,False]).itertuples():
    hl=r.cooccurrence_events*sl[r.financing_theme]; hm=r.cooccurrence_events*sm[r.uhc_outcome]
    a0,a1=al[r.financing_theme],al[r.financing_theme]+hl; b0,b1=ami[r.uhc_outcome],ami[r.uhc_outcome]+hm
    ribbon(.065,a0,a1,.46,b0,b1,tc[r.financing_theme]); al[r.financing_theme]=a1; ami[r.uhc_outcome]=b1
amo={k:pm[k][0] for k in out_order}; ar={k:pr[k][0] for k in signal_order}; sr={k:(pr[k][1]-pr[k][0])/rt[k] for k in signal_order}
for r in R.sort_values(["uhc_outcome","cooccurrence_events"],ascending=[True,False]).itertuples():
    hm=r.cooccurrence_events*sm[r.uhc_outcome]; hr=r.cooccurrence_events*sr[r.signal_class]
    a0,a1=amo[r.uhc_outcome],amo[r.uhc_outcome]+hm; b0,b1=ar[r.signal_class],ar[r.signal_class]+hr
    ribbon(.48,a0,a1,.885,b0,b1,oc[r.uhc_outcome]); amo[r.uhc_outcome]=a1; ar[r.signal_class]=b1
ax.text(.055,.985,"Financing theme",ha="center",va="bottom",weight="bold",fontsize=10)
ax.text(.47,.985,"UHC outcome domain",ha="center",va="bottom",weight="bold",fontsize=10)
ax.text(.895,.985,"Abstract signal class",ha="center",va="bottom",weight="bold",fontsize=10)
fig.tight_layout(); savefig(fig,"Figure_7_Health_Financing_Data_Driven_Alluvial")

# Figure 8 term network/density
compiled=compile_map(TERM_PATTERNS)
presence={t:pub.text.str.contains(p,regex=True).to_numpy() for t,p in compiled.items()}
freq={t:int(m.sum()) for t,m in presence.items()}
terms=[t for t,n in freq.items() if n>=20]
G=nx.Graph(); [G.add_node(t,frequency=freq[t]) for t in terms]
edge_rows=[]
for i,a in enumerate(terms):
    for b in terms[i+1:]:
        n=int(np.logical_and(presence[a],presence[b]).sum())
        if n>=12:
            assoc=n/math.sqrt(freq[a]*freq[b]); edge_rows.append({"term1":a,"term2":b,"cooccurrences":n,"association_strength":assoc})
            if assoc>=.06:G.add_edge(a,b,weight=assoc,cooccurrences=n)
pd.DataFrame([{"term":t,"document_frequency":freq[t]} for t in terms]).sort_values("document_frequency",ascending=False).to_csv(DERIVED/"Figure_8_Term_Frequencies.csv",index=False)
pd.DataFrame(edge_rows).sort_values(["association_strength","cooccurrences"],ascending=False).to_csv(DERIVED/"Figure_8_Term_Cooccurrence_Edges.csv",index=False)
G.remove_nodes_from(list(nx.isolates(G)))
pos=nx.spring_layout(G,seed=SEED,weight="weight",iterations=550,k=1.45/math.sqrt(max(1,len(G))))
comms=list(nx.algorithms.community.greedy_modularity_communities(G,weight="weight")); ci={n:i for i,c in enumerate(comms) for n in c}
nodes=list(G); weights=np.array([G.nodes[n]["frequency"] for n in nodes],float); sizes=60+1900*(weights/weights.max())**.72
nc=[cycle[ci[n]%len(cycle)] for n in nodes]
fig,axs=plt.subplots(1,2,figsize=(13,6.5))
ax=axs[0]; ev=np.array([G[u][v]["weight"] for u,v in G.edges()]); em=ev.max() if len(ev) else 1
nx.draw_networkx_edges(G,pos,ax=ax,width=[.2+1.8*G[u][v]["weight"]/em for u,v in G.edges()],alpha=.16)
nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color=nc,alpha=.90,linewidths=.4,edgecolors="white")
labs=sorted(nodes,key=lambda n:G.nodes[n]["frequency"],reverse=True)[:10]
center=np.mean(np.array([pos[n] for n in nodes]),axis=0)
label_pos={}
for n in labs:
    v=np.array(pos[n])-center
    norm=np.linalg.norm(v) or 1.0
    label_pos[n]=np.array(pos[n])+0.065*v/norm
nx.draw_networkx_labels(G,label_pos,labels={n:n for n in labs},font_size=7.1,ax=ax,bbox=dict(boxstyle="round,pad=.08",fc="white",ec="none",alpha=.80))
ax.set_title("(a) Term co-occurrence network"); ax.axis("off")
ax=axs[1]; xy=np.array([pos[n] for n in nodes]); pad=.18; xmin,xmax=xy[:,0].min()-pad,xy[:,0].max()+pad; ymin,ymax=xy[:,1].min()-pad,xy[:,1].max()+pad
H,_,_=np.histogram2d(xy[:,0],xy[:,1],bins=350,range=[[xmin,xmax],[ymin,ymax]],weights=weights); Z=gaussian_filter(H.T,sigma=20)
ax.imshow(Z,origin="lower",extent=[xmin,xmax,ymin,ymax],aspect="auto",alpha=.92)
for n in labs[:8]:
    v=np.array(pos[n])-center; norm=np.linalg.norm(v) or 1.0; lp=np.array(pos[n])+0.050*v/norm
    ax.text(lp[0],lp[1],n,ha="center",va="center",fontsize=7.2,bbox=dict(boxstyle="round,pad=.08",fc="white",ec="none",alpha=.75))
ax.set_title("(b) Document-frequency density map"); ax.axis("off")
fig.tight_layout(); savefig(fig,"Figure_8_Health_Financing_Word_Density_and_Cooccurrence")

# Figure 9 chord
pmat=pairs.pivot(index="financing_theme",columns="uhc_outcome",values="records").fillna(0)
torder=pmat.sum(axis=1).sort_values(ascending=False).index.tolist()
oorder=pmat.sum(axis=0).sort_values(ascending=False).head(7).index.tolist()
Mfull=pmat.loc[torder,oorder].to_numpy(float); pd.DataFrame(Mfull,index=torder,columns=oorder).to_csv(DERIVED/"Figure_9_Chord_Matrix.csv")
M=Mfull.copy(); M[M<10]=0; keep=M.sum(axis=1)>0; torder=[t for t,k in zip(torder,keep) if k]; M=M[keep,:]
labels=torder+oorder; totals=np.r_[M.sum(axis=1),M.sum(axis=0)]; gap=.025; group=.16; unit=(2*np.pi-group-gap*len(labels))/totals.sum()
angles=[]; a=group/2
for i,total in enumerate(totals):
    s=a; e=a+total*unit; angles.append((s,e)); a=e+gap
    if i==len(torder)-1:a+=group
sub=[x[0] for x in angles]; seg={}
for i,t in enumerate(torder):
    for j,o in enumerate(oorder):
        v=M[i,j]
        if v<=0:continue
        s0=sub[i];s1=s0+v*unit;sub[i]=s1;oi=len(torder)+j;e0=sub[oi];e1=e0+v*unit;sub[oi]=e1;seg[(i,j)]=(s0,s1,e0,e1,v)
fig,ax=plt.subplots(figsize=(9,9),subplot_kw={"aspect":"equal"}); ax.axis("off"); r=1.; w=.075; ncol={i:cycle[i%len(cycle)] for i in range(len(labels))}
for i,(s,e) in enumerate(angles):
    ax.add_patch(Wedge((0,0),r,np.degrees(s),np.degrees(e),width=w,facecolor=ncol[i],alpha=.90))
    m=(s+e)/2;x,y=1.14*np.cos(m),1.14*np.sin(m);lab=labels[i].replace("/","/\n",1) if "/" in labels[i] and len(labels[i])>18 else labels[i]
    ax.text(x,y,lab,ha="left" if x>=0 else "right",va="center",fontsize=7.7)
for (i,j),(s0,s1,e0,e1,v) in sorted(seg.items(),key=lambda z:z[1][-1]):
    xy=lambda th:np.array([(r-w)*np.cos(th),(r-w)*np.sin(th)])
    A,B,C,D=xy(s0),xy(s1),xy(e0),xy(e1)
    verts=[tuple(A),tuple(A*.18),tuple(C*.18),tuple(C),tuple(D),tuple(D*.18),tuple(B*.18),tuple(B),tuple(A)]
    codes=[MplPath.MOVETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.LINETO,MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts,codes),facecolor=ncol[i],edgecolor="none",alpha=.18))
ax.set_xlim(-1.42,1.42);ax.set_ylim(-1.42,1.42);fig.tight_layout();savefig(fig,"Figure_9_Health_Financing_Theme_Outcome_Chord")

summary={
    "corpus_records":len(pub),
    "signal_class_counts":pub.signal_class.value_counts().to_dict(),
    "top_theme_outcome_pairs":pairs.sort_values("records",ascending=False).head(10).to_dict("records"),
    "figure9_visual_filter_min_pair_records":10,
    "note":"Descriptive title/abstract co-occurrence mapping; not effect estimation or final study-level evidence synthesis."}
summary_out = Path(os.environ.get("HPP_VIS_SUMMARY", str(ROOT / "outputs/data_driven_visualisation_summary.json"))).resolve(); summary_out.parent.mkdir(parents=True, exist_ok=True); summary_out.write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("Regenerated Figures 6-9 and underlying CSVs.")
