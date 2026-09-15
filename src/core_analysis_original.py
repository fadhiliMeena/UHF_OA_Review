import gzip, json, re, unicodedata, math, os, textwrap, hashlib
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import pycountry
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm

ROOT=Path('/mnt/data/HPP_COMPLETE')
DATA=ROOT/'data'
OUT=ROOT/'analysis'
TABLES=OUT/'tables'
FIGS=OUT/'figures'
NETS=OUT/'networks'
for d in (OUT,TABLES,FIGS,NETS): d.mkdir(parents=True,exist_ok=True)
FULL=DATA/'LLMIC_health_financing_OpenAlex_FULL_WORKS.jsonl.gz'
REFS=DATA/'LLMIC_health_financing_CITED_REFERENCES.jsonl.gz'
COUNTRIES=pd.read_csv(DATA/'country_classification_FY2027.csv')
TARGET=set(COUNTRIES.country)
INCOME=dict(zip(COUNTRIES.country,COUNTRIES.income_group))

# ---------- helpers ----------
def reconstruct(inv):
    if not inv: return ''
    p=[]
    for word, poss in inv.items():
        for pos in poss: p.append((pos,word))
    return ' '.join(w for _,w in sorted(p))

def norm_title(s):
    s=unicodedata.normalize('NFKD',s or '').lower()
    s=re.sub(r'[^a-z0-9]+',' ',s)
    return re.sub(r'\s+',' ',s).strip()

def short_id(x):
    return (x or '').split('/')[-1]

def journal_source(w):
    # Prefer a true journal among all locations.
    locs=w.get('locations') or []
    for loc in locs:
        src=(loc or {}).get('source') or {}
        if src.get('type')=='journal' and src.get('display_name'):
            return src
    src=((w.get('primary_location') or {}).get('source') or {})
    return src

def iso_name(code):
    if not code: return None
    special={'XK':'Kosovo','PS':'Palestine','TW':'Taiwan'}
    if code in special: return special[code]
    try: return pycountry.countries.get(alpha_2=code).name
    except Exception: return code

# ---------- country aliases for studied-country identification ----------
aliases={c:[c] for c in TARGET}
aliases.update({
'Democratic Republic of the Congo':['Democratic Republic of the Congo','Democratic Republic of Congo','DR Congo','Congo-Kinshasa','DRC'],
'Republic of the Congo':['Republic of the Congo','Republic of Congo','Congo-Brazzaville'],
'North Korea':['North Korea',"Democratic People's Republic of Korea",'Korea, Democratic People\'s Republic','DPRK'],
'The Gambia':['The Gambia','Gambia'],
"Côte d'Ivoire":["Côte d'Ivoire","Cote d'Ivoire",'Ivory Coast'],
'Egypt':['Egypt','Arab Republic of Egypt'],
'Kyrgyzstan':['Kyrgyzstan','Kyrgyz Republic'],
'Laos':['Laos','Lao PDR',"Lao People's Democratic Republic"],
'São Tomé and Príncipe':['São Tomé and Príncipe','Sao Tome and Principe'],
'Syria':['Syria','Syrian Arab Republic'],
'Tanzania':['Tanzania','United Republic of Tanzania'],
'Timor-Leste':['Timor-Leste','Timor Leste','East Timor'],
'Venezuela':['Venezuela','Bolivarian Republic of Venezuela'],
'West Bank and Gaza':['West Bank and Gaza','West Bank','Gaza','Palestine','Palestinian territories','occupied Palestinian territory'],
'Yemen':['Yemen','Republic of Yemen'],
})
# Explicitly add terms vulnerable to overlap.
aliases['Papua New Guinea']=['Papua New Guinea']
aliases['Guinea-Bissau']=['Guinea-Bissau','Guinea Bissau']

alias_entries=[]
for country,als in aliases.items():
    for a in als:
        # robust boundaries for spaces/hyphens/apostrophes
        pat=re.compile(r'(?<![A-Za-z])'+re.escape(a)+r'(?![A-Za-z])',re.I)
        alias_entries.append((len(a),country,a,pat))
alias_entries.sort(reverse=True,key=lambda x:x[0])

def studied_countries(text):
    text=text or ''
    matches=[]
    occupied=[]
    for _,country,a,pat in alias_entries:
        for m in pat.finditer(text):
            span=(m.start(),m.end())
            if any(not (span[1]<=s or span[0]>=e) for s,e in occupied):
                continue
            matches.append((span[0],span[1],country,a))
            occupied.append(span)
    return sorted(set(x[2] for x in matches))

# ---------- financing relevance screen ----------
strong_patterns={
'health_financing':[r'\bhealth(?:care| care)? financ(?:e|ing|ed|es)\b',r'\bfinanc(?:e|ing) (?:of )?health(?: care|care)?\b',r'\bhealth system financ'],
'insurance_prepayment':[r'\bhealth insurance\b',r'\bsocial health insurance\b',r'\bnational health insurance\b',r'\bcommunity[- ]based health insurance\b',r'\bmutual health insurance\b',r'\bmicro[- ]?health insurance\b',r'\bmedical insurance\b',r'\bhealth prepayment\b',r'\bprepayment (?:for|of) health'],
'financial_protection':[r'\bfinancial (?:risk )?protection\b',r'\bcatastrophic (?:health|medical) (?:expenditure|spending|payment|cost)\b',r'\bout[- ]of[- ]pocket (?:health|medical|healthcare|health care)? ?(?:expenditure|spending|payment|cost|expense)s?\b',r'\bimpoverish(?:ing|ment).*?(?:health|medical)',r'\bmedical impoverishment\b'],
'pooling':[r'\b(?:health )?risk pooling\b',r'\bpooling (?:of )?(?:health )?funds\b',r'\bhealth financing pool'],
'purchasing_payment':[r'\bstrategic (?:health )?purchasing\b',r'\bprovider payment\b',r'\bprovider reimbursement\b',r'\bhealth(?:care| care) purchasing\b',r'\bresults?[- ]based financing\b',r'\bperformance[- ]based financing\b',r'\bpay[- ]for[- ]performance\b',r'\bpayment[- ]for[- ]performance\b'],
'public_revenue':[r'\bgovernment health expenditure\b',r'\bpublic health expenditure\b',r'\bdomestic health expenditure\b',r'\bhousehold health expenditure\b',r'\bcurrent health expenditure\b',r'\bnational health accounts?\b',r'\bhealth accounts?\b',r'\btax[- ]based health financ',r'\btax[- ]funded health',r'\bfiscal space for health\b',r'\bdomestic resource mobili[sz]ation for health\b',r'\bdevelopment assistance for health\b',r'\bexternal health financing\b',r'\bdonor financing for health\b'],
'cost_sharing':[r'\buser fees?\b',r'\bcost[- ]sharing\b',r'\bco[- ]?payments?\b',r'\bcopayments?\b',r'\bcoinsurance\b']
}
compiled={k:[re.compile(p,re.I) for p in v] for k,v in strong_patterns.items()}
specific_meta=re.compile(r'^(health insurance|national health insurance|social health insurance|community(?:-based)? health insurance|health economics|health spending|health care financing|healthcare financing|health financing|financial protection|catastrophic health expenditure|out[- ]of[- ]pocket|user fee|public health expenditure|government health expenditure|public finance|strategic purchasing|purchasing|provider payment|reimbursement|performance[- ]based financing|results[- ]based financing|risk pooling|social insurance)$',re.I)
artifact_title=re.compile(r'^(cover|editorial note|alternative language abstract|strobe statement|lecture no\.?|correction|erratum|retraction)',re.I)

def finance_evidence(w):
    title=(w.get('title') or '').lower()
    abstract=reconstruct(w.get('abstract_inverted_index')).lower()
    cats_title=[]; cats_abs=[]
    for cat,pats in compiled.items():
        if any(p.search(title) for p in pats): cats_title.append(cat)
        if any(p.search(abstract) for p in pats): cats_abs.append(cat)
    kws=[(k.get('display_name') or '').strip() for k in (w.get('keywords') or [])]
    topics=[(t.get('display_name') or '').strip() for t in (w.get('topics') or [])]
    spec=any(specific_meta.search(x) for x in kws+topics if x)
    return title,abstract,cats_title,cats_abs,spec

def inclusion(w):
    title,abstract,cats_title,cats_abs,spec=finance_evidence(w)
    if w.get('is_paratext') or artifact_title.search(title): return False,'paratext/artifact'
    if cats_title: return True,'financing term in title'
    if cats_abs and spec: return True,'financing term in abstract + specific OpenAlex finance keyword/topic'
    return False,'not centrally financing-focused'

# ---------- load and screen ----------
allworks=[]
with gzip.open(FULL,'rt',encoding='utf-8') as f:
    for line in f: allworks.append(json.loads(line))

screen_rows=[]; prelim=[]
for w in allworks:
    flag,reason=inclusion(w)
    screen_rows.append({'openalex_id':short_id(w.get('id')),'title':w.get('title'),'year':w.get('publication_year'),'include':flag,'reason':reason})
    if flag: prelim.append(w)

# dedupe on DOI, then normalized title-year
seen=set(); primary=[]; duplicate_ids=[]
for w in prelim:
    doi=(w.get('doi') or '').strip().lower()
    key=('doi',doi) if doi else ('ty',norm_title(w.get('title')),w.get('publication_year'))
    if key in seen:
        duplicate_ids.append(short_id(w.get('id')))
    else:
        seen.add(key); primary.append(w)

pd.DataFrame(screen_rows).to_csv(TABLES/'screening_decisions.csv',index=False)
with gzip.open(OUT/'primary_corpus.jsonl.gz','wt',encoding='utf-8') as f:
    for w in primary: f.write(json.dumps(w,ensure_ascii=False)+'\n')

# strict title-term sensitivity corpus, after same dedupe
strict=[]; seen2=set()
for w in allworks:
    if w.get('is_paratext') or artifact_title.search((w.get('title') or '').lower()): continue
    _,_,cats_title,_,_=finance_evidence(w)
    if not cats_title: continue
    doi=(w.get('doi') or '').strip().lower(); key=('doi',doi) if doi else ('ty',norm_title(w.get('title')),w.get('publication_year'))
    if key not in seen2: seen2.add(key); strict.append(w)

# ---------- structured rows ----------
pubrows=[]
work_by_id={short_id(w.get('id')):w for w in primary}
for w in primary:
    wid=short_id(w.get('id')); src=journal_source(w)
    abstract=reconstruct(w.get('abstract_inverted_index'))
    pubrows.append({
        'work_id':wid,'doi':w.get('doi'),'title':w.get('title'),'year':w.get('publication_year'),'date':w.get('publication_date'),
        'type':w.get('type'),'language':w.get('language'),'citations':w.get('cited_by_count') or 0,'fwci':w.get('fwci'),
        'journal':src.get('display_name'),'source_type':src.get('type'),'source_id':short_id(src.get('id')),
        'is_oa':(w.get('open_access') or {}).get('is_oa'),'oa_status':(w.get('open_access') or {}).get('oa_status'),
        'abstract':abstract,'ref_count':len(w.get('referenced_works') or []),'country_count':w.get('countries_distinct_count'),
        'institution_count':w.get('institutions_distinct_count')
    })
pubs=pd.DataFrame(pubrows)
pubs.to_csv(TABLES/'publications.csv',index=False)

# ---------- authorships, affiliations ----------
authrows=[]; instrows=[]; country_work=defaultdict(set); work_country_sets={}
first_info={}; corr_info={}

def target_country_from_raw(auth):
    txt='; '.join(auth.get('raw_affiliation_strings') or [])
    sc=studied_countries(txt)
    return sc[0] if len(sc)==1 else None

def author_country_codes(auth):
    # structured OpenAlex codes; also use raw target-country string as override/addition
    codes=set(auth.get('countries') or [])
    for inst in auth.get('institutions') or []:
        if inst.get('country_code'): codes.add(inst['country_code'])
    raw_target=target_country_from_raw(auth)
    names={iso_name(c) for c in codes if c}
    if raw_target: names.add(raw_target)
    return sorted(x for x in names if x)

for w in primary:
    wid=short_id(w.get('id')); year=w.get('publication_year')
    cset=set()
    first=[]; corr=[]
    for a in w.get('authorships') or []:
        author=a.get('author') or {}; aid=short_id(author.get('id')); an=author.get('display_name') or a.get('raw_author_name')
        names=author_country_codes(a); cset.update(names)
        if a.get('author_position')=='first': first.extend(names)
        if a.get('is_corresponding'): corr.extend(names)
        authrows.append({'work_id':wid,'year':year,'author_id':aid,'author':an,'position':a.get('author_position'),'corresponding':bool(a.get('is_corresponding')),'countries':'|'.join(names)})
        for inst in a.get('institutions') or []:
            iid=short_id(inst.get('id'))
            instrows.append({'work_id':wid,'year':year,'author_id':aid,'institution_id':iid,'institution':inst.get('display_name'),'country_code':inst.get('country_code'),'country':iso_name(inst.get('country_code')),'institution_type':inst.get('type')})
    work_country_sets[wid]=cset
    for c in cset: country_work[c].add(wid)
    first_info[wid]=sorted(set(first)); corr_info[wid]=sorted(set(corr))

auth=pd.DataFrame(authrows); inst=pd.DataFrame(instrows)
auth.to_csv(TABLES/'authorships.csv',index=False); inst.to_csv(TABLES/'affiliations.csv',index=False)

# ---------- studied countries ----------
studyrows=[]; studied_by_work={}
for w in primary:
    wid=short_id(w.get('id')); txt=(w.get('title') or '')+' '+reconstruct(w.get('abstract_inverted_index'))
    sc=studied_countries(txt); studied_by_work[wid]=sc
    for c in sc: studyrows.append({'work_id':wid,'year':w.get('publication_year'),'country':c,'income_group':INCOME.get(c)})
study=pd.DataFrame(studyrows); study.to_csv(TABLES/'studied_countries.csv',index=False)

# ---------- annual production ----------
annual=pubs.groupby('year').agg(publications=('work_id','nunique'),citations=('citations','sum'),mean_citations=('citations','mean'),mean_fwci=('fwci','mean')).reset_index().sort_values('year')
annual.to_csv(TABLES/'annual_scientific_production.csv',index=False)

# ---------- leadership ----------
periods=[(1990,2004,'1990-2004'),(2005,2010,'2005-2010'),(2011,2015,'2011-2015'),(2016,2020,'2016-2020'),(2021,2026,'2021-2026')]
def period_label(y):
    for lo,hi,l in periods:
        if lo<=y<=hi:return l
    return None

def target_lead(names): return bool(set(names)&TARGET)
def intl(cset): return len(cset)>=2
leadrows=[]
for w in primary:
    wid=short_id(w.get('id')); y=w.get('publication_year'); cset=work_country_sets[wid]
    first=first_info[wid]; corr=corr_info[wid]
    leadrows.append({'work_id':wid,'year':y,'period':period_label(y),'first_country_known':bool(first),'first_target':target_lead(first) if first else np.nan,'corr_country_known':bool(corr),'corr_target':target_lead(corr) if corr else np.nan,'international_collaboration':intl(cset),'n_author_countries':len(cset),'studied_count':len(studied_by_work[wid])})
lead=pd.DataFrame(leadrows); lead.to_csv(TABLES/'leadership_work_level.csv',index=False)
period_stats=[]
for p,g in lead.groupby('period',sort=False):
    period_stats.append({'period':p,'publications':len(g),'first_known_n':g.first_country_known.sum(),'first_target_share':g.loc[g.first_country_known,'first_target'].astype(float).mean(),'corr_known_n':g.corr_country_known.sum(),'corr_target_share':g.loc[g.corr_country_known,'corr_target'].astype(float).mean(),'international_collaboration_share':g.international_collaboration.mean()})
lead_period=pd.DataFrame(period_stats); lead_period['period']=pd.Categorical(lead_period.period,[x[2] for x in periods],ordered=True);lead_period=lead_period.sort_values('period');lead_period.to_csv(TABLES/'leadership_by_period.csv',index=False)

# logistic trend per decade
def logit_trend(df,outcome,known):
    d=df[df[known]].dropna(subset=[outcome]).copy();d['decade']=(d.year-1990)/10
    X=sm.add_constant(d[['decade']]); model=sm.GLM(d[outcome].astype(float),X,family=sm.families.Binomial()).fit()
    b=model.params['decade']; se=model.bse['decade']; return {'n':len(d),'OR_per_decade':math.exp(b),'CI_low':math.exp(b-1.96*se),'CI_high':math.exp(b+1.96*se),'p_value':model.pvalues['decade']}
reg=pd.DataFrame([{'outcome':'Target-country first authorship',**logit_trend(lead,'first_target','first_country_known')},{'outcome':'Target-country corresponding authorship',**logit_trend(lead,'corr_target','corr_country_known')}])
# strict sensitivity leadership
strict_ids=set(short_id(w.get('id')) for w in strict)
lead_strict=lead[lead.work_id.isin(strict_ids)]
reg_s=pd.DataFrame([{'outcome':'Target-country first authorship',**logit_trend(lead_strict,'first_target','first_country_known')},{'outcome':'Target-country corresponding authorship',**logit_trend(lead_strict,'corr_target','corr_country_known')}]);reg['corpus']='Primary';reg_s['corpus']='Strict title-term sensitivity';pd.concat([reg,reg_s]).to_csv(TABLES/'leadership_logistic_models.csv',index=False)

# ---------- local leadership by studied country ----------
att=[]
for c in sorted(TARGET):
    ids=set(study.loc[study.country==c,'work_id'])
    single=[wid for wid in ids if studied_by_work[wid]==[c]]
    known=[wid for wid in single if first_info[wid]]
    local=[wid for wid in known if c in first_info[wid]]
    att.append({'country':c,'income_group':INCOME[c],'study_mentions':len(ids),'country_specific_papers':len(single),'country_specific_first_country_known':len(known),'local_first_author_n':len(local),'local_first_author_share_country_specific':len(local)/len(known) if known else np.nan})
attention=pd.DataFrame(att).sort_values('study_mentions',ascending=False);attention.to_csv(TABLES/'studied_country_attention.csv',index=False)
vals=attention.study_mentions.to_numpy(float)
def gini(x):
    x=np.array(x,float); x=x[x>=0];
    if x.sum()==0:return np.nan
    x=np.sort(x); n=len(x); return (2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n
country_ineq=pd.DataFrame([{'indicator':'Country-attention Gini','value':gini(vals)},{'indicator':'Top-10 share of target-country mentions','value':attention.head(10).study_mentions.sum()/attention.study_mentions.sum()},{'indicator':'Economies with <=10 mentions','value':(attention.study_mentions<=10).sum()},{'indicator':'Economies with zero mentions','value':(attention.study_mentions==0).sum()}]);country_ineq.to_csv(TABLES/'country_attention_inequality.csv',index=False)

# ---------- financing themes ----------
theme_patterns={
'Insurance/prepayment':compiled['insurance_prepayment'],
'Financial protection/OOP':compiled['financial_protection'],
'Revenue/public financing':compiled['public_revenue'],
'Pooling':compiled['pooling'],
'Purchasing/provider payment':[re.compile(x,re.I) for x in [r'\bstrategic (?:health )?purchasing\b',r'\bprovider payment\b',r'\bprovider reimbursement\b',r'\bhealth(?:care| care) purchasing\b']],
'Performance-based financing':[re.compile(x,re.I) for x in [r'\bresults?[- ]based financing\b',r'\bperformance[- ]based financing\b',r'\bpay[- ]for[- ]performance\b',r'\bpayment[- ]for[- ]performance\b']],
'User fees/cost sharing':compiled['cost_sharing'],
'External financing':[re.compile(x,re.I) for x in [r'\bdevelopment assistance for health\b',r'\bexternal health financing\b',r'\bdonor financing for health\b',r'\bforeign aid\b',r'\bdevelopment assistance\b']],
'Willingness to pay':[re.compile(r'\bwillingness to pay\b',re.I),re.compile(r'\bwillingness-to-pay\b',re.I)]
}
themerows=[]
for w in primary:
    wid=short_id(w.get('id')); text=((w.get('title') or '')+' '+reconstruct(w.get('abstract_inverted_index'))).lower()
    for th,pats in theme_patterns.items():
        if any(p.search(text) for p in pats): themerows.append({'work_id':wid,'year':w.get('publication_year'),'period':period_label(w.get('publication_year')),'theme':th})
themes=pd.DataFrame(themerows).drop_duplicates(); themes.to_csv(TABLES/'financing_theme_assignments.csv',index=False)
themefreq=themes.groupby('theme').work_id.nunique().reset_index(name='papers');themefreq['share']=themefreq.papers/len(primary);themefreq=themefreq.sort_values('papers',ascending=False);themefreq.to_csv(TABLES/'financing_theme_frequency.csv',index=False)
themep=themes.groupby(['period','theme']).work_id.nunique().reset_index(name='papers'); periodden=lead.groupby('period').work_id.nunique().to_dict();themep['share']=themep.apply(lambda r:r.papers/periodden.get(r.period,np.nan),axis=1);themep.to_csv(TABLES/'financing_theme_by_period.csv',index=False)

# ---------- journals / Bradford ----------
src=pubs[pubs.journal.notna()].groupby('journal').agg(publications=('work_id','nunique'),citations=('citations','sum'),mean_citations=('citations','mean'),mean_fwci=('fwci','mean')).reset_index().sort_values(['publications','citations'],ascending=False)
src.to_csv(TABLES/'source_performance.csv',index=False)
# Bradford zones based cumulative thirds of articles
src2=src.copy(); total=src2.publications.sum(); src2['cum']=src2.publications.cumsum();src2['zone']=np.where(src2['cum']<=total/3,1,np.where(src2['cum']<=2*total/3,2,3));src2.to_csv(TABLES/'bradford_sources.csv',index=False)

# ---------- author and institution performance ----------
auth_unique=auth.drop_duplicates(['work_id','author_id'])
auth_perf=auth_unique.groupby(['author_id','author']).agg(publications=('work_id','nunique')).reset_index()
# corpus h-index
citation_map=dict(zip(pubs.work_id,pubs.citations))
def hindex(ids):
    cs=sorted((citation_map.get(x,0) for x in set(ids)),reverse=True); return max([i for i,c in enumerate(cs,1) if c>=i],default=0)
hs=auth_unique.groupby(['author_id','author']).work_id.apply(list).reset_index(name='works');hs['corpus_h_index']=hs.works.apply(hindex);auth_perf=auth_perf.merge(hs[['author_id','author','corpus_h_index']],on=['author_id','author']).sort_values(['publications','corpus_h_index'],ascending=False);auth_perf.to_csv(TABLES/'author_performance.csv',index=False)
instu=inst.dropna(subset=['institution']).drop_duplicates(['work_id','institution_id']);inst_perf=instu.groupby(['institution_id','institution','country']).agg(publications=('work_id','nunique')).reset_index().sort_values('publications',ascending=False);inst_perf.to_csv(TABLES/'institution_performance.csv',index=False)

# ---------- country productivity and collaboration ----------
cp=[]
for c,ids in country_work.items(): cp.append({'country':c,'full_count':len(ids),'target_income_group':INCOME.get(c,'Other economy')})
country_prod=pd.DataFrame(cp).sort_values('full_count',ascending=False);country_prod.to_csv(TABLES/'country_productivity.csv',index=False)
edgec=Counter()
for wid,cset in work_country_sets.items():
    cs=sorted(cset)
    for a,b in combinations(cs,2): edgec[(a,b)]+=1
country_edges=pd.DataFrame([{'country1':a,'country2':b,'collaborations':w} for (a,b),w in edgec.items()]).sort_values('collaborations',ascending=False);country_edges.to_csv(TABLES/'country_collaboration_edges.csv',index=False)

# ---------- keywords ----------
stop=set(x.lower() for x in ['Health care','Environmental health','Public health','Humans','Human','Adult','Female','Male','Article','Research','Developing country','Developing countries','Patient','Patients','Study','Healthcare','Medicine'])
kwrows=[]
for w in primary:
    wid=short_id(w.get('id'))
    for k in w.get('keywords') or []:
        name=(k.get('display_name') or '').strip()
        if name and name.lower() not in stop: kwrows.append({'work_id':wid,'year':w.get('publication_year'),'keyword':name,'score':k.get('score')})
kw=pd.DataFrame(kwrows).drop_duplicates(['work_id','keyword']);kw.to_csv(TABLES/'keywords.csv',index=False)
kwfreq=kw.groupby('keyword').work_id.nunique().reset_index(name='occurrences').sort_values('occurrences',ascending=False);kwfreq.to_csv(TABLES/'keyword_frequency.csv',index=False)

# curated relevant keyword network: select top frequent terms that match broad policy concepts, plus top 60 by frequency after generic stop
policy_re=re.compile(r'(health|insur|financ|payment|expend|poverty|equity|revenue|subsid|fee|welfare|purchas|reimburs|income|coverage|capitation|risk|social|public|private|cost|access|quality|utili[sz]|maternal|rural|government|policy|system|facility|econom)',re.I)
sel=[]
for t in kwfreq.keyword:
    if len(sel)>=60:break
    if policy_re.search(t): sel.append(t)
kwsub=kw[kw.keyword.isin(sel)]
kwedge=Counter()
for wid,g in kwsub.groupby('work_id'):
    ks=sorted(set(g.keyword))
    for a,b in combinations(ks,2): kwedge[(a,b)]+=1
pd.DataFrame([{'term1':a,'term2':b,'cooccurrences':n} for (a,b),n in kwedge.items()]).sort_values('cooccurrences',ascending=False).to_csv(NETS/'keyword_cooccurrence_edges.csv',index=False)

# ---------- references: local frequencies from primary corpus ----------
refcount=Counter(); refsets={}
for w in primary:
    wid=short_id(w.get('id')); rs=[short_id(x) for x in (w.get('referenced_works') or []) if x];refsets[wid]=set(rs);refcount.update(rs)
ref_freq=pd.DataFrame([{'reference_id':r,'local_citations':n} for r,n in refcount.items()]).sort_values('local_citations',ascending=False);ref_freq.to_csv(TABLES/'primary_reference_frequency.csv',index=False)
needed=set(refcount)
# stream metadata only for cited refs in primary corpus
refmeta={}
with gzip.open(REFS,'rt',encoding='utf-8') as f:
    for line in f:
        r=json.loads(line); rid=short_id(r.get('id'))
        if rid in needed: refmeta[rid]=r

# top cited refs and RPYS
tr=[]; years=Counter()
for rid,n in refcount.items():
    r=refmeta.get(rid); 
    if not r: continue
    year=r.get('publication_year');
    if year: years[year]+=n
    src=journal_source(r)
    auths=r.get('authorships') or []
    first=(auths[0].get('author') or {}).get('display_name') if auths else None
    tr.append({'reference_id':rid,'local_citations':n,'year':year,'first_author':first,'title':r.get('title'),'journal':src.get('display_name'),'doi':r.get('doi'),'global_citations':r.get('cited_by_count')})
toprefs=pd.DataFrame(tr).sort_values('local_citations',ascending=False);toprefs.to_csv(TABLES/'top_locally_cited_references.csv',index=False)
rpys=pd.DataFrame([{'publication_year':y,'cited_reference_occurrences':n} for y,n in sorted(years.items())])
rpys['median5']=rpys.cited_reference_occurrences.rolling(5,center=True,min_periods=1).median();rpys['deviation_from_5yr_median']=rpys.cited_reference_occurrences-rpys.median5;rpys.to_csv(TABLES/'rpys.csv',index=False)

# ---------- co-citation top 60 ----------
top_ref_ids=set(toprefs.head(60).reference_id)
coc=Counter()
for rs in refsets.values():
    s=sorted(rs & top_ref_ids)
    for a,b in combinations(s,2): coc[(a,b)]+=1
pd.DataFrame([{'ref1':a,'ref2':b,'cocitations':n} for (a,b),n in coc.items()]).sort_values('cocitations',ascending=False).to_csv(NETS/'reference_cocitation_edges.csv',index=False)

# ---------- bibliographic coupling among top countries ----------
# Country reference sets: union references from works with an author in each top country; coupling = shared unique refs, cosine normalised.
topcountries=country_prod.head(30).country.tolist(); cref={c:set() for c in topcountries}
for wid,cset in work_country_sets.items():
    rs=refsets.get(wid,set())
    for c in cset:
        if c in cref: cref[c].update(rs)
bc=[]
for a,b in combinations(topcountries,2):
    inter=len(cref[a]&cref[b]); denom=math.sqrt(len(cref[a])*len(cref[b])) if cref[a] and cref[b] else 0
    if inter: bc.append({'country1':a,'country2':b,'shared_references':inter,'cosine_coupling':inter/denom if denom else 0})
pd.DataFrame(bc).sort_values('cosine_coupling',ascending=False).to_csv(NETS/'country_bibliographic_coupling.csv',index=False)

# ---------- summary ----------
y2025=annual.loc[annual.year==2025,'publications']; y2000=annual.loc[annual.year==2000,'publications']
cagr=((float(y2025.iloc[0])/float(y2000.iloc[0]))**(1/25)-1) if len(y2025) and len(y2000) and y2000.iloc[0]>0 else np.nan
summary={
'retrieved_records':len(allworks),'screened_in_before_dedup':len(prelim),'duplicates_removed':len(prelim)-len(primary),'primary_corpus':len(primary),'strict_sensitivity_corpus':len(strict),
'authors':auth.author_id.nunique(),'institutions':inst.institution_id.nunique(),'total_citations':int(pubs.citations.sum()),'mean_citations':float(pubs.citations.mean()),'median_citations':float(pubs.citations.median()),'mean_fwci':float(pubs.fwci.mean()),
'works_with_studied_country':int(study.work_id.nunique()),'country_attention_gini':float(country_ineq.iloc[0].value),'top10_attention_share':float(country_ineq.iloc[1].value),
'international_collaboration_share':float(lead.international_collaboration.mean()),'cagr_2000_2025':float(cagr),'reference_links':int(sum(refcount.values())),'unique_references':len(refcount),'reference_metadata_matched':len(refmeta)
}
(Path(OUT/'analysis_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')

# ---------- figures ----------
plt.rcParams.update({'font.size':10,'axes.titlesize':13,'axes.labelsize':11,'figure.dpi':140})
def savefig(fig,name):
    fig.savefig(FIGS/(name+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(FIGS/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)

# Fig1 flow
fig,ax=plt.subplots(figsize=(8,7));ax.axis('off')
boxes=[(.5,.86,f'OpenAlex records retrieved\nN = {len(allworks):,}'),(.5,.63,f'High-specificity financing relevance screen\nN = {len(prelim):,}'),(.5,.40,f'Duplicate DOI/title-year records removed\nN = {len(prelim)-len(primary):,}'),(.5,.17,f'Primary analytic corpus\nN = {len(primary):,}')]
for x,y,txt in boxes: ax.text(x,y,txt,ha='center',va='center',fontsize=13,bbox=dict(boxstyle='round,pad=.75',fc='white',ec='black',lw=1.1),transform=ax.transAxes)
for a,b in [(.79,.70),(.56,.47),(.33,.24)]: ax.annotate('',xy=(.5,b),xytext=(.5,a),arrowprops=dict(arrowstyle='->',lw=1.2),xycoords=ax.transAxes)
ax.set_title('Study identification and analytic corpus construction',pad=15)
savefig(fig,'Fig1_corpus_flow')

# Fig2 annual
fig,ax=plt.subplots(figsize=(9,5.5));ax.plot(annual.year,annual.publications,marker='o',markersize=3);ax.set(xlabel='Publication year',ylabel='Publications',title='Annual scientific production, 1990-2026');ax.grid(alpha=.2);ax.axvline(2025,ls='--',lw=1);ax.text(2025.15,max(annual.publications)*.72,'2026 incomplete',fontsize=9);savefig(fig,'Fig2_annual_production')

# Fig3 leadership
lp=lead_period.copy();fig,ax=plt.subplots(figsize=(9,5.5));x=np.arange(len(lp));ax.plot(x,100*lp.first_target_share,marker='o',label='Target-country first author');ax.plot(x,100*lp.corr_target_share,marker='s',label='Target-country corresponding author');ax.plot(x,100*lp.international_collaboration_share,marker='^',label='International collaboration');ax.set_xticks(x,lp.period.astype(str));ax.set(ylabel='Share of publications (%)',xlabel='Publication period',title='Research leadership and international collaboration');ax.set_ylim(0,90);ax.grid(alpha=.2);ax.legend(frameon=False);savefig(fig,'Fig3_leadership_trends')

# Fig4 attention local leadership
xdat=attention.head(20).sort_values('study_mentions');fig,ax=plt.subplots(figsize=(9,7));sizes=60+500*xdat.local_first_author_share_country_specific.fillna(0);ax.scatter(xdat.study_mentions,np.arange(len(xdat)),s=sizes,alpha=.72);ax.set_yticks(np.arange(len(xdat)),xdat.country);ax.set(xlabel='Publications mentioning the country in title/abstract',title='Research attention and local first-author leadership');ax.grid(axis='x',alpha=.2);ax.text(.98,.02,'Bubble size represents local first-author share\nin country-specific studies',ha='right',va='bottom',transform=ax.transAxes,fontsize=9);savefig(fig,'Fig4_country_attention_local_leadership')

# generic network/density helper
def paired_network_density(G,node_label,node_weight,title,name,top_labels=35,seed=2026):
    if len(G)==0:return
    comms=list(nx.algorithms.community.louvain_communities(G,weight='weight',seed=seed)) if G.number_of_edges() else [set(G.nodes())]
    cmap={n:i for i,c in enumerate(comms) for n in c}
    pos=nx.spring_layout(G,weight='weight',seed=seed,iterations=350,k=1.3/math.sqrt(max(1,len(G))))
    nodes=list(G.nodes()); weights=np.array([node_weight(n) for n in nodes],float); sizes=90+1900*(weights/max(weights.max(),1))**.72
    edgeweights=[G[u][v].get('weight',1) for u,v in G.edges()]; em=max(edgeweights) if edgeweights else 1
    fig,axs=plt.subplots(1,2,figsize=(16,8))
    ax=axs[0];nx.draw_networkx_edges(G,pos,ax=ax,width=[.15+1.5*G[u][v].get('weight',1)/em for u,v in G.edges()],alpha=.22,edge_color='gray');nx.draw_networkx_nodes(G,pos,ax=ax,node_size=sizes,node_color=[cmap[n] for n in nodes],cmap=plt.cm.tab20,alpha=.88,edgecolors='white',linewidths=.4)
    lab=sorted(nodes,key=node_weight,reverse=True)[:top_labels];nx.draw_networkx_labels(G,pos,labels={n:node_label(n) for n in lab},font_size=8,ax=ax);ax.axis('off');ax.set_title('(a) Network visualization')
    xy=np.array([pos[n] for n in nodes]);pad=.15;xmin,xmax=xy[:,0].min()-pad,xy[:,0].max()+pad;ymin,ymax=xy[:,1].min()-pad,xy[:,1].max()+pad;H,_,_=np.histogram2d(xy[:,0],xy[:,1],bins=320,range=[[xmin,xmax],[ymin,ymax]],weights=weights);Z=gaussian_filter(H.T,sigma=18);ax=axs[1];ax.imshow(Z,origin='lower',extent=[xmin,xmax,ymin,ymax],aspect='auto',cmap='YlOrRd',alpha=.92);[ax.text(pos[n][0],pos[n][1],node_label(n),ha='center',va='center',fontsize=8) for n in lab];ax.axis('off');ax.set_title('(b) Density visualization');fig.suptitle(title,fontsize=14,y=.99);savefig(fig,name)

# keyword network graph
freqdict=dict(zip(kwfreq.keyword,kwfreq.occurrences));G=nx.Graph()
for t in sel:
    if freqdict.get(t,0)>=12:G.add_node(t,weight=freqdict[t])
for (a,b),n in kwedge.items():
    if a in G and b in G and n>=4:G.add_edge(a,b,weight=n)
G.remove_nodes_from(list(nx.isolates(G)));paired_network_density(G,lambda n:n,lambda n:G.nodes[n]['weight'],'Conceptual structure of health-financing research','Fig5_keyword_network_density',38)

# theme evolution
piv=themep.pivot(index='period',columns='theme',values='share').fillna(0).reindex([p[2] for p in periods]);fig,ax=plt.subplots(figsize=(10,6));
for c in piv.columns:ax.plot(range(len(piv)),100*piv[c],marker='o',label=c)
ax.set_xticks(range(len(piv)),piv.index);ax.set(xlabel='Publication period',ylabel='Share of publications (%)',title='Evolution of health-financing research themes');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=8,frameon=False);savefig(fig,'Fig6_financing_theme_evolution')

# country network density supplement
cpw=dict(zip(country_prod.country,country_prod.full_count));top=set(country_prod.head(45).country);CG=nx.Graph();
for c in top:CG.add_node(c,weight=cpw[c])
for (a,b),n in edgec.items():
    if a in top and b in top and n>=3:CG.add_edge(a,b,weight=n)
CG.remove_nodes_from(list(nx.isolates(CG)));paired_network_density(CG,lambda n:n,lambda n:CG.nodes[n]['weight'],'International collaboration structure','SuppFig1_country_collaboration_network_density',32)

# co-citation network supplement
labmeta={r.reference_id:r for r in toprefs.itertuples()};RG=nx.Graph();
for rid in top_ref_ids:
    rr=labmeta.get(rid);RG.add_node(rid,weight=(rr.local_citations if rr else refcount[rid]))
for (a,b),n in coc.items():
    if n>=4:RG.add_edge(a,b,weight=n)
RG.remove_nodes_from(list(nx.isolates(RG)))
def reflabel(rid):
    r=labmeta.get(rid); return f"{(r.first_author or 'Anon').split()[-1]} {int(r.year) if pd.notna(r.year) else ''}" if r else rid
paired_network_density(RG,reflabel,lambda n:RG.nodes[n]['weight'],'Reference co-citation structure','SuppFig2_reference_cocitation_network_density',30)

# RPYS supplement
r=rpys[(rpys.publication_year>=1950)&(rpys.publication_year<=2026)];fig,ax=plt.subplots(figsize=(10,5.5));ax.plot(r.publication_year,r.cited_reference_occurrences);peaks=r.nlargest(10,'deviation_from_5yr_median');ax.scatter(peaks.publication_year,peaks.cited_reference_occurrences,s=25);[ax.text(x.publication_year,x.cited_reference_occurrences,str(int(x.publication_year)),fontsize=8,ha='center',va='bottom') for x in peaks.itertuples()];ax.set(xlabel='Reference publication year',ylabel='Cited-reference occurrences',title='Reference publication year spectroscopy');ax.grid(alpha=.2);savefig(fig,'SuppFig3_RPYS')

# top sources supplement
s=src.head(20).sort_values('publications');fig,ax=plt.subplots(figsize=(9,7));ax.barh(s.journal,s.publications);ax.set(xlabel='Publications',title='Twenty most productive publication sources');savefig(fig,'SuppFig4_top_sources')

# tables for manuscript
summary_table=pd.DataFrame([
['Retrieved OpenAlex records',len(allworks)],['Screened-in before deduplication',len(prelim)],['Duplicate records removed',len(prelim)-len(primary)],['Primary analytic corpus',len(primary)],['Strict title-term sensitivity corpus',len(strict)],['Unique authors',auth.author_id.nunique()],['Unique institutions',inst.institution_id.nunique()],['Total OpenAlex citations',int(pubs.citations.sum())],['Median citations per publication',float(pubs.citations.median())],['Mean FWCI',round(float(pubs.fwci.mean()),2)],['Publications with a target country identified',int(study.work_id.nunique())],['International collaboration (%)',round(100*lead.international_collaboration.mean(),1)],['Country-attention Gini',round(gini(vals),3)],['Reference links in analytic corpus',sum(refcount.values())],['Unique cited works in analytic corpus',len(refcount)]],columns=['Indicator','Value']);summary_table.to_csv(TABLES/'Table1_corpus_characteristics.csv',index=False)

# main result tables
lead_period.to_csv(TABLES/'Table2_leadership_by_period.csv',index=False)
attention.head(20).to_csv(TABLES/'Table3_top_country_attention.csv',index=False)
themefreq.to_csv(TABLES/'Table4_financing_themes.csv',index=False)
toprefs.head(25).to_csv(TABLES/'SuppTable_top_references.csv',index=False)
src.head(25).to_csv(TABLES/'SuppTable_top_sources.csv',index=False)
auth_perf.head(25).to_csv(TABLES/'SuppTable_top_authors.csv',index=False)
inst_perf.head(25).to_csv(TABLES/'SuppTable_top_institutions.csv',index=False)
country_prod.head(30).to_csv(TABLES/'SuppTable_country_productivity.csv',index=False)
country_edges.head(50).to_csv(TABLES/'SuppTable_country_collaboration_edges.csv',index=False)

print(json.dumps(summary,indent=2))
print('Top themes:\n',themefreq.head(12).to_string(index=False))
print('Leadership periods:\n',lead_period.to_string(index=False))
print('Regression:\n',pd.concat([reg,reg_s]).to_string(index=False))
print('Top attention:\n',attention.head(15).to_string(index=False))
print('Top refs:\n',toprefs.head(12)[['local_citations','year','first_author','title','journal','doi']].to_string(index=False))
