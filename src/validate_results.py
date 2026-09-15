#!/usr/bin/env python3
from pathlib import Path
import argparse, json, math
import pandas as pd

def close(a,b,tol=5e-4): return abs(float(a)-float(b)) <= tol

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument("--mode",choices=["frozen","live"],default="frozen"); ap.add_argument("--workdir",type=Path,default=None); args=ap.parse_args()
    repo=args.repo.resolve(); exp=json.loads((repo/"config/expected_results.json").read_text())
    errors=[]; checks=[]
    def chk(name,got,want,tol=None):
        ok=(got==want) if tol is None else close(got,want,tol); checks.append((name,got,want,ok));
        if not ok: errors.append(name)
    if args.mode=="frozen":
        screen=pd.read_csv(repo/"data/frozen/screening_decisions.csv")
        dups=pd.read_csv(repo/"data/manual/robust_duplicates_25.csv")
        corpus=pd.read_csv(repo/"data/frozen/systematic_map_corpus_5898.csv")
        summary=json.loads((repo/"data/frozen/analysis_summary.json").read_text())
        lead=pd.read_csv(repo/"data/frozen/core_analysis_tables/leadership_by_period.csv")
        logit=pd.read_csv(repo/"data/frozen/core_analysis_tables/leadership_logistic_models.csv")
        themes=pd.read_csv(repo/"data/frozen/core_analysis_tables/financing_theme_frequency.csv")
        sig=pd.read_csv(repo/"data/frozen/abstract_potential_challenge_signals.csv")
        chk("retrieved_records",len(screen),exp["retrieved_records"])
        chk("robust_duplicates_full_retrieval",len(dups),exp["robust_duplicates_full_retrieval"])
        chk("unique_records_after_robust_deduplication",len(screen)-len(dups),exp["unique_records_after_robust_deduplication"])
        chk("screen_eligible_before_deduplication",int(screen.screen_include.sum()),exp["screen_eligible_before_deduplication"])
        chk("duplicates_among_screen_eligible",int(screen.duplicate_removed.sum()),exp["duplicates_among_screen_eligible"])
        chk("final_systematic_map",len(corpus),exp["final_systematic_map"])
        for k,ek in [("authors","unique_authors"),("institutions","unique_institutions"),("total_citations","total_openalex_citations"),("median_citations","median_citations"),("works_with_studied_country","works_with_studied_country"),("reference_links","reference_links"),("unique_references","unique_references"),("reference_metadata_matched","reference_metadata_matched")]: chk(k,summary[k],exp[ek])
        for k,ek in [("mean_citations","mean_citations"),("mean_fwci","mean_fwci"),("open_access_share","open_access_share"),("international_collaboration_share","international_collaboration_share"),("country_attention_gini","country_attention_gini"),("top10_attention_share","top10_attention_share"),("cagr_2000_2025","cagr_2000_2025")]: chk(k,summary[k],exp[ek],1e-6)
        first=lead.set_index("period"); chk("first_authorship_1990_2004",first.loc["1990-2004","first_target_share"],exp["first_authorship_1990_2004"],1e-9); chk("first_authorship_2021_2026",first.loc["2021-2026","first_target_share"],exp["first_authorship_2021_2026"],1e-9)
        prim=logit[logit.corpus=="Primary"].set_index("outcome"); chk("first_authorship_or_per_decade",prim.loc["Target-country first authorship","OR_per_decade"],exp["first_authorship_or_per_decade"],1e-9); chk("corresponding_authorship_or_per_decade",prim.loc["Target-country corresponding authorship","OR_per_decade"],exp["corresponding_authorship_or_per_decade"],1e-9)
        expected_themes={"Insurance/prepayment":3983,"Financial protection/OOP":1761,"User fees/cost sharing":491,"Revenue/public financing":435,"Performance-based financing":342,"Willingness to pay":205,"Purchasing/provider payment":155,"Pooling":78,"External financing":61}
        tm=themes.set_index("theme");
        for k,v in expected_themes.items(): chk("theme:"+k,int(tm.loc[k,"papers"]),v)
        expected_sig={"Insurance/prepayment":(20.1,31.4,9.5),"Financial protection/OOP":(29.1,58.0,18.7),"User fees/cost sharing":(27.5,82.1,22.6),"Revenue/public financing":(10.8,30.6,5.1),"Performance-based financing":(19.0,18.4,5.6),"Willingness to pay":(18.5,35.6,6.8),"Purchasing/provider payment":(22.6,52.3,15.5),"Pooling":(33.3,62.8,23.1),"External financing":(4.9,31.1,3.3)}
        ss=sig.set_index("Financing theme")
        for k,(p,c,b) in expected_sig.items(): chk("signal potential:"+k,ss.loc[k,"Potential signal %"],p,0.05); chk("signal challenge:"+k,ss.loc[k,"Challenge signal %"],c,0.05); chk("signal both:"+k,ss.loc[k,"Both %"],b,0.05)
    else:
        work=(args.workdir or repo/"work").resolve(); summary=json.loads((work/"analysis/analysis_summary.json").read_text())
        for k in ["retrieved_records","primary_corpus","authors","institutions","total_citations","country_attention_gini","top10_attention_share"]:
            want=exp.get({"primary_corpus":"final_systematic_map"}.get(k,k),None); got=summary.get(k); checks.append((k,got,want, True if want is None else close(got,want,0.01*max(1,abs(float(want))))))
    for n,g,w,ok in checks: print(("PASS" if ok else "FAIL"),f"{n}: {g}  expected={w}")
    if errors: raise SystemExit(f"Validation failed: {len(errors)} checks")
    print(f"Validation passed: {len(checks)} checks.")

if __name__=="__main__": main()
