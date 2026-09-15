#!/usr/bin/env python3
from pathlib import Path
import argparse, os, shutil, subprocess, sys

def run(cmd, env=None):
    print("\n$", " ".join(map(str,cmd)))
    subprocess.run(cmd, check=True, env=env)

def main():
    ap=argparse.ArgumentParser(description="Reproduce the HPP UHC-financing systematic map")
    ap.add_argument("--mode",choices=["frozen","live"],default="frozen")
    ap.add_argument("--force-download",action="store_true")
    args=ap.parse_args(); repo=Path(__file__).resolve().parent; py=sys.executable
    if args.mode=="frozen":
        out=repo/"outputs/frozen"; out.mkdir(parents=True,exist_ok=True)
        run([py,repo/"src/audit_diagnostic.py","--repo",repo,"--out",out/"audit_diagnostic.json"])
        run([py,repo/"src/core_figures_from_frozen.py","--repo",repo,"--out",out/"figures"])
        env=os.environ.copy(); env.update({"HPP_REPO_ROOT":str(repo),"HPP_CORPUS":str(repo/"data/frozen/systematic_map_corpus_5898.csv"),"HPP_THEME_ASSIGN":str(repo/"data/frozen/financing_theme_assignments.csv"),"HPP_FIGDIR":str(out/"figures"),"HPP_DERIVED":str(out/"derived"),"HPP_SIGNAL_OUT":str(out/"abstract_potential_challenge_signals.csv"),"HPP_TIFF_DPI":"300","HPP_VIS_SUMMARY":str(out/"data_driven_visualisation_summary.json")})
        run([py,repo/"src/knowledge_maps.py"],env=env)
        run([py,repo/"src/validate_results.py","--repo",repo,"--mode","frozen"])
        print("\nFrozen reproduction complete. See outputs/frozen/.")
    else:
        work=repo/"work"; (work/"data").mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/"config/country_classification_FY2027.csv",work/"data/country_classification_FY2027.csv")
        cmd=[py,repo/"src/openalex_download.py","--repo",repo,"--workdir",work]
        if args.force_download: cmd.append("--force")
        run(cmd)
        env=os.environ.copy(); env["HPP_WORKDIR"]=str(work)
        run([py,repo/"src/core_analysis_live.py"],env=env)
        out=repo/"outputs/live"; out.mkdir(parents=True,exist_ok=True)
        env.update({"HPP_REPO_ROOT":str(repo),"HPP_CORPUS":str(work/"analysis/tables/publications.csv"),"HPP_THEME_ASSIGN":str(work/"analysis/tables/financing_theme_assignments.csv"),"HPP_FIGDIR":str(out/"figures"),"HPP_DERIVED":str(out/"derived"),"HPP_SIGNAL_OUT":str(out/"abstract_potential_challenge_signals.csv"),"HPP_TIFF_DPI":"300","HPP_VIS_SUMMARY":str(out/"data_driven_visualisation_summary.json")})
        run([py,repo/"src/knowledge_maps.py"],env=env)
        run([py,repo/"src/audit_diagnostic.py","--repo",repo,"--out",out/"audit_diagnostic.json"])
        run([py,repo/"src/validate_results.py","--repo",repo,"--mode","live","--workdir",work])
        print("\nLive reproduction complete. OpenAlex is a changing index, so live counts may differ from the frozen 2026 submission snapshot.")

if __name__=="__main__": main()
