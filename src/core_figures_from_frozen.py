#!/usr/bin/env python3
"""Regenerate main Figures 1-5 from archived derived tables.

The numerical content is frozen; small visual differences can arise from local
fonts and matplotlib versions.
"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,out,stem):
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/f'{stem}.png',dpi=300,bbox_inches='tight')
    fig.savefig(out/f'{stem}.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--out',type=Path,default=None); a=ap.parse_args()
    r=a.repo.resolve(); out=(a.out or r/'outputs/frozen/figures').resolve(); tab=r/'data/frozen/core_analysis_tables'

    # Figure 1: final systematic-map flow
    fig,ax=plt.subplots(figsize=(9,5.8)); ax.axis('off')
    boxes=[(.10,.57,'IDENTIFICATION\n14,646 records\nOpenAlex'),(.34,.57,'DEDUPLICATION\n14,621 unique\n25 duplicates flagged'),(.60,.57,'SCREENING + AUDIT\n8,723 not retained\n350-record manual audit'),(.86,.57,'FINAL MAP\n5,898 records\n72 economies')]
    for x,y,t in boxes: ax.text(x,y,t,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.65',fc='white',ec='black',lw=1.1),transform=ax.transAxes)
    for x0,x1 in [(.19,.25),(.45,.50),(.71,.76)]: ax.annotate('',xy=(x1,.57),xytext=(x0,.57),arrowprops=dict(arrowstyle='->',lw=1.3),xycoords=ax.transAxes)
    ax.text(.5,.88,'Identification and record-level selection flow for the systematic mapping review',ha='center',fontsize=13,fontweight='bold',transform=ax.transAxes)
    ax.text(.5,.20,'The fixed audit characterized screening behavior; the final corpus is interpreted as conservative rather than exhaustive.',ha='center',fontsize=9,transform=ax.transAxes)
    save(fig,out,'Figure_1_Health_Financing_Systematic_Mapping_Review_Flow')

    # Figure 2
    annual=pd.read_csv(tab/'annual_scientific_production.csv')
    fig,ax=plt.subplots(figsize=(9,5.3)); ax.plot(annual.year,annual.publications,marker='o',markersize=2.5); ax.set(xlabel='Publication year',ylabel='Publications',title='Annual development of the health-financing knowledge base, 1990–2026'); ax.grid(alpha=.2); ax.axvline(2025,ls='--',lw=1); ax.text(2025.1,max(annual.publications)*.7,'2026 partial indexing year',fontsize=8); save(fig,out,'Figure_2_Health_Financing_Knowledge_Development')

    # Figure 3
    lead=pd.read_csv(tab/'leadership_by_period.csv'); x=np.arange(len(lead))
    fig,ax=plt.subplots(figsize=(9,5.3)); ax.plot(x,100*lead.first_target_share,marker='o',label='Target-country first author'); ax.plot(x,100*lead.corr_target_share,marker='s',label='Target-country corresponding author'); ax.plot(x,100*lead.international_collaboration_share,marker='^',label='International collaboration'); ax.set_xticks(x,lead.period); ax.set(ylabel='Share of publications (%)',xlabel='Publication period',title='Research leadership and international collaboration by publication period'); ax.grid(alpha=.2); ax.legend(frameon=False); save(fig,out,'Figure_3_Health_Financing_Research_Leadership_Trends')

    # Figure 4
    att=pd.read_csv(tab/'studied_country_attention.csv').head(20).sort_values('study_mentions')
    fig,ax=plt.subplots(figsize=(9,7)); sizes=60+500*att.local_first_author_share_country_specific.fillna(0); ax.scatter(att.study_mentions,np.arange(len(att)),s=sizes,alpha=.72); ax.set_yticks(np.arange(len(att)),att.country); ax.set(xlabel='Publications mentioning country in title/abstract',title='Geographic research attention and local first-author leadership'); ax.grid(axis='x',alpha=.2); ax.text(.98,.02,'Bubble size = local first-author share\nin country-specific records',ha='right',va='bottom',transform=ax.transAxes,fontsize=8); save(fig,out,'Figure_4_Health_Financing_Geographic_Attention_and_Leadership')

    # Figure 5
    tp=pd.read_csv(tab/'financing_theme_by_period.csv'); periods=['1990-2004','2005-2010','2011-2015','2016-2020','2021-2026']; piv=tp.pivot(index='period',columns='theme',values='share').fillna(0).reindex(periods)
    fig,ax=plt.subplots(figsize=(10,6));
    for c in piv.columns: ax.plot(range(len(piv)),100*piv[c],marker='o',label=c)
    ax.set_xticks(range(len(piv)),piv.index); ax.set(xlabel='Publication period',ylabel='Share of mapped records (%)',title='Evolution of health-financing research themes'); ax.grid(alpha=.2); ax.legend(ncol=2,fontsize=8,frameon=False); save(fig,out,'Figure_5_Health_Financing_Theme_Evolution')
    print('Regenerated Figures 1-5 from frozen analytical tables.')

if __name__=='__main__': main()
