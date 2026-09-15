"""Reproduce descriptive abstract-level potential/challenge signal mapping.

This script is a supplementary knowledge-mapping analysis. It does not perform
study-level effect estimation or critical appraisal and must not be interpreted
as a substitute for full-text systematic evidence synthesis.
"""
import re
import pandas as pd

INPUT = "../Supplementary/Supplementary_Data_S2_Publications.csv"  # if supplied/reconstructed
OUTPUT = "../Supplementary/Supplementary_Table_S1_Abstract_Potential_Challenge_Signals.csv"

THEMES = {
    "Insurance/prepayment": [r"\bhealth insurance\b", r"\bsocial health insurance\b", r"\bnational health insurance\b", r"\bcommunity[- ]based health insurance\b", r"\bprepayment\b"],
    "Financial protection/OOP": [r"\bfinancial (?:risk )?protection\b", r"\bcatastrophic (?:health|medical) (?:expenditure|spending|payment|cost)", r"\bout[- ]of[- ]pocket\b", r"\bimpoverish"],
    "Revenue/public financing": [r"\bgovernment health expenditure\b", r"\bpublic health expenditure\b", r"\btax[- ]?(?:based|funded)\b", r"\bfiscal space\b", r"\bdomestic resource mobili[sz]ation\b", r"\bpublic financ"],
    "Pooling": [r"\brisk pooling\b", r"\bpooling (?:of )?(?:health )?funds\b", r"\bfragmented (?:pool|pooling)\b", r"\brisk pool"],
    "Purchasing/provider payment": [r"\bstrategic (?:health )?purchasing\b", r"\bprovider payment\b", r"\bprovider reimbursement\b", r"\bhealth(?:care| care) purchasing\b"],
    "Performance-based financing": [r"\bresults?[- ]based financing\b", r"\bperformance[- ]based financing\b", r"\bpay[- ]for[- ]performance\b", r"\bpayment[- ]for[- ]performance\b"],
    "User fees/cost sharing": [r"\buser fees?\b", r"\bcost[- ]sharing\b", r"\bco[- ]?payments?\b", r"\bcopayments?\b", r"\bcoinsurance\b"],
    "External financing": [r"\bdevelopment assistance for health\b", r"\bexternal health financing\b", r"\bdonor financing\b", r"\bforeign aid\b", r"\bdevelopment assistance\b"],
}
POTENTIAL = {
    "Coverage/access potential": [r"\bexpand(?:ed|ing)? (?:population )?coverage\b", r"\bincrease(?:d|s|ing)? (?:healthcare |health care )?(?:access|utili[sz]ation|coverage)\b", r"\bimprov(?:e|ed|es|ing) (?:access|utili[sz]ation|coverage)\b"],
    "Financial-protection potential": [r"\breduc(?:e|ed|es|ing) (?:out[- ]of[- ]pocket|catastrophic|financial burden)", r"\bimprov(?:e|ed|es|ing) financial protection\b", r"\bprotect(?:s|ed|ing)? households?\b"],
    "Equity potential": [r"\bimprov(?:e|ed|es|ing) equit", r"\bmore equitable\b", r"\bprogressive financ", r"\bsubsid(?:y|ies|ized|ised|ization|isation)"],
    "Resource/pooling potential": [r"\bresource mobili[sz]ation\b", r"\brisk sharing\b", r"\bcross[- ]subsid", r"\blarger risk pool", r"\bconsolidat(?:e|ed|ion) (?:risk )?pool"],
    "Efficiency/quality potential": [r"\bimprov(?:e|ed|es|ing) efficien", r"\bimprov(?:e|ed|es|ing) quality\b", r"\bbetter value\b"],
}
CHALLENGES = {
    "Affordability/OOP challenge": [r"\bhigh out[- ]of[- ]pocket\b", r"\bcatastrophic (?:health|medical) (?:expenditure|spending|payment|cost)", r"\bfinancial hardship\b", r"\bfinancial barrier", r"\buser fee"],
    "Coverage/enrolment challenge": [r"\blow (?:insurance )?(?:coverage|enrol+ment)\b", r"\bcoverage gaps?\b", r"\bunder[- ]?enrol", r"\bnon[- ]?enrol", r"\bvoluntary enrol"],
    "Informality/selection challenge": [r"\binformal sector\b", r"\binformal workers?\b", r"\badverse selection\b", r"\brisk selection\b"],
    "Fragmentation/pooling challenge": [r"\bfragment(?:ed|ation)\b", r"\bsmall (?:risk )?pools?\b", r"\bmultiple (?:risk )?pools?\b"],
    "Fiscal/sustainability challenge": [r"\blimited fiscal space\b", r"\bconstrained fiscal space\b", r"\binsufficient (?:public )?fund", r"\bunderfund", r"\bfinancial sustainability\b", r"\bfiscal sustainability\b", r"\bunsustain"],
    "Governance/capacity challenge": [r"\bweak governance\b", r"\badministrative (?:capacity|cost|burden|challenge)", r"\bimplementation (?:challenge|barrier|constraint)", r"\bcorruption\b", r"\bleakage\b"],
    "Donor-dependence challenge": [r"\bdonor depend", r"\baid depend", r"\bexternal depend", r"\bvolatile (?:aid|funding|financing)", r"\bdeclining aid\b"],
    "Supply/provider challenge": [r"\bservice readiness\b", r"\bprovider (?:shortage|behavio|response|incentive)", r"\bsupply[- ]side (?:constraint|barrier)"],
    "Inequity/regressivity challenge": [r"\binequit", r"\bregressive\b", r"\bdisparit", r"\bexclusion\b"],
}

def any_match(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)

df = pd.read_csv(INPUT)
df["text"] = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.lower()
for name, pats in THEMES.items():
    df[name] = df["text"].map(lambda x: any_match(x, pats))
for name, pats in POTENTIAL.items():
    df[name] = df["text"].map(lambda x: any_match(x, pats))
for name, pats in CHALLENGES.items():
    df[name] = df["text"].map(lambda x: any_match(x, pats))

rows = []
for theme in THEMES:
    sub = df[df[theme]]
    potential = sub[list(POTENTIAL)].any(axis=1)
    challenge = sub[list(CHALLENGES)].any(axis=1)
    rows.append({
        "Financing theme": theme,
        "Records": len(sub),
        "Any potential signal": int(potential.sum()),
        "Any challenge signal": int(challenge.sum()),
        "Both potential and challenge": int((potential & challenge).sum()),
        "Potential signal %": round(100 * potential.mean(), 1) if len(sub) else 0,
        "Challenge signal %": round(100 * challenge.mean(), 1) if len(sub) else 0,
        "Both %": round(100 * (potential & challenge).mean(), 1) if len(sub) else 0,
    })

pd.DataFrame(rows).sort_values("Records", ascending=False).to_csv(OUTPUT, index=False)
print(f"Wrote {OUTPUT}")
