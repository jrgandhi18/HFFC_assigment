from __future__ import annotations

import re
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

FAQ_URL = "https://homefirstindia.com/faqs"


@dataclass
class FAQDoc:
    question: str
    answer: str
    source: str = FAQ_URL


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fetch_homefirst_faqs(limit: int = 30) -> list[FAQDoc]:
    docs: list[FAQDoc] = []
    try:
        res = requests.get(FAQ_URL, timeout=25)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # The page renders many FAQ titles in h3 tags.
        for h3 in soup.find_all("h3"):
            q = _normalize(h3.get_text(" ", strip=True))
            if not q or "?" not in q:
                continue

            answer = "Please refer to HomeFirst FAQ for exact policy details."
            sibling = h3.find_next_sibling()
            if sibling:
                candidate = _normalize(sibling.get_text(" ", strip=True))
                if candidate and len(candidate) > 20:
                    answer = candidate

            docs.append(FAQDoc(question=q, answer=answer))
            if len(docs) >= limit:
                break
    except Exception:
        docs = []

    if docs:
        return docs

    # Fallback corpus if website structure changes.
    return [
        FAQDoc("What are the documents required for home loan?", "Commonly required documents include KYC, income proof, bank statements, and property papers."),
        FAQDoc("What percentage of property price can be financed?", "Prototype policy assumes up to 80% Loan-to-Value (LTV), subject to underwriting."),
        FAQDoc("Can I apply jointly with spouse?", "Joint applications with eligible co-applicants are typically allowed and can improve loan eligibility."),
        FAQDoc("Is prepayment allowed?", "Home loan prepayment is generally allowed; charges depend on loan type and lender policy."),
        FAQDoc("What is tenure for home loan?", "Tenure varies by profile and lender policy; longer tenure reduces EMI but increases total interest."),
        FAQDoc("I have low credit score, can I still apply?", "Applications with low score may still be considered with additional checks, collateral, and repayment capacity."),
        FAQDoc("How is eligibility calculated?", "Eligibility usually depends on monthly income, existing obligations, FOIR, LTV, and applicant profile."),
    ]


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 1}


def retrieve_faq_context(query: str, docs: list[FAQDoc], top_k: int = 3) -> list[FAQDoc]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return docs[:top_k]

    scored: list[tuple[float, FAQDoc]] = []
    for d in docs:
        d_tokens = _tokenize(f"{d.question} {d.answer}")
        overlap = len(q_tokens & d_tokens)
        denom = max(len(q_tokens | d_tokens), 1)
        score = overlap / denom
        if overlap > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]] or docs[:top_k]
