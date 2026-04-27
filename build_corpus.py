"""
build_corpus.py
===============
Run this ONCE before deploying to build a comprehensive PubMed corpus.
Covers the most common conditions a waiting room patient would present with.

Usage:
    python build_corpus.py

This saves processed_chunks.pkl to disk. Streamlit will load from cache
on every subsequent run — no re-fetching needed.

Estimated time: 10-15 minutes
Estimated cost: ~$0.50 in OpenAI embeddings (run once only)
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_and_chunk import chunks_from_pubmed, save_chunks

TOPICS = [

    # ── Cardiovascular ──────────────────────────────────────────────────
    "hypertension high blood pressure symptoms treatment",
    "chest pain causes diagnosis cardiac",
    "heart failure symptoms management treatment",
    "atrial fibrillation symptoms causes treatment",
    "coronary artery disease symptoms risk factors",
    "heart attack myocardial infarction symptoms",
    "stroke symptoms causes prevention",
    "peripheral artery disease symptoms treatment",
    "deep vein thrombosis pulmonary embolism symptoms",
    "high cholesterol hyperlipidemia treatment",

    # ── Metabolic / Endocrine ────────────────────────────────────────────
    "type 2 diabetes symptoms management treatment",
    "type 1 diabetes symptoms insulin management",
    "prediabetes prevention lifestyle intervention",
    "blood glucose hyperglycemia hypoglycemia management",
    "HbA1c diabetes monitoring targets",
    "hypothyroidism symptoms treatment levothyroxine",
    "hyperthyroidism symptoms treatment",
    "metabolic syndrome obesity treatment",
    "polycystic ovary syndrome PCOS symptoms treatment",
    "adrenal insufficiency symptoms treatment",

    # ── Respiratory ──────────────────────────────────────────────────────
    "asthma symptoms triggers management",
    "COPD chronic obstructive pulmonary disease symptoms",
    "shortness of breath dyspnea causes",
    "pneumonia symptoms treatment",
    "sleep apnea symptoms diagnosis treatment",
    "pulmonary hypertension symptoms treatment",
    "chronic cough causes diagnosis",
    "bronchitis symptoms treatment",

    # ── Gastrointestinal ─────────────────────────────────────────────────
    "acid reflux GERD symptoms treatment",
    "irritable bowel syndrome IBS symptoms management",
    "inflammatory bowel disease Crohn's colitis symptoms",
    "abdominal pain causes diagnosis",
    "constipation chronic causes treatment",
    "diarrhea chronic causes treatment",
    "fatty liver disease NAFLD symptoms",
    "gallstones symptoms treatment",
    "peptic ulcer symptoms treatment",
    "celiac disease symptoms diagnosis gluten",

    # ── Kidney / Urological ──────────────────────────────────────────────
    "chronic kidney disease symptoms stages management",
    "kidney stones symptoms treatment prevention",
    "urinary tract infection symptoms treatment",
    "benign prostatic hyperplasia BPH symptoms treatment",
    "urinary incontinence causes treatment",
    "proteinuria causes kidney function",

    # ── Musculoskeletal ──────────────────────────────────────────────────
    "osteoarthritis symptoms treatment management",
    "rheumatoid arthritis symptoms treatment",
    "back pain causes diagnosis treatment",
    "osteoporosis symptoms prevention treatment",
    "gout symptoms treatment uric acid",
    "fibromyalgia symptoms management",
    "lupus symptoms diagnosis treatment",

    # ── Neurological ─────────────────────────────────────────────────────
    "migraine headache symptoms treatment prevention",
    "tension headache causes treatment",
    "dizziness vertigo causes treatment",
    "neuropathy peripheral nerve symptoms treatment",
    "multiple sclerosis symptoms treatment",
    "Parkinson's disease symptoms treatment",
    "epilepsy seizure symptoms treatment",
    "memory loss cognitive decline dementia",

    # ── Mental Health ────────────────────────────────────────────────────
    "depression symptoms treatment antidepressants",
    "anxiety disorder symptoms treatment",
    "panic attacks symptoms treatment",
    "insomnia sleep disorders treatment",
    "ADHD attention deficit symptoms treatment adults",
    "PTSD symptoms treatment",
    "bipolar disorder symptoms treatment",

    # ── Women's Health ───────────────────────────────────────────────────
    "menopause symptoms management hormone therapy",
    "endometriosis symptoms treatment",
    "breast cancer screening symptoms risk factors",
    "cervical cancer screening prevention HPV",
    "ovarian cysts symptoms treatment",
    "pregnancy complications gestational diabetes",

    # ── Men's Health ─────────────────────────────────────────────────────
    "prostate cancer screening symptoms risk factors",
    "erectile dysfunction causes treatment",
    "testosterone deficiency symptoms treatment",

    # ── Infectious Disease ───────────────────────────────────────────────
    "influenza flu symptoms treatment prevention",
    "COVID-19 symptoms treatment long COVID",
    "pneumonia bacterial viral symptoms treatment",
    "skin infection cellulitis symptoms treatment",
    "Lyme disease symptoms diagnosis treatment",

    # ── Skin ─────────────────────────────────────────────────────────────
    "eczema atopic dermatitis symptoms treatment",
    "psoriasis symptoms treatment",
    "skin rash causes diagnosis",
    "acne treatment management",

    # ── Eyes / ENT ───────────────────────────────────────────────────────
    "glaucoma symptoms treatment eye pressure",
    "macular degeneration symptoms treatment",
    "hearing loss causes treatment",
    "sinusitis symptoms treatment",
    "tinnitus causes treatment",

    # ── Cancer (general) ─────────────────────────────────────────────────
    "cancer fatigue symptoms management",
    "cancer pain management treatment",
    "chemotherapy side effects management",

    # ── Preventive / General ─────────────────────────────────────────────
    "preventive care screenings recommendations adults",
    "vaccination immunization adults recommendations",
    "weight loss obesity management lifestyle",
    "nutrition diet chronic disease prevention",
    "exercise physical activity health benefits",
    "smoking cessation treatment options",
    "alcohol use disorder symptoms treatment",

]


def build(max_articles_per_topic: int = 10, cache_path: str = "processed_chunks.pkl"):
    print(f"Building corpus from {len(TOPICS)} topics...")
    print(f"Estimated articles: ~{len(TOPICS) * max_articles_per_topic}")
    print(f"Estimated time: 10-15 minutes\n")

    all_chunks = []
    failed = []

    for i, topic in enumerate(TOPICS):
        print(f"[{i+1}/{len(TOPICS)}] {topic}")
        try:
            _, chunks = chunks_from_pubmed(
                query=topic,
                max_articles=max_articles_per_topic,
                chunk_size=500,
                overlap=100,
            )
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks (total: {len(all_chunks)})")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(topic)
            time.sleep(1)  # back off on error

        # NCBI rate limit: be safe
        time.sleep(0.5)

    print(f"\n{'='*50}")
    print(f"✓ Corpus built: {len(all_chunks)} total chunks")
    print(f"✓ Topics succeeded: {len(TOPICS) - len(failed)}/{len(TOPICS)}")

    if failed:
        print(f"✗ Failed topics ({len(failed)}):")
        for t in failed:
            print(f"  - {t}")

    save_chunks(all_chunks, cache_path)
    print(f"\n✓ Saved to {cache_path}")
    print(f"  Upload this file to your deployment or run app.py to use it.")


if __name__ == "__main__":
    build(max_articles_per_topic=10)
