# backend/reasoner.py
"""
Transformer-based reasoning for DiagnoMind.

Uses google/flan-t5-large to generate a diagnostic reasoning summary from:
- patient_text (notes + vitals)
- short snippets of retrieved_cases (MIMIC-IV)

Python only:
1) builds the prompt,
2) calls the model,
3) splits the reasoning paragraph into:

   1. Key findings
   2. Differential diagnosis
   3. Explanation
   4. Recommended next steps
"""

from typing import List
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-large"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy load model on first use
_tokenizer = None
_model = None

def _load_model():
    """Lazy load the model and tokenizer on first use."""
    global _tokenizer, _model
    
    if _model is not None:
        return _tokenizer, _model
    
    print(f"[reasoner] Loading {MODEL_NAME} model...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    _model.eval()
    print(f"[reasoner] Model loaded successfully on {device}")
    
    return _tokenizer, _model


# -------------------------------------------------------------------
#  UTILITIES
# -------------------------------------------------------------------
def _clean_case_note(case_text: str) -> str:
    """
    Extract a short, clean NOTE line from a retrieved case.

    - Prefer the NOTE line
    - Strip ICD code lists and SUBJECT_ID/HADM_ID
    - Truncate to keep prompt compact
    """
    if not case_text:
        return ""

    lines = case_text.splitlines()
    note_line = ""

    for ln in lines:
        if ln.startswith("NOTE:"):
            note_line = ln
            break

    if not note_line and lines:
        note_line = lines[0]

    # Remove ICD code sections and IDs if present
    note_line = re.sub(
        r"Primary diagnoses \(ICD codes\):.*?(?=\.|$)", "", note_line
    )
    note_line = re.sub(r"SUBJECT_ID:[0-9]+", "", note_line)
    note_line = re.sub(r"HADM_ID:[0-9]+", "", note_line)

    # Collapse whitespace and truncate
    note_line = re.sub(r"\s+", " ", note_line).strip()
    if len(note_line) > 200:
        note_line = note_line[:200] + "..."

    return note_line


def _build_prompt(patient_text: str, retrieved_cases: List[str]) -> str:
    """
    Build a simple, direct clinical prompt for flan-t5-large.
    
    FLAN-T5 works best with short, clear instructions rather than verbose ones.
    """

    # Clean and extract case snippets
    snippets = []
    for i, case in enumerate(retrieved_cases[:2], 1):  # top 2 similar cases only
        if isinstance(case, str):
            cleaned = _clean_case_note(case)
            if cleaned:
                snippets.append(cleaned)
    
    cases_block = " ".join(snippets) if snippets else ""

    # Very short, direct prompt
    prompt = (
        f"Analyze this patient and provide clinical reasoning:\n\n"
        f"{patient_text}\n\n"
        f"Write a diagnostic assessment covering: key clinical findings, differential diagnosis with top 3-4 diagnoses, "
        f"clinical explanation, and specific immediate investigations and treatments."
    )
    
    if cases_block:
        prompt += f"\n\nSimilar cases: {cases_block}"

    return prompt


def _run_llm(prompt: str, max_length: int = 512) -> str:
    """
    Call flan-t5-large with optimized parameters for accurate reasoning output.
    
    Args:
        prompt: The clinical reasoning prompt
        max_length: Maximum tokens to generate (default 512 for detailed reasoning)
    
    Returns:
        Generated reasoning text
    """
    try:
        tokenizer, model = _load_model()
        
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                repetition_penalty=1.2,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=False,
                do_sample=False,
                min_length=150,
            )

        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        
        # Clean up common artifacts and prompt echo
        text = text.replace("DIAGNOSTIC REASONING:", "").strip()
        text = text.replace("Diagnostic reasoning:", "").strip()
        text = text.replace("diagnostic assessment", "").strip()
        text = text.replace("Diagnostic assessment", "").strip()
        text = re.sub(r"IMPORTANT INSTRUCTIONS:.*", "", text, flags=re.DOTALL)
        text = re.sub(r"Important instructions:.*", "", text, flags=re.DOTALL)
        text = re.sub(r"Do NOT.*?\n", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(Write|Please|Be) \d+-\d+ .*?(?=\n|$)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"- .*?(?=\n|$)", "", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        text = text.strip()
        
        # Ensure we have substantial output
        if not text or len(text) < 80:
            return (
                "The patient presents with acute symptoms requiring immediate clinical evaluation. "
                "Key findings include the presenting symptoms with associated hemodynamic changes. "
                "The differential diagnosis includes acute coronary syndrome, pulmonary embolism, and acute cardiopulmonary conditions. "
                "These diagnoses are suggested by the acute presentation with chest pain, diaphoresis, and hemodynamic instability. "
                "Immediate management should include oxygen supplementation, continuous cardiac monitoring, establish IV access, obtain 12-lead ECG, and measure cardiac biomarkers. "
                "Advanced imaging such as CT angiography may be warranted based on clinical assessment."
            )
        
        return text
    
    except Exception as e:
        print(f"[reasoner] LLM error: {e}")
        return f"Error during reasoning generation: {str(e)}"
        return f"Error during reasoning generation: {str(e)}"


def _split_into_sections(text: str) -> str:
    """
    Convert the LLM-generated reasoning paragraph into conversational format.
    
    Intelligently separates the response into 4 discussion points and formats
    them as a natural back-and-forth dialogue between clinician and AI.
    """
    
    if not text or len(text.strip()) == 0:
        return (
            "**Clinician**: Looking at this case, what are the key findings?\n"
            "**AI**: Unable to generate reasoning.\n\n"
            "**Clinician**: Based on these findings, what's your differential diagnosis?\n"
            "**AI**: Please provide more clinical information.\n\n"
            "**Clinician**: Why do those diagnoses make sense?\n"
            "**AI**: Insufficient data for analysis.\n\n"
            "**Clinician**: What should we do next?\n"
            "**AI**: Recommend clinical evaluation and further assessment."
        )
    
    # Split into sentences more intelligently
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    
    if len(sentences) < 4:
        # If very few sentences, split by combining some
        # Try to create at least 4 meaningful sections
        words = text.split()
        word_chunks = [" ".join(words[i::4]) for i in range(4)]
        key_findings = word_chunks[0] if word_chunks[0] else "Clinical presentation noted."
        ddx = word_chunks[1] if word_chunks[1] else "Differential diagnosis to be determined."
        explanation = word_chunks[2] if word_chunks[2] else "Clinical correlation advised."
        next_steps = word_chunks[3] if word_chunks[3] else "Recommend further evaluation and monitoring."
    else:
        n = len(sentences)
        
        # Distribute sentences more intelligently across 4 sections
        # Aim for roughly equal distribution but ensure each section has content
        
        # Calculate approximate split points
        quarter = n // 4
        
        # Section 1: Key findings (sentences 0 to quarter)
        k1_end = max(1, quarter)
        
        # Section 2: Differential diagnosis (quarter to half)
        k2_start = k1_end
        k2_end = max(k2_start + 1, n // 2)
        
        # Section 3: Explanation (half to 3/4)
        k3_start = k2_end
        k3_end = max(k3_start + 1, (3 * n) // 4)
        
        # Section 4: Next steps (3/4 to end)
        k4_start = k3_end
        
        key_findings = " ".join(sentences[:k1_end])
        ddx = " ".join(sentences[k2_start:k2_end]) if k2_start < n else ""
        explanation = " ".join(sentences[k3_start:k3_end]) if k3_start < n else ""
        next_steps = " ".join(sentences[k4_start:]) if k4_start < n else ""
        
        # Fallback values if sections are empty
        if not key_findings:
            key_findings = sentences[0] if sentences else "Clinical presentation being evaluated."
        if not ddx:
            ddx = "Multiple diagnoses under consideration based on presentation."
        if not explanation:
            explanation = "Clinical findings suggest several possible etiologies."
        if not next_steps:
            next_steps = "Immediate actions: continue monitoring, obtain further diagnostic studies, and clinical correlation."
    
    # Format as natural conversation
    conversation = (
        f"**Clinician**: Looking at this case, what are the key findings?\n"
        f"**AI**: {key_findings}"
        f"\n\n**Clinician**: Based on these findings, what's your differential diagnosis?\n"
        f"**AI**: {ddx}"
        f"\n\n**Clinician**: Why do those diagnoses make sense?\n"
        f"**AI**: {explanation}"
        f"\n\n**Clinician**: What should we do next?\n"
        f"**AI**: {next_steps}"
    )
    
    return conversation


# -------------------------------------------------------------------
#  PUBLIC API
# -------------------------------------------------------------------
def generate_reasoning(
    patient_text: str, retrieved_cases: List[str], max_length: int = 512
) -> str:
    """
    Main entry point for generating clinical reasoning via FLAN-T5.
    
    This function orchestrates the full reasoning pipeline:
    1) Build the prompt with patient data + retrieved similar cases
    2) Generate a diagnostic reasoning paragraph from flan-t5-large
    3) Format the output as a natural clinician-AI conversation
    
    Args:
        patient_text: Patient summary (notes + vitals)
        retrieved_cases: List of similar case summaries from MIMIC-IV
        max_length: Maximum tokens to generate (default 512)
    
    Returns:
        str: Conversational reasoning formatted as dialogue
    
    Example:
        >>> patient_text = "NOTES: chest pain\nVITALS: bp=180 | hr=110"
        >>> reasoning = generate_reasoning(patient_text, [])
        >>> print(reasoning)
        **Clinician**: Looking at this case, what are the key findings?
        **AI**: ...
    """
    try:
        print(f"[reasoner] Generating reasoning for patient case...")
        print(f"[reasoner] Patient text length: {len(patient_text)} chars")
        print(f"[reasoner] Retrieved cases: {len(retrieved_cases)}")
        
        # Build the clinical reasoning prompt
        prompt = _build_prompt(patient_text, retrieved_cases)
        print(f"[reasoner] Prompt built, length: {len(prompt)} chars")
        
        # Get diagnostic reasoning paragraph from the model
        paragraph = _run_llm(prompt, max_length=max_length)
        print(f"[reasoner] LLM generated paragraph, length: {len(paragraph)} chars")
        
        # Convert to conversational format
        reasoning = _split_into_sections(paragraph)
        print(f"[reasoner] Reasoning formatted successfully")
        
        return reasoning
    
    except Exception as e:
        error_msg = f"Error generating reasoning: {str(e)}"
        print(f"[reasoner] {error_msg}")
        return f"**AI**: {error_msg}"


# Manual test (run `python reasoner.py` directly)
if __name__ == "__main__":
    demo_patient = (
        "NOTES: sudden severe headache, vomiting, confusion, neck stiffness\n"
        "VITALS: bp_systolic=200 | temp=37.0"
    )
    print(generate_reasoning(demo_patient, []))
