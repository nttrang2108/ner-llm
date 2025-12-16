"""Prompt templates for Vietnamese NER extraction

Optimized based on VLSP 2018 dataset patterns and reference implementation.
Uses English instructions for better LLM understanding.
"""

import json
from typing import List, Dict


# -------------------------------------------------------
# SHARED CONFIGURATION
# -------------------------------------------------------

SYSTEM_ROLE = "You are an expert in Named Entity Recognition (NER) for Vietnamese text."


# -------------------------------------------------------
# 1. ZERO-SHOT PROMPT
# -------------------------------------------------------

def build_zero_shot_prompt(text: str) -> str:
    """
    Zero-shot prompt optimized for Vietnamese NER.

    Based on reference implementation with proven accuracy on VLSP 2018 dataset.
    Uses comprehensive entity definitions and strict extraction rules.

    Args:
        text: Input Vietnamese text

    Returns:
        Formatted prompt string
    """
    return f"""{SYSTEM_ROLE}

TASK: Extract ALL named entities that appear DIRECTLY in the text. Categorize them into 3 types:

1. PERSON (People names):
   - Full names: "Nguyễn Văn A", "Đỗ Tất Lợi", "Angelina Jolie"
   - Titles + names: "ông A", "bà B", "anh Hùng", "chị Mai", "Mr. Kim"
   - Stage names/nicknames: "Hòa Minzy", "Đức Phúc"
   - Character names: "Bao Thanh Thiên", "Bao Chửng", "Kim Siêu Quần"
   - Alternate mentions: "Porzingis", "K. Porzingis", "Kristaps Porzingis" (all are same person)
   - Foreign names: "Brad Pitt", "Donald Trump", "Angela Merkel"
   - Author credits: Extract names after dashes or "Theo"

2. ORGANIZATIONS (Institutions, companies, teams):
   - Government agencies: "Bộ GD-ĐT", "Bộ Giáo dục và Đào tạo", "UBND tỉnh"
   - Companies: "CTCP Dầu thực vật Tường An", "TAC", "Oracle"
   - Schools: "THCS Kỳ Thượng", "Trường Tiểu học Kỳ Thịnh 2"
   - Sports teams/clubs: "Manchester United", "Hà Nội FC", "Quảng Nam", "Raptors"
   - International orgs: "WHO", "FIFA", "VFF", "UEFA", "EU"
   - Media: "phim trường Siêu Quần", "Báo Tiền phong", "VnExpress"
   - Note: Team names often appear without "FC" or "CLB" prefix

3. ADDRESS (Locations, places):
   - Countries: "Việt Nam", "Mỹ", "Trung Quốc", "Brazil"
   - Provinces/cities: "tỉnh Hà Tĩnh", "TP.HCM", "Hà Nội", "Thanh Đảo"
   - Districts: "quận 3", "huyện Kỳ Anh", "thị xã Kỳ Anh"
   - Wards/communes: "phường Kỳ Trinh", "xã Kỳ Nam"
   - Specific places: "sân Tam Kỳ", "Biển Đen", "đường Bàn Cờ", "Air Canada Centre"
   - Note: Vietnamese uses prefixes like "tỉnh", "thành phố", "quận", "huyện"

CRITICAL RULES:
✓ ONLY extract entities that appear DIRECTLY in the text (NO inference, NO translation)
✓ PRESERVE original spelling, accents, and capitalization from source text
✓ If same person has multiple mentions (e.g., "Đức Phúc" and "Phúc") → list ALL
✓ Each entity appears ONCE per category (remove duplicates)
✓ If no entities found → return empty array []
✓ Remove trailing punctuation: "ông A." → "ông A"
✓ Include ALL alternate names for same entity

VIETNAMESE TEXT TO ANALYZE:
{text}

RETURN ONLY THIS JSON (no explanations, no markdown):
{{
  "person": [],
  "organizations": [],
  "address": []
}}
"""


# -------------------------------------------------------
# 2. FEW-SHOT PROMPT
# -------------------------------------------------------

def build_few_shot_prompt(text: str, examples: List[Dict]) -> str:
    """
    Few-shot prompt with examples for Vietnamese NER.

    Demonstrates correct extraction patterns using real examples from VLSP 2018.

    Args:
        text: Input Vietnamese text
        examples: List of example dicts with 'input' and 'output' keys

    Returns:
        Formatted prompt string
    """
    if not examples:
        return build_zero_shot_prompt(text)

    # Format examples properly
    examples_text = ""
    for i, ex in enumerate(examples[:3], 1):
        ex_text = ex['input'][:500]
        ex_output = json.dumps(ex['output'], ensure_ascii=False, indent=2)
        examples_text += f"""
EXAMPLE {i}:
Vietnamese Text: {ex_text}
Extracted Entities:
{ex_output}
---
"""

    return f"""{SYSTEM_ROLE}

LEARN FROM THESE EXAMPLES showing correct Vietnamese entity extraction:

{examples_text}

KEY PATTERNS TO RECOGNIZE:
1. PERSON: Vietnamese names (3+ words), foreign names, titles (ông/bà/anh/chị), stage names, character names
2. ORGANIZATIONS:
   - Vietnamese gov agencies start with "Bộ", "UBND", "CTCP"
   - Schools often have "Trường", "THCS", "THPT"
   - Sports teams may appear as just city names: "Quảng Nam", "Hà Nội"
3. ADDRESS:
   - Administrative hierarchy: "tỉnh" (province), "thành phố" (city), "quận" (district), "huyện" (district), "xã" (commune), "phường" (ward)
   - Always keep the prefix with the location name

EXTRACTION RULES:
✓ Extract ONLY entities that appear DIRECTLY in the text
✓ PRESERVE original Vietnamese spelling with accents (dấu)
✓ List ALL alternate mentions of same entity separately
✓ Remove duplicates within each category
✓ Keep original capitalization from text

NOW EXTRACT ENTITIES FROM THIS NEW TEXT:
{text}

RETURN ONLY JSON (no explanations, no markdown):
{{"person": [], "organizations": [], "address": []}}
"""


# -------------------------------------------------------
# 3. CHAIN-OF-THOUGHT PROMPT
# -------------------------------------------------------

def build_chain_of_thought_prompt(text: str) -> str:
    """
    Chain-of-Thought prompt for Vietnamese NER.

    Guides the model through systematic step-by-step entity extraction.
    Based on reference implementation with detailed reasoning steps.

    Args:
        text: Input Vietnamese text

    Returns:
        Formatted prompt string
    """
    return f"""{SYSTEM_ROLE}
Analyze this Vietnamese text step by step to extract all named entities.

VIETNAMESE TEXT:
{text}

STEP 1: Understand the context
- What is this text about? (news, sports, entertainment, politics, etc.)
- This helps identify entity types (athletes vs actors vs politicians)

STEP 2: Find ALL PERSON entities
Scan for:
- Vietnamese names (usually 2-4 words, capitalized): "Nguyễn Văn A", "Đỗ Tất Lợi"
- Titles + names: "ông Kim", "bà Nghĩa", "anh Hùng", "chị Mai", "Mr. Smith"
- Names in quotes (often stage names or character names): "Bao Thanh Thiên", "Hòa Minzy"
- Foreign names in Latin script: "Brad Pitt", "LeBron James", "Angela Merkel"
- Abbreviated names: "Phúc" if "Đức Phúc" appears, "Kim" if "Kim Siêu Quần" appears
- Author credits (after dash or "Theo"): "- Thanh Hoài", "Theo TH"
IMPORTANT: If same person has multiple mentions → ADD ALL (e.g., both "Porzingis" AND "Kristaps Porzingis")

STEP 3: Find ALL ORGANIZATIONS
Look for Vietnamese patterns:
- Government: starts with "Bộ" (ministry), "UBND", "Liên đoàn", "Ủy ban"
- Companies: contains "CTCP", "Công ty", "Corporation"
- Schools: starts with "Trường", contains "THCS", "THPT", "Tiểu học"
- Sports teams: often just city names ("Hà Nội", "Quảng Nam") or with "FC", "CLB"
- Acronyms: WHO, FIFA, VFF, GD-ĐT, TAC, EU, NASA
- Media: "Báo" + name, magazine names

STEP 4: Find ALL ADDRESS entities
Vietnamese administrative hierarchy (keep prefixes):
- Province: "tỉnh" + name (e.g., "tỉnh Hà Tĩnh", "tỉnh Quảng Bình")
- City: "thành phố" or "TP" + name (e.g., "TP.HCM", "thành phố Hà Nội")
- District: "quận" or "huyện" or "thị xã" + name
- Ward/commune: "phường" or "xã" + name
- Countries: "Việt Nam", "Mỹ", "Trung Quốc", "Brazil"
- Specific places: stadiums ("sân" + name), seas ("Biển" + name), streets ("đường" + name)

STEP 5: Clean and deduplicate
- Remove exact duplicates from each category
- Keep original spelling/capitalization from source text
- Remove trailing punctuation (. , ; :)
- Do NOT guess or infer entities not in text
- Do NOT translate entities

RETURN ONLY THE FINAL JSON RESULT (no explanations, no markdown):
{{"person": [], "organizations": [], "address": []}}
"""


# -------------------------------------------------------
# 4. CUSTOM PROMPT (Legacy support)
# -------------------------------------------------------

def build_custom_prompt(
    text: str,
    instruction: str = "",
    examples: List[Dict] = None,
    rules: List[str] = None
) -> str:
    """
    Build a custom prompt with configurable components.

    Legacy function for backward compatibility.
    For production use, prefer the optimized zero_shot, few_shot, or cot prompts.

    Args:
        text: Input Vietnamese text
        instruction: Custom instruction text
        examples: Optional list of examples
        rules: Optional list of extraction rules

    Returns:
        Formatted prompt string
    """
    prompt_parts = [f"{SYSTEM_ROLE}\n"]

    if instruction:
        prompt_parts.append(f"{instruction}\n")

    if examples:
        prompt_parts.append("\nLEARN FROM THESE EXAMPLES:\n")
        for i, ex in enumerate(examples[:3], 1):
            ex_text = ex['input'][:500]
            ex_output = json.dumps(ex['output'], ensure_ascii=False, indent=2)
            prompt_parts.append(f"""
EXAMPLE {i}:
Text: {ex_text}
Entities:
{ex_output}
---
""")

    if rules:
        prompt_parts.append("\nEXTRACTION RULES:\n")
        for rule in rules:
            prompt_parts.append(f"- {rule}\n")

    prompt_parts.append(f"\nVIETNAMESE TEXT:\n{text}\n")
    prompt_parts.append('\nRETURN ONLY JSON:\n{"person": [], "organizations": [], "address": []}\n')

    return "".join(prompt_parts)


# -------------------------------------------------------
# 5. INSTRUCT-STYLE PROMPT (For Fine-tuned Models)
# -------------------------------------------------------

def build_prompt(text: str, instruction: str = None) -> str:
    """
    Build inference prompt for NER task in instruction-following format.
    
    This is the standard format used for fine-tuned models and evaluation.
    Compatible with instruction-tuned LLMs (e.g., Mistral-Instruct, Llama-Instruct).
    
    Args:
        text: Input text for NER
        instruction: Custom instruction (optional). If None, uses default Vietnamese NER instruction.
    
    Returns:
        Formatted prompt string in ### Instruction / ### Input / ### Response format
    
    Example:
        >>> prompt = build_prompt("Ông Nguyễn Văn A làm việc tại công ty ABC ở Hà Nội.")
        >>> # Returns formatted prompt ready for model inference
    """
    if instruction is None:
        instruction = (
            "You are a Vietnamese Named Entity Recognition (NER) expert. "
            "Extract named entities from the given text and classify them into three categories:\n"
            "- person: Names of people\n"
            "- organizations: Names of organizations, companies, institutions\n"
            "- address: Location names, addresses\n\n"
            "Return your answer as a JSON object with these three keys. "
            "Each value should be a list of strings. "
            "If a category has no entities, return an empty list. "
            "Do not invent entities that are not present in the text."
        )
    
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:"
