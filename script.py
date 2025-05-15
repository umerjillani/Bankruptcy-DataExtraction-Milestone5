import os
import pandas as pd
import re
import csv 
import requests
from PyPDF2 import PdfReader
from io import BytesIO
from datetime import datetime
from openai import OpenAI
import tempfile
from rag import RAGSystem
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY')  # Reads from .env
)
# Constants for column names
COLUMNS = [
    "STATUS", "COMPANY NAME", "CASE NO.", "COURT", "CORPORATE HQ STATE", "INDUSTRY", 
    "CHIEF RESTRUCTURING OFFICER", "COUNSEL FOR DEBTORS AND DIP 1", "COUNSEL FOR DEBTORS AND DIP 2", 
    "FA/IB TO THE DEBTOR", "COUNSEL FOR COMMITTEE OF UNSECURED CREDITORS 1", 
    "COUNSEL FOR COMMITTEE OF UNSECURED CREDITORS 2", "FA FOR COMMITTEE OF UNSECURED CREDITORS 1", 
    "FA FOR COMMITTEE OF UNSECURED CREDITORS 2", "CLAIMS AGENT", "JUDGE", 
    "APPOINTED LIQUIDATING TRUSTEE", "VOTING DEADLINE", "CONFIRMATION HEARING DEADLINE", 
    "Notes from last time check"
]


# Compile regex patterns once
CASE_NO_PATTERN = re.compile(r"Case\s*No\.?\s*[:\.]?\s*([\w-]+\s*[\d]*)", re.IGNORECASE)
PDF_CASE_NO_PATTERN = re.compile(r"Case\s*[:\.]?\s*([\d]+\s*-\s*[\d]+)", re.IGNORECASE)
JUDGE_PATTERN = re.compile(
    r"Judge\s+([A-Za-z\s\.]+?)(?=\s+added to case|$)",  # Capture full name including periods
    re.IGNORECASE
)
CLAIMS_AGENT_PATTERN = re.compile(
    r"Application to Appoint (?:Claims/Noticing|Claims|Noticing) Agent\s+"
    r"([\w\s&\.,-]+?)(?=\s*\(|\s*Filed By|$)",
    re.IGNORECASE)

FA_IB_PATTERNS = [
    re.compile(r'\b(?:Employment|Retention)\b.*?of\s+([A-Za-z0-9&.,\-\s]+?)\s+as\s+(?:Financial\s+Advisor|Investment\s+Banker)\b', re.IGNORECASE),
    re.compile(r'\bApplication\s+of\s+([A-Za-z0-9&.,\-\s]+?)\s+as\s+(?:Financial\s+Advisor|Investment\s+Banker)\b', re.IGNORECASE),
    re.compile(r'\b(?:Employ|Retain)\b\s+([A-Za-z0-9&.,\-\s]+?)\s+as\s+(?:Financial\s+Advisor|Investment\s+Banker)\b', re.IGNORECASE),
    re.compile(r'\bReimbursement\s+of\s+Expenses\s+of\s+([A-Za-z0-9&.,\-\s]+?)\s+as\s+(?:Financial\s+Advisor|Investment\s+Bankers?)\b', re.IGNORECASE),
]

# CONFIRMATION_HEARING_PATTERN = re.compile(
#     r"Confirmation\s+Hearing\s+scheduled\s+for\s+(\d{1,2}/\d{1,2}/\d{4})"  # Date capture group
#     r"(?=\s+at\s+\d{1,2}:\d{2}\s+[AP]M)",  # Positive lookahead for time format
#     re.IGNORECASE
# )

CRO_KEYWORDS = [
    "chief restructuring officer",
    "As chief restructuring officer"# Catches phrases like "[Firm] as restructuring officer"
]

US_STATES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
    'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
    'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
    'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy',
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
    'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
    'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
    'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
    'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
    'new hampshire', 'new jersey', 'new mexico', 'new york',
    'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
    'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
    'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
    'west virginia', 'wisconsin', 'wyoming'
}

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def extract_case_number(description, recap_urls):
    """Extracts Case Number from description, pdf_urls, or recap_urls."""
    match = CASE_NO_PATTERN.search(description)
    if match:
        return match.group(1).strip()
    
    for url in recap_urls:
        try:
            response = requests.get(url)
            reader = PdfReader(BytesIO(response.content))
            first_page_text = reader.pages[0].extract_text()
            pdf_match = PDF_CASE_NO_PATTERN.search(first_page_text)
            if pdf_match:
                return pdf_match.group(1).strip()
        except Exception:
            continue
    return None

def filter_cro_lines(descriptions):
    """Collect descriptions containing explicit CRO keywords."""
    cro_lines = []
    for desc in descriptions:
        desc_lower = desc.lower()
        if any(keyword in desc_lower for keyword in CRO_KEYWORDS):
            cro_lines.append(desc)  # Keep original description
    return cro_lines

def truncate_text_for_openai(text, max_tokens=4000):
    """Truncate text to approximately stay within token limits"""
    # Rough estimate: 1 token ≈ 4 chars for English text
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text

def openai_role(role_name: str, prompt_text: str) -> dict:
    """
    Queries OpenAI to extract role-specific information from legal documents.
    
    Args:
        role_name: The type of role to extract (e.g., "CRO", "DIP1 & DIP2")
        prompt_text: Text content to analyze
        
    Returns:
        Dictionary containing extracted information
    """
    # Define templates for different roles
    PROMPTS = {
        "CRO": f"""Below are docket entry descriptions from a bankruptcy case. 
                        
        IMPORTANT: Extract ONLY the Chief Restructuring Officer (CRO) firm and individual name 
        if they are EXPLICITLY mentioned as a CRO in the text. Do not extract names of law firms,
        debtors, or other parties unless they are clearly designated as CROs.

        A valid CRO mention typically includes phrases like:
        - "Designating [Name] as Chief Restructuring Officer"
        - "Retain [Firm] to provide a Chief Restructuring Officer"
        - "Employ [Firm] as...Chief Restructuring Officer"

        Do NOT extract:
        - The debtor company name
        - Law firms representing the debtor
        - The company name in the case title is NOT the CRO firm
        - Financial advisors not designated as CROs
        - Names or firms mentioned in other contexts

        CONTEXTUAL ANALYSIS (when patterns don't match):
        - Look for definitive articles ("the Chief Restructuring Officer")
        - Check for capitalization of official titles
        - Verify the individual/firm is being appointed (not just mentioned)

        Examples of correct extraction:
        POSITIVE (extract these):
        - "Designating Philip J. Gund as Chief Restructuring Officer" → Philip J. Gund
        - "Retain Riveron Management Services to provide CRO" → Riveron Management Services
        - "Employ Getzler Henrich to provide David R. Campbell as CRO" → Getzler Henrich (David R. Campbell)
        - "Steven Shenker shall serve as Chief Restructuring Officer" → Steven Shenker
        - "Notice of Deposition of Mark Smith, Chief Restructuring Officer of Fulcrum Bioenergy" → Mark Smith (ignore "Fulcrum Bioenergy")

        If you cannot find a clear CRO designation, return "TBD" only.

        Return in this exact format:
        CRO: FIRM_NAME (INDIVIDUAL_NAME) - if both are found
        CRO: FIRM_NAME - if only firm is found
        CRO: INDIVIDUAL_NAME - if only individual is found
        CRO: TBD - if nothing is found

        Text:
        {truncate_text_for_openai(prompt_text)}""",

        "DIP1 & DIP2": f"""Extract the following information from the provided text:
1. Primary Counsel (DIP1):
   - Extract the primary bankruptcy counsel firm name
   - This is the main firm handling the bankruptcy case
   - Exclude any mentions of co-counsel or special counsel
   - If multiple legal entities are mentioned as part of the same firm, include all of them
   - Example: "Kirkland & Ellis LLP and Kirkland & Ellis International LLP" should be extracted as one firm
   - Look for phrases like "counsel for the debtors", "bankruptcy counsel to the debtor", etc.
   - Exclude any mentions that include "co-counsel" or "special counsel"

2. Co-Counsel (DIP2):
   - Extract the co-counsel firm name if present
   - Only include firms explicitly mentioned as co-counsel
   - Exclude special counsel or other legal roles
   - If multiple legal entities are mentioned as part of the same firm, include all of them
   - Look for phrases like "co-counsel for the debtors", "bankruptcy co-counsel to the debtors", etc.
   - Must include explicit "co-counsel" designation

Text to analyze:
{prompt_text}

Return the information in the following JSON format:
{{
    "DIP1": "primary counsel firm name or None if not found",
    "DIP2": "co-counsel firm name or None if not found"
}}"""
    }

    try:
        # Get the appropriate prompt for the requested role
        if role_name not in PROMPTS:
            return {"error": f"Unknown role: {role_name}"}
            
        # Query OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a legal document analyst specialized in bankruptcy court dockets."},
                {"role": "user", "content": PROMPTS[role_name]}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse results based on role
        if role_name == "CRO":
            # Extract CRO from format "CRO: [value]"
            cro_match = re.match(r"CRO:\s*(.*)", result)
            if cro_match:
                cro_value = cro_match.group(1).strip()
                # Remove any quotes that might be in the response
                cro_value = cro_value.replace('"', '').replace("'", '')
                return {"CRO": cro_value}
            return {"CRO": "TBD"}
            
        elif role_name == "DIP1 & DIP2":
            # Extract DIP1 and DIP2 from multi-line format
            dip1_match = re.search(r"DIP1:\s*(.*?)(?=$|\nDIP2:)", result, re.DOTALL)
            dip2_match = re.search(r"DIP2:\s*(.*?)(?=$)", result, re.DOTALL)
            
            dip1 = dip1_match.group(1).strip() if dip1_match else "TBD"
            dip2 = dip2_match.group(1).strip() if dip2_match else "TBD"
            
            # Remove any quotes and clean up the firm names
            dip1 = dip1.replace('"', '').replace("'", '').strip()
            dip2 = dip2.replace('"', '').replace("'", '').strip()
            
            # Additional cleaning for firm names
            dip1 = re.sub(r'\s+', ' ', dip1)  # Normalize whitespace
            dip2 = re.sub(r'\s+', ' ', dip2)  # Normalize whitespace
            
            return {"DIP1": dip1, "DIP2": dip2}
        
        # For future role types
        return {"error": "Unsupported role type"}
        
    except Exception as e:
        print(f"Error querying OpenAI for {role_name}: {e}")
        # Return appropriate default values based on role
        if role_name == "CRO":
            return {"CRO": "TBD"}
        elif role_name == "DIP1 & DIP2":
            return {"DIP1": "TBD", "DIP2": "TBD"}
        return {"error": str(e)}
    
def extract_cro_with_openai(descriptions):
    """
    Extract CRO information using hybrid regex/OpenAI approach.
    Returns formatted string: "FIRM (PERSON)" or just one component.
    """
    # Filter to only relevant lines
    cro_lines = filter_cro_lines(descriptions)
    if not cro_lines:
        return "TBD"
    
    # Join into single block for OpenAI
    prompt_text = "\n".join(cro_lines)
    
    # Query OpenAI - use the new openai_role function
    result = openai_role("CRO", prompt_text)
    cro_value = result.get("CRO", "TBD")
    
    # Return the cleaned string result
    return cro_value    

def extract_dip1_with_openai(descriptions):
    """Extracts DIP1 using regex patterns and OpenAI for cleanup."""
    patterns = [
        # Primary Pattern
        (
            r"(?:Retain\s+)?([A-Za-z0-9&,\.\-\s\(\)]+?(?:(?:LLP|LLC|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.)\b)?)\s*"
            r"(?:,?\s*(?:as\s*)(?!Co-)(?:Attorney(?:s)?|Counsel|Bankruptcy\s+Counsel||Conflicts\s+Counsel)\s*)"
            r"(?:for|to)\s+(?:the\s+)?Debtor(?:s)?(?:\s+and\s+Debtor(?:s)?\s+in\s+Possession)?"
        ),
        # Secondary Pattern
        (
            r"(?:Retain\s+)?([A-Za-z0-9&,\.\-\s]+?(?:\([A-Za-z]+\))?\s*(?:LLP|LLC|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.))\s*"
            r"(?:,?\s*(?:as\s*)?(?:Attorneys|Counsel|Bankruptcy\s+Counsel)\s*)"
            r"(?:for|to)\s+(?:the\s+)?Debtor(?:s)?(?:\s+and\s+Debtor(?:s)?\s+in\s+Possession)?"
        )
    ]


    # Check patterns in priority order
    for pattern in patterns:
        for text in descriptions:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                firm_name = match.group(1).strip()
                print(f"Before cleanup: '{firm_name}'")
                
                # Use OpenAI to clean up the firm name
                prompt = f"""Clean up the following law firm name by removing any unnecessary prefixes, suffixes, or extra text. 
                Return ONLY the clean firm name. If there are multiple entities of the same firm, include all of them.
                
                Example inputs and outputs:
                Input: "Employment of Kirkland & Ellis LLP as Bankruptcy Counsel"
                Output: "Kirkland & Ellis LLP"
                
                Input: "Retention of Kirkland & Ellis LLP and Kirkland & Ellis International LLP"
                Output: "Kirkland & Ellis LLP and Kirkland & Ellis International LLP"
                
                Input: "Application of Kirkland & Ellis LLP as Bankruptcy Counsel to the Debtors"
                Output: "Kirkland & Ellis LLP"
                
                Now clean up this firm name:
                {firm_name}"""
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a legal document analyst specialized in cleaning up law firm names."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    cleaned_name = response.choices[0].message.content.strip()
                    print(f"After OpenAI cleanup: '{cleaned_name}'")
                    return cleaned_name
                except Exception as e:
                    print(f"Error in OpenAI cleanup: {e}")
                    return firm_name  # Return original if cleanup fails

    return "TBD"

def extract_dip2_with_openai(descriptions):
    """Extracts DIP2 using regex patterns and OpenAI for cleanup."""
    patterns = [
        # Primary pattern with jurisdiction/type modifiers
        (
            r"(?:Retain\s+)?([A-Za-z0-9&,\.\-\s\(\)]+?(?:LLP|LLC|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.))\s*"
            r"(?:,?\s*(?:as\s*)?(?:[A-Za-z\s]*?Co-Counsel|Bankruptcy\s+Co-Counsel))\s*"
            r"(?:for|to)\s+(?:the\s+)?Debtors?"
        )
    ]

    for pattern in patterns:
        for text in descriptions:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                firm_name = match.group(1).strip()
                print(f"Before cleanup: '{firm_name}'")

                # Use OpenAI to clean up the firm name
                prompt = f"""Clean up the following co-counsel law firm name by removing any unnecessary prefixes, suffixes, or extra text. 
                Return ONLY the clean firm name. If there are multiple entities of the same firm, include all of them.
                
                Example inputs and outputs:
                Input: "Employment of Young Conaway Stargatt & Taylor, LLP as Co-Counsel"
                Output: "Young Conaway Stargatt & Taylor, LLP"
                
                Input: "Retention of Young Conaway Stargatt & Taylor, LLP and Young Conaway International LLP"
                Output: "Young Conaway Stargatt & Taylor, LLP and Young Conaway International LLP"
                
                Input: "Application of Young Conaway Stargatt & Taylor, LLP as Delaware Co-Counsel"
                Output: "Young Conaway Stargatt & Taylor, LLP"
                
                Now clean up this firm name:
                {firm_name}"""
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a legal document analyst specialized in cleaning up law firm names."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1
                    )
                    cleaned_name = response.choices[0].message.content.strip()
                    print(f"After OpenAI cleanup: '{cleaned_name}'")
                    return cleaned_name
                except Exception as e:
                    print(f"Error in OpenAI cleanup: {e}")
                    return firm_name  # Return original if cleanup fails

    print("DEBUG (DIP2): No matches found in any text")
    return "TBD"


def extract_fa_ib_debtor(descriptions):
    """
    Extract Financial Advisor (FA) and Investment Banker (IB) information from docket entries.
    Process:
    1. Use regex patterns to identify relevant docket entries
    2. Pass these entries to OpenAI for extraction
    3. Format the response as:
       - "FIRM_NAME (FA)" if only Financial Advisor is found
       - "FIRM_NAME (IB)" if only Investment Banker is found  
       - "FIRM_NAME_1 (FA) / FIRM_NAME_2 (IB)" if both are found
       - "TBD" if nothing is found
    """
    # Initialize set to track pattern matches (to avoid duplicates)
    relevant_descriptions = set()
    
    # Define pattern groups
    fa_patterns = [
        r"as\s+Financial\s+Advisor\s+to\s+the\s+Debtor(?:s)?",
        r"as\s+Financial\s+Advisor\s+to\s+the\s+Debtor(?:s)?\s+and\s+Debtor(?:s)?-in-Possession",
        r"as\s+Accountant\s+and\s+Financial\s+Advisor"
    ]
    
    ib_patterns = [
        r"as\s+Investment\s+Banker\s+for\s+the\s+Debtor\s+and\s+Debtor\s+in\s+Possession",
        r"as\s+Investment\s+Banker\s+to\s+the\s+Debtor(?:s)?",
        r"as\s+Investment\s+Banker(?!\s+to|\s+for\s+the\s+(?:Official\s+Committee|Committee))",
        r"as\s+Investment\s+Banker\s+and\s+Capital\s+Markets"
    ]
    
    dual_patterns = [
        r"as\s+Financial\s+Advisor\s+and\s+Investment\s+Banker\s+for\s+the\s+Debtor(?:s)?\s+and\s+Debtor(?:s)?\s+in\s+Possession"
    ]
    
    # Track which patterns have been matched
    fa_pattern_matched = False
    ib_pattern_matched = False
    dual_pattern_matched = False
    
    # Process each description
    for desc in descriptions:
        # Check for dual role first (most specific)
        if not dual_pattern_matched:
            for pattern in dual_patterns:
                if re.search(pattern, desc, re.IGNORECASE):
                    print(f"Dual FA/IB match found: {desc[:100]}...")
                    relevant_descriptions.add(desc)
                    dual_pattern_matched = True
                    break
        
        # Check for FA patterns if we haven't matched one yet
        if not fa_pattern_matched and not dual_pattern_matched:
            for pattern in fa_patterns:
                if re.search(pattern, desc, re.IGNORECASE):
                    print(f"FA match found: {desc[:100]}...")
                    relevant_descriptions.add(desc)
                    fa_pattern_matched = True
                    break
        
        # Check for IB patterns if we haven't matched one yet
        if not ib_pattern_matched and not dual_pattern_matched:
            for pattern in ib_patterns:
                if re.search(pattern, desc, re.IGNORECASE):
                    print(f"IB match found: {desc[:100]}...")
                    relevant_descriptions.add(desc)
                    ib_pattern_matched = True
                    break
    
    # If no matches found, return TBD
    if not relevant_descriptions:
        print("No FA/IB matches found")
        return "TBD"
    
    # Create prompt for OpenAI
    prompt_text = "\n\n".join(relevant_descriptions)
    
    # Prepare the OpenAI prompt with better guidance
    openai_prompt = f"""Extract ONLY the Financial Advisor (FA) and/or Investment Banker (IB) firms that are explicitly serving the DEBTOR company from the following bankruptcy docket entries.

EXTREMELY IMPORTANT RULES:
1. ONLY extract firms that are EXPLICITLY stated as serving the debtor, NOT committees or other parties
2. A firm is serving the debtor if the text mentions phrases like:
   - "as Financial Advisor to the Debtor(s)"
   - "as Investment Banker for the Debtor"
   - "as Financial Advisor for the Debtor and Debtor-in-Possession"
3. DO NOT extract a firm if it's identified as serving a committee, like:
   - "Financial Advisor to the Official Committee"
   - "Investment Banker for the Committee"
4. DO NOT assume the filing company is the FA or IB unless explicitly stated
5. If you cannot confidently identify a firm serving as FA or IB for the DEBTOR specifically, return "TBD"

Look for patterns like:
- "Employment of [FIRM NAME] as Financial Advisor to the Debtors"
- "Retention of [FIRM NAME] as Investment Banker for the Debtor"

Return the information in ONE of these specific formats:
1. If only a Financial Advisor is found: "FIRM_NAME (FA)"
2. If only an Investment Banker is found: "FIRM_NAME (IB)"
3. If both roles are found with different firms: "FIRM_NAME_1 (FA) / FIRM_NAME_2 (IB)"
4. If one firm serves both roles: "FIRM_NAME (FA/IB)"
5. If no firm can be confidently identified as serving the debtor: "TBD"

DO NOT include any explanation. Return exactly one line in one of the formats above.

Docket entries:
{prompt_text}
"""
    
    try:
        # Query OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a bankruptcy document analyst specialized in identifying financial advisors and investment bankers for the debtor company only. Return results in a single line without numbering."},
                {"role": "user", "content": openai_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        print(f"OpenAI response: {result}")
        
        # Validate and format the result
        # Remove any numbering from the response
        result = re.sub(r'^\d+\.\s+', '', result)
        
        # Remove multiple lines if present
        if '\n' in result:
            result = result.split('\n')[0].strip()
        
        # Check if result matches expected formats
        if result == "TBD" or re.match(r".*\(FA\).*", result) or re.match(r".*\(IB\).*", result) or re.match(r".*\(FA/IB\).*", result):
            return result
        else:
            # If format doesn't match expectations, attempt to extract and format properly
            fa_match = re.search(r"([\w\s,&\.]+)\s*\(FA\)", result)
            ib_match = re.search(r"([\w\s,&\.]+)\s*\(IB\)", result)
            
            if fa_match and ib_match:
                return f"{fa_match.group(1).strip()} (FA) / {ib_match.group(1).strip()} (IB)"
            elif fa_match:
                return f"{fa_match.group(1).strip()} (FA)"
            elif ib_match:
                return f"{ib_match.group(1).strip()} (IB)"
            else:
                print(f"Response format invalid: {result}")
                return "TBD"
            
    except Exception as e:
        print(f"Error in OpenAI extraction: {e}")
        return "TBD"

def extract_committee_counsel1(all_texts):
    """Extracts firm names serving as Counsel to the Official Committee of Unsecured Creditors from docket entries."""
    # Primary pattern to locate firm name near "Counsel to the Committee"
    main_pattern = re.compile(
        r"([A-Za-z0-9&,\.\-\s\(\)]+?(?:LLP|LLC|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.))\s*"
        r"(?:,?\s*(?:as\s*)?)?(?:Counsel)\s+(?:to|for)\s+the\s+(?:Official\s+)?Committee\s+of\s+Unsecured\s+Creditors",
        re.IGNORECASE
    )
    
    # Cleanup phrases from your original function
    cleanup_phrases = [
        r"Retain(?:ing)?\s+",
        r"Employment\s+of",
        r"Retention\s+of",
        r"Application\s+for\s+Compensation\s+of",
        r"Application\s+of",
        r"Monthly\s+Fee\s+Application\s+of"
    ]
    
    # Create one comprehensive regex that finds any of these phrases plus everything before them
    prefix_pattern = r'^.*?(?:' + '|'.join(cleanup_phrases) + ')'
    prefix_cleanup = re.compile(prefix_pattern, re.IGNORECASE)
    
    # Process each text entry
    for text in all_texts:
        match = main_pattern.search(text)
        if match:
            firm_name = match.group(1).strip()
            
            # Apply the comprehensive cleanup regex repeatedly until no more matches
            previous_firm_name = ""
            while previous_firm_name != firm_name:
                previous_firm_name = firm_name
                firm_name = prefix_cleanup.sub('', firm_name).strip()
            
            # Remove any trailing commas or periods
            firm_name = re.sub(r'[,\.]$', '', firm_name).strip()
            
            return firm_name
    
    return "TBD"

def extract_committee_counsel2(all_texts):
    """
    Extracts firm names serving as Delaware/Co-Counsel for the Committee of Unsecured Creditors.
    """
    # Main pattern to extract firm name followed by Co-Counsel or Delaware Counsel
    co_counsel_pattern = re.compile(
        r"([A-Za-z0-9&,\.\-\s\(\)]+?(?:LLP|LLC|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.))\s*"
        r"(?:,?\s*(?:as\s*)?)?(?:Co-\s*Counsel|Delaware\s+Counsel)\s+(?:to|for)\s+the\s+(?:Official\s+)?Committee\s+of\s+Unsecured\s+Creditors",
        re.IGNORECASE
    )
    
    # Extract cleanup phrases from the original regex
    cleanup_phrases = [
        r"Retain(?:ing)?\s+",
        r"Employment\s+of",
        r"Retention\s+of",
        r"Increase\s+of",
        r"Application\s+for\s+Compensation\s+of",
        r"Application\s+of",
        r"Monthly\s+Fee\s+Application\s+of"
    ]
    
    # Create one comprehensive regex that finds any of these phrases plus everything before them
    prefix_pattern = r'^.*?(?:' + '|'.join(cleanup_phrases) + ')'
    prefix_cleanup = re.compile(prefix_pattern, re.IGNORECASE)
    
    for text in all_texts:
        match = co_counsel_pattern.search(text)
        if match:
            firm_name = match.group(1).strip()
            
            # Apply the cleanup regex repeatedly until no more matches
            previous_firm_name = ""
            while previous_firm_name != firm_name:
                previous_firm_name = firm_name
                firm_name = prefix_cleanup.sub('', firm_name).strip()
            
            # Remove any trailing commas or periods
            firm_name = re.sub(r'[,\.]$', '', firm_name).strip()
            print("AFTER PATTERN & CLEANUP:", firm_name)  # Debug: print cleaned result
            return firm_name
    
    return "TBD"

def extract_committee_financial_advisor1(all_texts):
    """
    Extracts firm names serving as Financial Advisors or Investment Bankers 
    to the Official Committee of Unsecured Creditors from docket entries.
    """
    advisor_pattern = re.compile(
        r"([A-Za-z0-9&,\.\-\s\(\)]+?(?:LLP|LLC|Inc\.|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.|Advisors|Group|Partners|Consulting|Capital))\s*"
        r"(?:,?\s*(?:as\s*)?)?(?:Financial\s+Advisor|Investment\s+Bankers?)\s+(?:to|for)\s+the\s+(?:Official\s+)?Committee\s+of\s+Unsecured\s+Creditors",
        re.IGNORECASE
    )

    # Extract cleanup phrases from the original regex
    cleanup_phrases = [
        r"Retain(?:ing)?\s+",
        r"Employ(?:ment)?\s+of",
        r"Retention\s+(?:and|of)\s+Employment\s+of",
        r"Application\s+(?:for\s+Compensation\s+)?of",
        r"Monthly\s+Fee\s+Application\s+of",
        r"Application\s+of",
        r"Retention\s+of",
        r"Expenses\s+of",
        r"Motion\s+to\s+Employ\s+",
        r"Order\s+Authorizing\s+the\s+Employment\s+of"
    ]
    
    # Create one comprehensive regex that finds any of these phrases plus everything before them
    prefix_pattern = r'^.*?(?:' + '|'.join(cleanup_phrases) + ')'
    prefix_cleanup = re.compile(prefix_pattern, re.IGNORECASE)

    for text in all_texts:
        match = advisor_pattern.search(text)
        if match:
            firm_name = match.group(1).strip()
            
            # Apply the cleanup regex repeatedly until no more matches
            previous_firm_name = ""
            while previous_firm_name != firm_name:
                previous_firm_name = firm_name
                firm_name = prefix_cleanup.sub('', firm_name).strip()
            
            # Remove any trailing commas or periods
            firm_name = re.sub(r'[,\.]$', '', firm_name).strip()
            return firm_name

    return "TBD"

def extract_committee_financial_advisor2(all_texts):
    """
    Extracts firm names serving as secondary Financial Advisors 
    (e.g., Co-Financial Advisor) to the Official Committee of Unsecured Creditors.
    """
    co_fa_pattern = re.compile(
    r"([A-Za-z0-9&,\.\-\s\(\)]+?(?:LLP|LLC|Inc\.|P\.C\.|P\.A\.|L\.L\.P\.|L\.P\.|Advisors|Consulting|Group|Capital|Partners))\s*"
    r"(?:,?\s*(?:as\s*)?)?(?:Co-)\s*(?:Financial\s+Advisor|Investment\s+Banker)\s+(?:to|for)\s+the\s+(?:Official\s+)?Committee\s+of\s+Unsecured\s+Creditors",
    re.IGNORECASE
        )

    # Extract cleanup phrases from the original regex
    cleanup_phrases = [
        r"Retain(?:ing)?\s+", 
        r"Employment\s+of",
        r"Retention\s+of",
        r"Expenses\s+of",
        r"Application\s+of",
        r"Order\s+Approving",
        r"Application\s+to\s+Employ"
    ]
    
    # Create one comprehensive regex that finds any of these phrases plus everything before them
    prefix_pattern = r'^.*?(?:' + '|'.join(cleanup_phrases) + ')'
    prefix_cleanup = re.compile(prefix_pattern, re.IGNORECASE)

    for text in all_texts:
        match = co_fa_pattern.search(text)
        if match:
            firm_name = match.group(1).strip()
            
            # Apply the cleanup regex repeatedly until no more matches
            previous_firm_name = ""
            while previous_firm_name != firm_name:
                previous_firm_name = firm_name
                firm_name = prefix_cleanup.sub('', firm_name).strip()
            
            # Remove any trailing commas or periods
            firm_name = re.sub(r'[,\.]$', '', firm_name).strip()
            return firm_name

    return "TBD"

def clean_pdf_url(row):
    """Clean PDF URL from both IA and Local columns with prefix handling"""
    # Try recapdocument_filepath_ia first
    ia_url = str(row.get('recapdocument_filepath_ia', ''))
    if ia_url.lower() not in ['nan', 'none', '']:
        match = re.search(r'(https?://.*?\.pdf)', ia_url)
        if match:
            print("Using recapdocument_filepath_ia URL")
            return match.group(1)

    # Fallback to recapdocument_filepath_local
    local_path = str(row.get('recapdocument_filepath_local', ''))
    if local_path.lower() not in ['nan', 'none', '']:
        # Remove existing prefix if present
        clean_path = re.sub(r'^https?://[^/]+/', '', local_path)
        # Add new prefix and check validity
        full_url = f"https://storage.courtlistener.com/{clean_path}"
        if full_url.lower().endswith('.pdf'):
            print("Using modified recapdocument_filepath_local URL")
            return full_url
    
    return None

def extract_liquidating_trustee_with_gpt(text):
    """Use OpenAI to extract liquidating trustee name from text"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Extract the name of the appointed Liquidating Trustee. Return ONLY the name of the person or entity."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def extract_plan_administrator_with_gpt(text):
    """Use OpenAI to extract plan administrator name from text"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Extract the name of the Plan Administrator who is acting on behalf of a Liquidating Trustee. Return ONLY the name of the person or entity."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def extract_liquidating_trustee_from_plan_supplement(df):
    """Process Plan Supplement PDFs to extract Liquidating Trustee information"""
    print("\n=== STARTING PLAN SUPPLEMENT PROCESSING FOR LIQUIDATING TRUSTEE ===")
    rag = RAGSystem()
    liquidating_trustee = None
    
    # Define fingerprint directory
    fingerprint_dir = os.path.join(os.path.dirname(rag.vector_store_dir), "fingerprints")
    os.makedirs(fingerprint_dir, exist_ok=True)
    
    # STAGE 1: First filter Plan Supplement documents from recapdocument_description
    recap_plan_supplement_df = df[df['recapdocument_description'].str.contains('Plan Supplement', case=False, na=False)]
    print(f"Found {len(recap_plan_supplement_df)} plan supplement documents in recapdocument_description to process")

    # Process all recapdocument_description matches first
    for idx, row in recap_plan_supplement_df.iterrows():
        print(f"\n--- Processing recapdocument_description document {idx+1}/{len(recap_plan_supplement_df)} ---")
        
        # Get cleaned URL from either column
        pdf_url = clean_pdf_url(row)
        
        if not pdf_url:
            print("No valid PDF URL found in either column")
            continue
            
        print(f"Final PDF URL: {pdf_url}")
        tmp_path = None
        
        try:
            # Process the PDF - pass current value and get updated value back
            tmp_path, liquidating_trustee = process_pdf_and_extract_trustee(
                pdf_url, rag, fingerprint_dir, liquidating_trustee
            )
            
            # If trustee found, we can stop processing
            if liquidating_trustee:
                print("✅ Liquidating Trustee or Plan Administrator found, stopping PDF processing")
                break

        except Exception as e:
            print(f"❌ PROCESSING ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                print(f"🧹 Cleaning up temp file: {tmp_path}")
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"⚠️ Failed to delete temp file: {str(e)}")
    
    # STAGE 2: If no trustee found, check docketentry_description next
    if not liquidating_trustee:
        # Filter Plan Supplement documents from docketentry_description (excluding ones already processed)
        docket_plan_supplement_df = df[
            (df['docketentry_description'].str.contains('Plan Supplement', case=False, na=False)) & 
            (~df.index.isin(recap_plan_supplement_df.index))  # Exclude already processed rows
        ]
        print(f"Found {len(docket_plan_supplement_df)} additional plan supplement documents in docketentry_description to process")
        
        # Process all docketentry_description matches
        for idx, row in docket_plan_supplement_df.iterrows():
            print(f"\n--- Processing docketentry_description document {idx+1}/{len(docket_plan_supplement_df)} ---")
            
            # Get cleaned URL from either column
            pdf_url = clean_pdf_url(row)
            
            if not pdf_url:
                print("No valid PDF URL found in either column")
                continue
                
            print(f"Final PDF URL: {pdf_url}")
            tmp_path = None
            
            try:
                # Process the PDF - pass current value and get updated value back
                tmp_path, liquidating_trustee = process_pdf_and_extract_trustee(
                    pdf_url, rag, fingerprint_dir, liquidating_trustee
                )
                
                # If trustee found, we can stop processing
                if liquidating_trustee:
                    print("✅ Liquidating Trustee or Plan Administrator found, stopping PDF processing")
                    break

            except Exception as e:
                print(f"❌ PROCESSING ERROR: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    print(f"🧹 Cleaning up temp file: {tmp_path}")
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        print(f"⚠️ Failed to delete temp file: {str(e)}")
                
    print("\n=== PROCESSING COMPLETE ===")
    
    # Clean up old vector stores
    base_vector_dir = os.path.dirname(rag.vector_store_dir)
    rag.cleanup_old_vector_stores(base_vector_dir, max_stores=5)
    
    return liquidating_trustee

# Helper function to extract the PDF processing code (to avoid repetition)
def process_pdf_and_extract_trustee(pdf_url, rag, fingerprint_dir, current_trustee):
    """Process a single PDF to extract liquidating trustee or plan administrator"""
    # Use passed-in value instead of global
    liquidating_trustee = current_trustee
    
    tmp_path = None
    
    # Download PDF once
    print(f"⬇️ Downloading PDF from: {pdf_url}")
    response = requests.get(pdf_url)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
        print(f"💾 Saved temp PDF: {tmp_path}")

    # Process with RAG (with fingerprinting) only once
    print("🔧 Ingesting PDF into RAG...")
    rag.ingest(tmp_path, fingerprint_dir=fingerprint_dir)
    
    # Try to extract the Liquidating Trustee if not found yet
    if not liquidating_trustee:
        print("❓ Querying RAG: 'Who is appointed as the Liquidating Trustee?'")
        answer = rag.ask("Who is appointed as the Liquidating Trustee? Who will serve as the Liquidating Trustee?")
        print(f"📄 RAW RAG ANSWER: {answer}")
        
        trustee_name = extract_liquidating_trustee_with_gpt(str(answer))
        print(f"👤 EXTRACTED LIQUIDATING TRUSTEE: {trustee_name}")
        
        # Check if we got a valid name (not "No liquidating trustee found" or similar)
        if trustee_name and not any(x in trustee_name.lower() for x in ['no', 'not found', 'none', 'unable']):
            print(f"✅ VALID LIQUIDATING TRUSTEE FOUND: {trustee_name}")
            liquidating_trustee = trustee_name
        else:
            # If no Liquidating Trustee, try for Plan Administrator
            print("❓ Querying RAG: 'Who is the Plan Administrator acting on behalf of a Liquidating Trustee?'")
            answer = rag.ask("Who is the Plan Administrator? Is there a Plan Administrator acting on behalf of a Liquidating Trustee?")
            print(f"📄 RAW RAG ANSWER: {answer}")
            
            admin_name = extract_plan_administrator_with_gpt(str(answer))
            print(f"👤 EXTRACTED PLAN ADMINISTRATOR: {admin_name}")
            
            if admin_name and not any(x in admin_name.lower() for x in ['no', 'not found', 'none', 'unable']):
                print(f"✅ PLAN ADMINISTRATOR FOUND AS FALLBACK: {admin_name}")
                liquidating_trustee = admin_name
    
    # Return the tmp_path and the potentially updated trustee variable
    return tmp_path, liquidating_trustee

def extract_hearing_date_with_gpt(text):
    """Use OpenAI to extract date from text"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Extract the hearing date in MM/DD/YYYY format. Return ONLY the date."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def extract_voting_deadline_with_gpt(text):
    """Use OpenAI to extract voting deadline from text"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Extract the voting deadline date in MM/DD/YYYY format. Return ONLY the date."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def extract_deadlines_from_disclosure_statements(df):
    """Process Disclosure Statement PDFs once and extract both dates"""
    print("\n=== STARTING DISCLOSURE STATEMENT PROCESSING FOR ALL DEADLINES ===")
    rag = RAGSystem()
    confirmation_hearing_date = None
    voting_deadline = None
    
    # Define fingerprint directory
    fingerprint_dir = os.path.join(os.path.dirname(rag.vector_store_dir), "fingerprints")
    os.makedirs(fingerprint_dir, exist_ok=True)
    
    # STAGE 1: First filter disclosure statements from recapdocument_description
    recap_disclosure_df = df[df['recapdocument_description'].str.contains('Disclosure Statement', case=False, na=False)]
    print(f"Found {len(recap_disclosure_df)} disclosure statements in recapdocument_description to process")

    # Process all recapdocument_description matches first
    for idx, row in recap_disclosure_df.iterrows():
        print(f"\n--- Processing recapdocument_description document {idx+1}/{len(recap_disclosure_df)} ---")
        
        # Get cleaned URL from either column
        pdf_url = clean_pdf_url(row)
        
        if not pdf_url:
            print("No valid PDF URL found in either column")
            continue
            
        print(f"Final PDF URL: {pdf_url}")
        tmp_path = None
        
        try:
            # Process the PDF - pass current values and get updated values back
            tmp_path, confirmation_hearing_date, voting_deadline = process_pdf_and_extract_dates(
                pdf_url, rag, fingerprint_dir, confirmation_hearing_date, voting_deadline
            )
            
            # Break if both dates found
            if confirmation_hearing_date and voting_deadline:
                print("✅ Both deadlines found, stopping PDF processing")
                break

        except Exception as e:
            print(f"❌ PROCESSING ERROR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                print(f"🧹 Cleaning up temp file: {tmp_path}")
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"⚠️ Failed to delete temp file: {str(e)}")
    
    # STAGE 2: If not both dates found, check docketentry_description next
    if not (confirmation_hearing_date and voting_deadline):
        # Filter disclosure statements from docketentry_description (excluding ones already processed)
        docket_disclosure_df = df[
            (df['docketentry_description'].str.contains('Disclosure Statement', case=False, na=False)) & 
            (~df.index.isin(recap_disclosure_df.index))  # Exclude already processed rows
        ]
        print(f"Found {len(docket_disclosure_df)} additional disclosure statements in docketentry_description to process")
        
        # Process all docketentry_description matches
        for idx, row in docket_disclosure_df.iterrows():
            print(f"\n--- Processing docketentry_description document {idx+1}/{len(docket_disclosure_df)} ---")
            
            # Get cleaned URL from either column
            pdf_url = clean_pdf_url(row)
            
            if not pdf_url:
                print("No valid PDF URL found in either column")
                continue
                
            print(f"Final PDF URL: {pdf_url}")
            tmp_path = None
            
            try:
                # Process the PDF - pass current values and get updated values back
                tmp_path, confirmation_hearing_date, voting_deadline = process_pdf_and_extract_dates(
                    pdf_url, rag, fingerprint_dir, confirmation_hearing_date, voting_deadline
                )
                
                # Break if both dates found
                if confirmation_hearing_date and voting_deadline:
                    print("✅ Both deadlines found, stopping PDF processing")
                    break

            except Exception as e:
                print(f"❌ PROCESSING ERROR: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    print(f"🧹 Cleaning up temp file: {tmp_path}")
                    try:
                        os.unlink(tmp_path)
                    except Exception as e:
                        print(f"⚠️ Failed to delete temp file: {str(e)}")
                
    print("\n=== PROCESSING COMPLETE ===")
    
    # Clean up old vector stores
    base_vector_dir = os.path.dirname(rag.vector_store_dir)
    rag.cleanup_old_vector_stores(base_vector_dir, max_stores=5)
    
    return confirmation_hearing_date, voting_deadline

# Helper function to extract the PDF processing code (to avoid repetition)
def process_pdf_and_extract_dates(pdf_url, rag, fingerprint_dir, current_hearing_date, current_voting_deadline):
    """Process a single PDF to extract dates"""
    # Use passed-in values instead of globals
    confirmation_hearing_date = current_hearing_date
    voting_deadline = current_voting_deadline
    
    tmp_path = None
    
    # Download PDF once
    print(f"⬇️ Downloading PDF from: {pdf_url}")
    response = requests.get(pdf_url)
    response.raise_for_status()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
        print(f"💾 Saved temp PDF: {tmp_path}")

    # Process with RAG (with fingerprinting) only once
    print("🔧 Ingesting PDF into RAG...")
    rag.ingest(tmp_path, fingerprint_dir=fingerprint_dir)
    
    # Extract confirmation hearing date if not found yet
    if not confirmation_hearing_date:
        print("❓ Querying RAG: 'What is the Confirmation Hearing Date in this PDF?'")
        answer = rag.ask("What is the Confirmation Hearing Date in this PDF?")
        print(f"📄 RAW RAG ANSWER: {answer}")
        
        date_str = extract_hearing_date_with_gpt(str(answer))
        print(f"📅 EXTRACTED CONFIRMATION DATE: {date_str}")
        
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
            print(f"✅ VALID CONFIRMATION DATE FOUND: {date_str}")
            confirmation_hearing_date = date_str
    
    # Extract voting deadline if not found yet
    if not voting_deadline:
        print("❓ Querying RAG: 'What is the Voting Deadline in this PDF?'")
        answer = rag.ask("What is the Voting Deadline or Ballot Deadline in this PDF? When must votes be submitted by?")
        print(f"📄 RAW RAG ANSWER: {answer}")
        
        date_str = extract_voting_deadline_with_gpt(str(answer))
        print(f"📅 EXTRACTED VOTING DEADLINE: {date_str}")
        
        if re.match(r'\d{1,2}/\d{1,2}/\d{4}', date_str):
            print(f"✅ VALID VOTING DEADLINE FOUND: {date_str}")
            voting_deadline = date_str
    
    # Return the tmp_path and the potentially updated date variables
    return tmp_path, confirmation_hearing_date, voting_deadline
  
def extract_case_details(df):
    """Extracts structured details from docket entries."""
    log("Extracting case details from docket entries...")
    
    # Collect all descriptions into a list
    descriptions = df['docketentry_description'].dropna().astype(str).tolist()
    recap_urls = df['recapdocument_filepath_local'].dropna().astype(str).tolist() if 'recapdocument_filepath_local' in df.columns else []
    
    # Initialize variables
    court_location = "Unknown"
    judge_name, claims_agent, cro_result, fa_ib, dip1, dip2,committee_counsel1,committee_counsel2 ,committee_counsel_fa1,committee_counsel_fa2,liquidating_trustee,voting_deadline,confirmation_hearing_date = None, None, "TBD", "TBD","TBD", "TBD","TBD","TBD","TBD","TBD","TBD",None, None
    
    # Extract case number from all descriptions
    case_no = extract_case_number(" ".join(descriptions), recap_urls)
    if not case_no:
        log("WARNING: No case number extracted!")

    # fa_ib = extract_fa_ib_debtor(descriptions)
    fa_ib = "TBD"    
    # Extract DIP1 and DIP2 using new functions
    # dip1 = extract_dip1_with_openai(descriptions)
    # dip2 = extract_dip2_with_openai(descriptions)
    dip1 = "TBD"
    dip2 = "TBD"
    

    # Extract Committee Counsel
    committee_counsel1 = extract_committee_counsel1(descriptions)
    committee_counsel2 = extract_committee_counsel2(descriptions)
    

    # Extract Committee Financial Advisor
    committee_counsel_fa1 = extract_committee_financial_advisor1(descriptions)
    committee_counsel_fa2 = extract_committee_financial_advisor2(descriptions)
    
    # Extract both deadlines in one go
    confirmation_hearing_date, voting_deadline = extract_deadlines_from_disclosure_statements(df)

    # Extract liquidating trustee
    liquidating_trustee = extract_liquidating_trustee_from_plan_supplement(df)
    
    # Process individual entries for other fields
    for desc in descriptions:
        if "US Bankruptcy Court" in desc and court_location == "Unknown":
            court_part = desc.split('(', 1)[0].strip()
            # Extract all possible location parts
            matches = re.findall(r"(?<=,\s)([A-Za-z\s]+?)(?=\.|,|$)", court_part, re.IGNORECASE)
            # Check matches in reverse to find the first valid state
            for part in reversed(matches):
                cleaned_part = part.strip().lower()
                if cleaned_part in US_STATES:
                    court_location = part.strip().title()
                    break        
        if not judge_name:
            judge_match = JUDGE_PATTERN.search(desc)
            if judge_match:
                judge_name = judge_match.group(1).strip()
        
        if not claims_agent:
            claims_agent_match = CLAIMS_AGENT_PATTERN.search(desc)
            if claims_agent_match:
                claims_agent = claims_agent_match.group(1).strip()
        
        if cro_result == "TBD":
            cro_result = extract_cro_with_openai(descriptions)
                    
    return case_no, court_location, judge_name, claims_agent, cro_result, fa_ib, dip1, dip2,committee_counsel1,committee_counsel2,committee_counsel_fa1,committee_counsel_fa2,liquidating_trustee,voting_deadline, confirmation_hearing_date


def process_folder(folder_path, output_file):
    """Processes all CSV files in a folder and appends new data to the output file in real-time."""
    log(f"Processing folder: {folder_path}")
    
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        log(f"ERROR: Folder '{folder_path}' not found!")
        return  

    # Load existing records if the output file exists
    existing_entries = set()
    if os.path.exists(output_file):
        try:
            # Read CSV with explicit column names and error handling
            existing_df = pd.read_csv(
                output_file,
                dtype=str,
                usecols=["COMPANY NAME", "CASE NO."],  # Explicitly target these columns
                on_bad_lines="warn",  # Warn on parsing errors
                engine="python"
            ).fillna("")
            
            # Validate columns exist
            if "COMPANY NAME" not in existing_df.columns or "CASE NO." not in existing_df.columns:
                log("ERROR: Output file missing required columns 'COMPANY NAME' or 'CASE NO.'")
                existing_df = pd.DataFrame(columns=COLUMNS)
            else:
                # Normalize and create entries set
                existing_df["COMPANY NAME"] = existing_df["COMPANY NAME"].str.lower().str.strip()
                existing_df["CASE NO."] = existing_df["CASE NO."].str.lower().str.strip()
                existing_entries = set(zip(existing_df["COMPANY NAME"], existing_df["CASE NO."]))
                log(f"Loaded {len(existing_entries)} valid existing entries.")
                
        except Exception as e:
            log(f"CRITICAL ERROR READING OUTPUT FILE: {str(e)}")
            existing_entries = set()
    else:
        existing_df = pd.DataFrame(columns=COLUMNS)
        existing_entries = set()

    # Process CSV files in folder
    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue
            
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, on_bad_lines="skip")
        except Exception as e:
            log(f"ERROR reading {file}: {e}")
            continue

        # Extract and normalize company name from filename
        company_name = os.path.basename(file).replace(".csv", "").split(".")[0].replace("-", " ").title()
        normalized_company_name = company_name.lower().strip()
        
        # DEBUG: Print filename processing
        log(f"\nProcessing file: {file}")
        
        # Extract case number from file contents
        descriptions = df['docketentry_description'].dropna().astype(str).tolist()
        case_no = extract_case_number(" ".join(descriptions), [])
        
        # Normalize case number
        if case_no:
            case_no = re.sub(r'\s+', ' ', case_no.strip()).lower()
            log(f"  Extracted case number: '{case_no}'")
        else:
            case_no = "unknown"
            log("  WARNING: No case number found in file")
        
        # Create comparison key
        entry_key = (normalized_company_name, case_no)
        log(f"  Comparison key: {entry_key}")
        
        # Skip if already processed
        if entry_key in existing_entries:
            log(f"  SKIPPING - Already processed: {entry_key}")
            continue  
        
        # Extract case details
        log("  Extracting case details...")
        case_no, court, judge, claims, cro_result, fa_ib, dip1, dip2, committee_counsel1,committee_counsel2,committee_counsel_fa1,committee_counsel_fa2,liquidating_trustee,voting_deadline,confirmation_hearing_date = extract_case_details(df)

        # Normalize the final case number
        if case_no:
            final_case_no = re.sub(r'\s+', ' ', case_no.strip()).lower()
        else:
            final_case_no = "unknown"
        
        # Create final entry key
        final_entry_key = (normalized_company_name, final_case_no)
        log(f"  Final comparison key: {final_entry_key}")
        
        # Check if this final version is already processed
        if final_entry_key in existing_entries:
            log(f"  SKIPPING - Final version already processed: {final_entry_key}")
            continue
            
        # Add new entry
        new_entry = [
            "ACTIVE", company_name, final_case_no, court, None, None, 
            cro_result if cro_result != "TBD" else "TBD",
            dip1, dip2, fa_ib, committee_counsel1, committee_counsel2, committee_counsel_fa1, committee_counsel_fa2, claims, judge, 
            liquidating_trustee if liquidating_trustee else "TBD", voting_deadline, confirmation_hearing_date, None
        ]
        
        # Append to output file
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                writer.writerow(COLUMNS)
            writer.writerow(new_entry)
        
        # Update the existing entries set
        existing_entries.add(final_entry_key)
        log(f"  Added to processed entries set: {final_entry_key}")

    log("Folder processing complete.")
    
    
# Main execution
if __name__ == "__main__":
    main_folder = r"Responses"
    output_file = "NewTrackerSheet.csv"
    process_folder(main_folder, output_file)