#!/usr/bin/env python3
"""
PDF extractor using built-in urllib and basic PDF parsing
This avoids requiring external dependencies
"""

def extract_text_basic(pdf_path):
    """
    Basic PDF text extraction without external libraries
    This reads the raw PDF stream and extracts visible text
    """
    try:
        with open(pdf_path, 'rb') as file:
            content = file.read()
            
        # Convert to string (errors='ignore' to handle binary data)
        text = content.decode('latin-1', errors='ignore')
        
        # Extract text between common PDF text markers
        import re
        
        # Look for text objects in PDF
        text_objects = re.findall(r'\((.*?)\)', text)
        
        # Also look for hex-encoded text
        extracted_text = '\n'.join(text_objects)
        
        # Clean up common PDF artifacts
        extracted_text = extracted_text.replace('\\n', '\n')
        extracted_text = extracted_text.replace('\\r', '')
        extracted_text = extracted_text.replace('\\(', '(')
        extracted_text = extracted_text.replace('\\)', ')')
        
        return extracted_text
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    pdf_path = "paper.pdf"
    output_path = "paper.txt"
    
    print("Extracting PDF text using basic method...")
    text = extract_text_basic(pdf_path)
    
    if text:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"✓ Extracted {len(text)} characters to {output_path}")
        print(f"✓ First 500 characters preview:")
        print("-" * 60)
        print(text[:500])
    else:
        print("✗ Extraction failed")
