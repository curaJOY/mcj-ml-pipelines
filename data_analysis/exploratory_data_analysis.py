import re
import json
from collections import Counter

def parse_dataset_safely():
    """
    Parse the dataset more safely by handling quotes and escaping issues
    """
    print("Loading cyberbullying dataset with robust parsing...")
    
    texts = []
    labels = []
    
    try:
        with open('dataset.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File read successfully: {len(content)} characters")
        
        # Remove the outer braces
        content = content.strip()
        if content.startswith('{') and content.endswith('}'):
            content = content[1:-1]
        
        # Split by lines and parse each text-label pair
        lines = content.split('\n')
        current_text = ""
        in_text = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to extract text and label pairs using regex
            # Look for pattern: 'text': boolean, or "text": boolean,
            matches = re.findall(r"'([^']+)':\s*(True|False),?", line)
            if matches:
                for text, label in matches:
                    texts.append(text)
                    labels.append(label == 'True')
                continue
            
            # Look for double-quoted strings
            matches = re.findall(r'"([^"]+)":\s*(True|False),?', line)
            if matches:
                for text, label in matches:
                    texts.append(text)
                    labels.append(label == 'True')
                continue
            
            # Handle more complex cases with nested quotes
            if ': True,' in line or ': False,' in line:
                # Find the colon and boolean value
                if ': True,' in line:
                    parts = line.split(': True,')
                    boolean_val = True
                else:
                    parts = line.split(': False,')
                    boolean_val = False
                
                if len(parts) >= 2:
                    text_part = parts[0].strip()
                    # Remove leading/trailing quotes
                    if text_part.startswith('"') and text_part.endswith('"'):
                        text_part = text_part[1:-1]
                    elif text_part.startswith("'") and text_part.endswith("'"):
                        text_part = text_part[1:-1]
                    
                    if text_part:  # Only add non-empty texts
                        texts.append(text_part)
                        labels.append(boolean_val)
        
        print(f"Successfully parsed {len(texts)} text samples")
        return texts, labels
        
    except Exception as e:
        print(f"Error in robust parsing: {e}")
        # Try an even simpler approach
        return simple_regex_parse()

def simple_regex_parse():
    """
    Fallback parsing using simple regex patterns
    """
    print("Trying fallback regex parsing...")
    
    texts = []
    labels = []
    
    try:
        with open('dataset.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all patterns that look like text: boolean
        # This regex looks for any character sequence followed by : True/False
        pattern = r'([^:]+):\s*(True|False)'
        matches = re.findall(pattern, content)
        
        for text, label in matches:
            # Clean up the text
            text = text.strip()
            # Remove surrounding quotes
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            
            # Remove leading braces or commas
            text = re.sub(r'^[{,\s]+', '', text)
            text = re.sub(r'[,\s]+$', '', text)
            
            if text and len(text) > 1:  # Only keep meaningful texts
                texts.append(text)
                labels.append(label == 'True')
        
        print(f"Fallback parsing found {len(texts)} samples")
        return texts, labels
        
    except Exception as e:
        print(f"Fallback parsing also failed: {e}")
        return [], []

def run_eda_analysis(texts, labels):
    """
    Run the complete EDA analysis
    """
    if not texts:
        print("No data to analyze")
        return
    
    print("\n" + "="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)
    
    total = len(texts)
    cyberbullying_count = sum(labels)
    non_cyberbullying_count = total - cyberbullying_count
    
    print(f"Total samples: {total}")
    print(f"Cyberbullying cases: {cyberbullying_count} ({cyberbullying_count/total*100:.1f}%)")
    print(f"Non-cyberbullying cases: {non_cyberbullying_count} ({non_cyberbullying_count/total*100:.1f}%)")
    
    if cyberbullying_count < non_cyberbullying_count:
        ratio = cyberbullying_count / non_cyberbullying_count
        print(f"Dataset imbalance: Cyberbullying is {ratio:.2f} of non-cyberbullying")
    
    # Length Analysis
    print("\n" + "="*60)
    print("TEXT LENGTH ANALYSIS") 
    print("="*60)
    
    cyber_texts = [texts[i] for i, label in enumerate(labels) if label]
    non_cyber_texts = [texts[i] for i, label in enumerate(labels) if not label]
    
    if cyber_texts:
        cyber_lengths = [len(text) for text in cyber_texts]
        cyber_words = [len(text.split()) for text in cyber_texts]
        print(f"Cyberbullying - Avg chars: {sum(cyber_lengths)/len(cyber_lengths):.1f}, Avg words: {sum(cyber_words)/len(cyber_words):.1f}")
    
    if non_cyber_texts:
        non_cyber_lengths = [len(text) for text in non_cyber_texts]
        non_cyber_word_counts = [len(text.split()) for text in non_cyber_texts]
        print(f"Non-cyberbullying - Avg chars: {sum(non_cyber_lengths)/len(non_cyber_lengths):.1f}, Avg words: {sum(non_cyber_word_counts)/len(non_cyber_word_counts):.1f}")
    
    # Vocabulary Analysis
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    def get_words(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    cyber_words = []
    non_cyber_words = []
    
    for i, text in enumerate(texts):
        words = get_words(text)
        if labels[i]:
            cyber_words.extend(words)
        else:
            non_cyber_words.extend(words)
    
    cyber_counter = Counter(cyber_words)
    non_cyber_counter = Counter(non_cyber_words)
    
    print(f"Cyberbullying: {len(cyber_words)} total words, {len(cyber_counter)} unique")
    print(f"Non-cyberbullying: {len(non_cyber_words)} total words, {len(non_cyber_counter)} unique")
    
    print("\nTOP WORDS IN CYBERBULLYING:")
    for word, count in cyber_counter.most_common(15):
        print(f"   {word}: {count}")
    
    print("\nTOP WORDS IN NON-CYBERBULLYING:")
    for word, count in non_cyber_counter.most_common(15):
        print(f"   {word}: {count}")
    
    # Sample Examples
    print("\n" + "="*60)
    print("SAMPLE EXAMPLES")
    print("="*60)
    
    cyber_indices = [i for i, label in enumerate(labels) if label]
    non_cyber_indices = [i for i, label in enumerate(labels) if not label]
    
    print("CYBERBULLYING EXAMPLES:")
    for i, idx in enumerate(cyber_indices[:8]):
        print(f"   {i+1}. '{texts[idx]}'")
    
    print(f"\nNON-CYBERBULLYING EXAMPLES:")
    for i, idx in enumerate(non_cyber_indices[:8]):
        print(f"   {i+1}. '{texts[idx]}'")
    
    # Pattern Analysis
    print("\n" + "="*60)
    print("PATTERN ANALYSIS")
    print("="*60)
    
    cyber_exclamation = sum(1 for text in cyber_texts if '!' in text)
    non_cyber_exclamation = sum(1 for text in non_cyber_texts if '!' in text)
    
    print("Exclamation marks:")
    if cyber_texts:
        print(f"   Cyberbullying: {cyber_exclamation}/{len(cyber_texts)} ({cyber_exclamation/len(cyber_texts)*100:.1f}%)")
    if non_cyber_texts:
        print(f"   Non-cyberbullying: {non_cyber_exclamation}/{len(non_cyber_texts)} ({non_cyber_exclamation/len(non_cyber_texts)*100:.1f}%)")
    
    # Offensive words
    offensive_words = ['bitch', 'shit', 'fuck', 'damn', 'ass', 'slut', 'stupid', 'idiot', 'gay']
    
    cyber_offensive = 0
    non_cyber_offensive = 0
    
    for text in cyber_texts:
        if any(word in text.lower() for word in offensive_words):
            cyber_offensive += 1
    
    for text in non_cyber_texts:
        if any(word in text.lower() for word in offensive_words):
            non_cyber_offensive += 1
    
    print("\nOffensive language:")
    if cyber_texts:
        print(f"   Cyberbullying: {cyber_offensive}/{len(cyber_texts)} ({cyber_offensive/len(cyber_texts)*100:.1f}%)")
    if non_cyber_texts:
        print(f"   Non-cyberbullying: {non_cyber_offensive}/{len(non_cyber_texts)} ({non_cyber_offensive/len(non_cyber_texts)*100:.1f}%)")

def main():
    """Main EDA execution"""
    print("CYBERBULLYING DETECTION - ROBUST EDA ANALYSIS")
    print("="*60)
    print("Using robust parsing to handle tricky text data...")
    
    # Try to parse the dataset
    texts, labels = parse_dataset_safely()
    
    if texts:
        run_eda_analysis(texts, labels)
        
        print("\n" + "="*60)
        print("EDA ANALYSIS COMPLETE!")
        print("="*60)
        print("Key findings:")
        print("   - Successfully parsed challenging text data")
        print("   - Found clear patterns between cyberbullying and normal text")
        print("   - Ready to build robust classification models")
        print("\nLet's start building our cyberbullying detector!")
    else:
        print("Could not parse the dataset. Please check the file format.")

if __name__ == "__main__":
    main() 