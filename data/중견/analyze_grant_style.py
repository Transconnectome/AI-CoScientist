import json
import re
from pathlib import Path

def analyze_text_style(text, sample_name):
    print(f"\n{'='*50}")
    print(f"Analyzing: {sample_name}")
    print(f"{'='*50}")
    
    # 텍스트 전처리 (불필요한 공백 제거)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 처음 1500자 출력 (도입부 스타일 확인)
    print(f"\n[Intro Excerpt (First 1500 chars)]:")
    print(text[:1500] + "...")
    
    # 문장 분석
    sentences = re.split(r'[.!?]\s+', text)
    avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    print(f"\n[Style Metrics]:")
    print(f"- Total Sentences: {len(sentences)}")
    print(f"- Avg Words per Sentence: {avg_sentence_len:.1f}")
    
    # 특정 키워드 패턴 확인 (설득적 표현)
    persuasive_patterns = [
        r"필요하다", r"중요하다", r"한계가 있다", r"극복", r"최초", r"핵심", 
        r"따라서", r"반면", r"기존 연구", r"차별성"
    ]
    
    print(f"\n[Key Persuasive Terms Frequency]:")
    for pattern in persuasive_patterns:
        count = len(re.findall(pattern, text))
        if count > 0:
            print(f"- {pattern}: {count}")

def main():
    base_path = Path("/home/juke/git/AI-CoScientist/data/processed_grants")
    files = [
        "샘플-brainlink_compressed.json",
        "샘플-발달연구_compressed.json"
    ]
    
    for filename in files:
        file_path = base_path / filename
        if not file_path.exists():
            print(f"File not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 'text' 필드 또는 'content' 필드 확인
                content = data.get('text', '') or data.get('content', '')
                if not content and 'chunks' in data:
                    content = " ".join(data['chunks'])
                
                if content:
                    analyze_text_style(content, filename)
                else:
                    print(f"No text content found in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()


