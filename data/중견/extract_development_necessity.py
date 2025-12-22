import json
import re
from pathlib import Path

def extract_necessity_content(file_path):
    print(f"\n{'='*50}")
    print(f"Extracting Necessity Content from: {file_path.name}")
    print(f"{'='*50}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        content = data.get('text', '') or data.get('content', '')
        if not content and 'chunks' in data:
            content = " ".join(data['chunks'])
            
        # "필요성", "배경", "문제점" 등의 키워드 주변 텍스트 추출
        # 또는 "1. 연구과제의 필요성" 섹션 추출
        
        # 1. 섹션 헤더 찾기
        patterns = [
            r'연구과제의 필요성', 
            r'연구개발의 필요성',
            r'최종 목표 및 내용', # 샘플-발달연구는 이 섹션에 배경이 포함됨
            r'연구개발과제의 개요'
        ]
        
        extracted_texts = []
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                start_pos = match.start()
                # 다음 섹션이나 3000자까지 추출
                end_pos = start_pos + 3000
                extracted = content[start_pos:end_pos]
                extracted_texts.append(f"--- Section: {pattern} ---\n{extracted}...")
        
        if not extracted_texts:
            print("No specific sections found. Searching for keywords...")
            # 키워드 기반 문장 추출
            keywords = ["발달장애", "조기 진단", "사회적 비용", "골든타임", "생애주기", "뇌 가소성"]
            sentences = re.split(r'[.!?]\s+', content)
            relevant_sentences = []
            for s in sentences:
                if any(k in s for k in keywords):
                    relevant_sentences.append(s)
            
            print("\n[Relevant Sentences]:")
            for s in relevant_sentences[:10]:
                print(f"- {s.strip()}")
        else:
            for text in extracted_texts:
                print(text)
                
    except Exception as e:
        print(f"Error: {e}")

def main():
    base_path = Path("/home/juke/git/AI-CoScientist/data/processed_grants")
    target_file = base_path / "샘플-발달연구_compressed.json"
    
    if target_file.exists():
        extract_necessity_content(target_file)
    else:
        print(f"File not found: {target_file}")

if __name__ == "__main__":
    main()


