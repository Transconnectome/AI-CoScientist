# 다이어그램 파이프라인 API 테스트 결과

테스트 일시: 2025-12-14 21:36:34

---

## 요약

| 모델 | 상태 | 소요시간 | 출력 |
|------|------|----------|------|
| DALL-E 3 | ✅ 성공 | 39.9초 | api_openai_dalle3.png |
| Gemini 2.0 Flash | ❌ 실패 | 1.7초 | 텍스트 응답 반환됨 |
| Kimi K2 (Moonshot) | ❌ 실패 | - | - |
| DeepSeek | ✅ 성공 | 60.6초 | api_deepseek_diagram.png |

---

## 상세 결과

### DALL-E 3

```json
{
  "model": "DALL-E 3",
  "output": "/Users/jiookcha/Desktop/_중견/claudedocs/diagram_pipelines/test_results/api_openai_dalle3.png",
  "time": 39.916126012802124,
  "success": true
}
```

### Gemini 2.0 Flash

```json
{
  "model": "Gemini 2.0 Flash",
  "note": "텍스트 응답 반환됨",
  "time": 1.6956720352172852,
  "success": false
}
```

### Kimi K2 (Moonshot)

```json
{
  "model": "Kimi K2 (Moonshot)",
  "error": "401 Client Error: Unauthorized for url: https://api.moonshot.cn/v1/chat/completions",
  "success": false
}
```

### DeepSeek

```json
{
  "model": "DeepSeek",
  "output": "/Users/jiookcha/Desktop/_중견/claudedocs/diagram_pipelines/test_results/api_deepseek_diagram.png",
  "code": "/Users/jiookcha/Desktop/_중견/claudedocs/diagram_pipelines/test_results/api_deepseek_code.py",
  "time": 60.613688707351685,
  "success": true
}
```

---

## 생성된 파일

- `api_openai_dalle3.png` (2493.1KB)
- `api_openai_dalle3_prompt.txt` (2.2KB)
- `api_deepseek_diagram.png` (227.2KB)
- `api_test_report.md` (0B)
- `api_deepseek_code.py` (5.8KB)
- `api_google_gemini_response.txt` (600B)
