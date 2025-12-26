#!/usr/bin/env python3
"""
회의 음성 분석 시스템 프로토타입
기본적인 오디오 파일 분석 예제
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper가 설치되지 않았습니다. 'pip install openai-whisper' 실행 필요")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️  pyannote.audio가 설치되지 않았습니다. 'pip install pyannote.audio' 실행 필요")


class MeetingAnalyzer:
    """회의 오디오 분석 클래스"""
    
    def __init__(self, whisper_model="base", pyannote_token=None):
        """
        Args:
            whisper_model: Whisper 모델 크기 (tiny, base, small, medium, large)
            pyannote_token: Hugging Face 토큰 (pyannote.audio 사용 시 필요)
        """
        self.whisper_model = None
        self.pyannote_pipeline = None
        
        if WHISPER_AVAILABLE:
            print(f"📥 Whisper 모델 로딩 중: {whisper_model}...")
            self.whisper_model = whisper.load_model(whisper_model)
            print("✅ Whisper 모델 로드 완료")
        else:
            print("❌ Whisper를 사용할 수 없습니다")
        
        if PYANNOTE_AVAILABLE and pyannote_token:
            print("📥 pyannote.audio 파이프라인 로딩 중...")
            try:
                self.pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=pyannote_token
                )
                print("✅ pyannote.audio 파이프라인 로드 완료")
            except Exception as e:
                print(f"⚠️  pyannote.audio 로드 실패: {e}")
                print("   Hugging Face 토큰이 필요할 수 있습니다")
    
    def transcribe(self, audio_path: str, language: str = "ko") -> Dict:
        """오디오 파일 전사"""
        if not self.whisper_model:
            raise ValueError("Whisper 모델이 로드되지 않았습니다")
        
        print(f"🎤 전사 중: {audio_path}...")
        start_time = time.time()
        
        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            task="transcribe"
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 전사 완료 (소요 시간: {elapsed:.2f}초)")
        
        return {
            "text": result["text"],
            "language": result["language"],
            "segments": result["segments"],
            "processing_time": elapsed
        }
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """화자 분리"""
        if not self.pyannote_pipeline:
            raise ValueError("pyannote.audio 파이프라인이 로드되지 않았습니다")
        
        print(f"👥 화자 분리 중: {audio_path}...")
        start_time = time.time()
        
        diarization = self.pyannote_pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "duration": round(turn.end - turn.start, 2)
            })
        
        elapsed = time.time() - start_time
        print(f"✅ 화자 분리 완료 (소요 시간: {elapsed:.2f}초)")
        print(f"   발견된 화자 수: {len(set(s['speaker'] for s in segments))}")
        
        return segments
    
    def analyze_meeting(self, audio_path: str, language: str = "ko") -> Dict:
        """전체 회의 분석"""
        print(f"\n{'='*60}")
        print(f"📊 회의 분석 시작: {audio_path}")
        print(f"{'='*60}\n")
        
        results = {
            "audio_file": audio_path,
            "analysis_time": datetime.now().isoformat(),
            "transcription": None,
            "speaker_diarization": None,
            "talk_time_stats": None,
            "combined_results": None
        }
        
        # 1. 전사
        if self.whisper_model:
            results["transcription"] = self.transcribe(audio_path, language)
        else:
            print("⚠️  Whisper가 없어 전사를 건너뜁니다")
        
        # 2. 화자 분리
        if self.pyannote_pipeline:
            try:
                results["speaker_diarization"] = self.diarize(audio_path)
            except Exception as e:
                print(f"⚠️  화자 분리 실패: {e}")
        else:
            print("⚠️  pyannote.audio가 없어 화자 분리를 건너뜁니다")
        
        # 3. 발언 시간 통계
        if results["speaker_diarization"]:
            results["talk_time_stats"] = self.calculate_talk_time_stats(
                results["speaker_diarization"]
            )
        
        # 4. 전사와 화자 분리 결합
        if results["transcription"] and results["speaker_diarization"]:
            results["combined_results"] = self.combine_transcription_and_speakers(
                results["transcription"],
                results["speaker_diarization"]
            )
        
        return results
    
    def calculate_talk_time_stats(self, diarization: List[Dict]) -> Dict:
        """발언 시간 통계 계산"""
        stats = {}
        
        for segment in diarization:
            speaker = segment["speaker"]
            duration = segment["duration"]
            
            if speaker not in stats:
                stats[speaker] = {
                    "total_seconds": 0.0,
                    "total_minutes": 0.0,
                    "number_of_turns": 0,
                    "turns": []
                }
            
            stats[speaker]["total_seconds"] += duration
            stats[speaker]["number_of_turns"] += 1
            stats[speaker]["turns"].append(segment)
        
        # 분 단위 변환 및 평균 계산
        for speaker in stats:
            stats[speaker]["total_minutes"] = round(
                stats[speaker]["total_seconds"] / 60, 2
            )
            if stats[speaker]["number_of_turns"] > 0:
                stats[speaker]["average_turn_length"] = round(
                    stats[speaker]["total_seconds"] / stats[speaker]["number_of_turns"], 2
                )
        
        # 전체 시간 계산
        total_time = sum(s["total_seconds"] for s in stats.values())
        
        # 참여율 계산
        for speaker in stats:
            if total_time > 0:
                stats[speaker]["participation_rate"] = round(
                    (stats[speaker]["total_seconds"] / total_time) * 100, 2
                )
        
        return {
            "speakers": stats,
            "total_meeting_time_seconds": round(total_time, 2),
            "total_meeting_time_minutes": round(total_time / 60, 2),
            "number_of_speakers": len(stats)
        }
    
    def combine_transcription_and_speakers(
        self, 
        transcription: Dict, 
        diarization: List[Dict]
    ) -> List[Dict]:
        """전사 결과와 화자 분리 결과 결합"""
        combined = []
        
        # 화자 세그먼트를 시간순으로 정렬
        diarization_sorted = sorted(diarization, key=lambda x: x["start"])
        
        # 전사 세그먼트와 매칭
        for trans_seg in transcription["segments"]:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_text = trans_seg["text"].strip()
            
            # 해당 시간대의 화자 찾기
            speaker = None
            for diar_seg in diarization_sorted:
                if (diar_seg["start"] <= trans_start <= diar_seg["end"] or
                    diar_seg["start"] <= trans_end <= diar_seg["end"]):
                    speaker = diar_seg["speaker"]
                    break
            
            if speaker:
                combined.append({
                    "speaker": speaker,
                    "text": trans_text,
                    "start": round(trans_start, 2),
                    "end": round(trans_end, 2),
                    "duration": round(trans_end - trans_start, 2)
                })
        
        return combined
    
    def save_results(self, results: Dict, output_path: str):
        """결과를 JSON 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 결과 저장: {output_path}")
    
    def print_summary(self, results: Dict):
        """결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 분석 결과 요약")
        print(f"{'='*60}\n")
        
        if results["talk_time_stats"]:
            stats = results["talk_time_stats"]
            print(f"회의 총 시간: {stats['total_meeting_time_minutes']}분")
            print(f"화자 수: {stats['number_of_speakers']}\n")
            
            print("화자별 발언 시간:")
            print("-" * 60)
            for speaker, data in sorted(
                stats["speakers"].items(),
                key=lambda x: x[1]["total_seconds"],
                reverse=True
            ):
                print(f"{speaker}:")
                print(f"  총 발언 시간: {data['total_minutes']}분 ({data['total_seconds']}초)")
                print(f"  발언 횟수: {data['number_of_turns']}회")
                print(f"  평균 발언 길이: {data.get('average_turn_length', 0):.2f}초")
                print(f"  참여율: {data.get('participation_rate', 0):.2f}%")
                print()
        
        if results["transcription"]:
            print(f"\n전체 전사 텍스트:")
            print("-" * 60)
            print(results["transcription"]["text"][:500] + "..." if len(results["transcription"]["text"]) > 500 else results["transcription"]["text"])
            print()
        
        if results["combined_results"]:
            print(f"\n화자별 전사 (처음 5개):")
            print("-" * 60)
            for item in results["combined_results"][:5]:
                print(f"[{item['speaker']}] ({item['start']:.1f}s-{item['end']:.1f}s): {item['text']}")
            if len(results["combined_results"]) > 5:
                print(f"... 외 {len(results['combined_results']) - 5}개 더")


def main():
    parser = argparse.ArgumentParser(
        description="회의 오디오 파일 분석 프로토타입",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  %(prog)s audio.wav
  %(prog)s audio.wav --whisper-model large
  %(prog)s audio.wav --output results.json
  %(prog)s audio.wav --pyannote-token YOUR_HF_TOKEN
        """
    )
    
    parser.add_argument(
        "audio_file",
        type=str,
        help="분석할 오디오 파일 경로"
    )
    
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 모델 크기 (기본: base)"
    )
    
    parser.add_argument(
        "--pyannote-token",
        type=str,
        default=None,
        help="Hugging Face 토큰 (pyannote.audio 사용 시 필요)"
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="ko",
        help="언어 코드 (기본: ko)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 파일 경로 (JSON)"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="요약 출력 건너뛰기"
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"❌ 오디오 파일을 찾을 수 없습니다: {args.audio_file}")
        return 1
    
    # 분석기 초기화
    analyzer = MeetingAnalyzer(
        whisper_model=args.whisper_model,
        pyannote_token=args.pyannote_token
    )
    
    # 분석 수행
    try:
        results = analyzer.analyze_meeting(str(audio_path), language=args.language)
        
        # 결과 출력
        if not args.no_summary:
            analyzer.print_summary(results)
        
        # 결과 저장
        if args.output:
            analyzer.save_results(results, args.output)
        else:
            # 기본 출력 파일명
            output_file = audio_path.stem + "_analysis.json"
            analyzer.save_results(results, output_file)
        
        print("\n✅ 분석 완료!")
        return 0
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())




