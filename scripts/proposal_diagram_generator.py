#!/usr/bin/env python3
"""
NRF 중견연구 제안서 다이어그램 자동 생성기
==========================================

명령어 한 줄로 제안서용 필수 그림 4종을 자동 생성합니다:
- Fig 1: 문제-갭-가설-기여 인포그래픽
- Fig 2: 방법 파이프라인/워크플로
- Fig 3: Aim별 실험/데이터 흐름 + 성공기준
- Fig 4: Gantt + 마일스톤 + Go/No-Go

사용법:
    # 기본 사용 (대화형)
    python scripts/proposal_diagram_generator.py

    # 설정 파일 사용
    python scripts/proposal_diagram_generator.py --config proposal_config.yaml

    # 직접 입력
    python scripts/proposal_diagram_generator.py --title "생애주기 뇌영상-유전체 AI" \
        --aims "Aim1:데이터 통합,Aim2:AI 모델 개발,Aim3:임상 검증"

필요 패키지:
    pip install matplotlib numpy pyyaml
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
    from matplotlib.lines import Line2D
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError:
    print("❌ matplotlib 미설치. 설치: pip install matplotlib numpy")
    sys.exit(1)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ========== 기본 설정 ==========
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "proposal_diagrams"

# 색상 팔레트 (NRF 제안서 스타일)
COLORS = {
    'primary': '#1976d2',      # 파랑 (메인)
    'secondary': '#f57c00',    # 주황 (강조)
    'success': '#388e3c',      # 초록 (성공/완료)
    'warning': '#fbc02d',      # 노랑 (주의)
    'danger': '#d32f2f',       # 빨강 (리스크)
    'info': '#7b1fa2',         # 보라 (정보)
    'light_blue': '#e3f2fd',
    'light_orange': '#fff3e0',
    'light_green': '#e8f5e9',
    'light_purple': '#f3e5f5',
    'light_pink': '#fce4ec',
    'gray': '#9e9e9e',
    'dark_gray': '#424242',
    'white': '#ffffff',
}


class ProposalDiagramGenerator:
    """NRF 제안서 다이어그램 생성기"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 제안서 설정 딕셔너리
                - title: 과제명
                - title_en: 영문 과제명
                - aims: Aim 목록 [{name, description, methods, success_criteria}]
                - timeline: 연구 기간 (년)
                - problem: 연구 문제
                - gap: 기존 연구 갭
                - hypothesis: 핵심 가설
                - contribution: 기대 기여
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', DEFAULT_OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 한글 폰트 설정
        self._setup_korean_font()

    def _setup_korean_font(self):
        """한글 폰트 설정 (koreanize-matplotlib 사용)"""
        try:
            import koreanize_matplotlib
            # koreanize_matplotlib이 자동으로 NanumGothic 폰트 설정
            print("✅ 한글 폰트 설정 완료 (NanumGothic)")
        except ImportError:
            # koreanize_matplotlib 없으면 기존 방식으로 시도
            import matplotlib.font_manager as fm
            korean_fonts = [
                'NanumGothic', 'NanumBarunGothic', 'Malgun Gothic',
                'AppleGothic', 'Noto Sans KR', 'Source Han Sans'
            ]
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            for font in korean_fonts:
                if font in available_fonts:
                    plt.rcParams['font.family'] = font
                    print(f"✅ 한글 폰트 설정 완료 ({font})")
                    break
            else:
                plt.rcParams['font.family'] = 'DejaVu Sans'
                print("⚠️ 한글 폰트 미발견 - DejaVu Sans 사용")

        plt.rcParams['axes.unicode_minus'] = False

    def generate_all(self) -> List[Path]:
        """모든 그림 4종 생성"""
        print("\n" + "=" * 60)
        print("🎨 NRF 제안서 다이어그램 자동 생성")
        print(f"📁 출력 폴더: {self.output_dir}")
        print("=" * 60)

        generated_files = []

        # Fig 1: 문제-갭-가설-기여
        fig1_path = self.generate_fig1_problem_gap_hypothesis()
        generated_files.append(fig1_path)

        # Fig 2: 방법 파이프라인
        fig2_path = self.generate_fig2_method_pipeline()
        generated_files.append(fig2_path)

        # Fig 3: Aim별 실험 흐름
        fig3_path = self.generate_fig3_aim_workflow()
        generated_files.append(fig3_path)

        # Fig 4: Gantt 차트
        fig4_path = self.generate_fig4_gantt_chart()
        generated_files.append(fig4_path)

        print("\n" + "=" * 60)
        print("✅ 모든 다이어그램 생성 완료!")
        print("=" * 60)
        for f in generated_files:
            print(f"  📄 {f}")

        return generated_files

    def generate_fig1_problem_gap_hypothesis(self) -> Path:
        """Fig 1: 문제-갭-가설-기여 인포그래픽"""
        print("\n📊 Fig 1: 문제-갭-가설-기여 생성 중...")

        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_facecolor('white')

        # 제목
        title = self.config.get('title', '연구 과제')
        ax.text(6, 7.5, title, fontsize=14, fontweight='bold', ha='center')

        # 4개 박스 위치 (이모지 사용 금지 - 공식 제안서 스타일)
        positions = [
            (1.5, 5.5, COLORS['danger'], '문제 (Problem)', self.config.get('problem', '현재 미해결 문제')),
            (4.5, 5.5, COLORS['warning'], '갭 (Gap)', self.config.get('gap', '기존 연구의 한계')),
            (7.5, 5.5, COLORS['primary'], '가설 (Hypothesis)', self.config.get('hypothesis', '핵심 연구 가설')),
            (10.5, 5.5, COLORS['success'], '기여 (Contribution)', self.config.get('contribution', '기대 연구 기여')),
        ]

        for x, y, color, label, content in positions:
            # 박스
            box = FancyBboxPatch(
                (x - 1.3, y - 1.8), 2.6, 3.2,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=color + '20', edgecolor=color, linewidth=2
            )
            ax.add_patch(box)

            # 레이블
            ax.text(x, y + 1.0, label, fontsize=10, fontweight='bold',
                   ha='center', color=color)

            # 내용 (줄바꿈 처리)
            wrapped_content = self._wrap_text(content, 15)
            ax.text(x, y - 0.3, wrapped_content, fontsize=9, ha='center', va='top',
                   wrap=True)

        # 화살표 연결
        arrow_props = dict(arrowstyle='->', color=COLORS['dark_gray'], lw=2)
        for i in range(3):
            start_x = positions[i][0] + 1.3
            end_x = positions[i + 1][0] - 1.3
            ax.annotate('', xy=(end_x, 5.5), xytext=(start_x, 5.5),
                       arrowprops=arrow_props)

        # 하단 요약
        summary_box = FancyBboxPatch(
            (1, 0.5), 10, 2.5,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=COLORS['light_blue'], edgecolor=COLORS['primary'], linewidth=2
        )
        ax.add_patch(summary_box)

        summary_text = f"본 연구는 {self.config.get('problem', '문제')}를 해결하기 위해 " \
                      f"{self.config.get('hypothesis', '가설')}을 검증하여 " \
                      f"{self.config.get('contribution', '기여')}를 달성하고자 함"
        ax.text(6, 1.75, self._wrap_text(summary_text, 60), fontsize=10,
               ha='center', va='center', style='italic')

        # 저장
        output_path = self.output_dir / "fig1_problem_gap_hypothesis.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"  ✅ {output_path}")
        return output_path

    def generate_fig2_method_pipeline(self) -> Path:
        """Fig 2: 방법 파이프라인/워크플로"""
        print("\n📊 Fig 2: 방법 파이프라인 생성 중...")

        aims = self.config.get('aims', [
            {'name': 'Aim 1', 'description': '데이터 수집 및 전처리'},
            {'name': 'Aim 2', 'description': '모델 개발'},
            {'name': 'Aim 3', 'description': '검증 및 적용'},
        ])

        fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_facecolor('white')

        # 제목
        ax.text(7, 7.5, '연구 방법 파이프라인', fontsize=14, fontweight='bold', ha='center')

        # 입력 단계
        input_box = FancyBboxPatch(
            (0.5, 4.5), 2, 2,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=COLORS['light_green'], edgecolor=COLORS['success'], linewidth=2
        )
        ax.add_patch(input_box)
        ax.text(1.5, 5.5, '입력\nInput', fontsize=10, fontweight='bold',
               ha='center', va='center')

        # Aim 블록들
        aim_colors = [COLORS['light_blue'], COLORS['light_orange'], COLORS['light_purple']]
        aim_borders = [COLORS['primary'], COLORS['secondary'], COLORS['info']]

        n_aims = len(aims)
        aim_width = 2.5
        aim_spacing = (10 - n_aims * aim_width) / (n_aims + 1)

        for i, aim in enumerate(aims):
            x = 3 + aim_spacing + i * (aim_width + aim_spacing)

            # Aim 박스
            aim_box = FancyBboxPatch(
                (x, 3.5), aim_width, 4,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=aim_colors[i % 3], edgecolor=aim_borders[i % 3], linewidth=2
            )
            ax.add_patch(aim_box)

            # Aim 제목
            ax.text(x + aim_width/2, 7.0, aim.get('name', f'Aim {i+1}'),
                   fontsize=11, fontweight='bold', ha='center',
                   color=aim_borders[i % 3])

            # Aim 설명
            desc = aim.get('description', '')
            ax.text(x + aim_width/2, 5.5, self._wrap_text(desc, 12),
                   fontsize=9, ha='center', va='center')

            # 방법들 (있으면)
            methods = aim.get('methods', [])
            for j, method in enumerate(methods[:3]):  # 최대 3개
                ax.text(x + aim_width/2, 4.5 - j * 0.5, f"• {method}",
                       fontsize=8, ha='center', va='top')

            # 화살표 (이전 요소에서)
            if i == 0:
                ax.annotate('', xy=(x, 5.5), xytext=(2.5, 5.5),
                           arrowprops=dict(arrowstyle='->', color=COLORS['dark_gray'], lw=2))
            else:
                prev_x = 3 + aim_spacing + (i-1) * (aim_width + aim_spacing) + aim_width
                ax.annotate('', xy=(x, 5.5), xytext=(prev_x, 5.5),
                           arrowprops=dict(arrowstyle='->', color=COLORS['dark_gray'], lw=2))

        # 출력 단계
        output_x = 3 + aim_spacing + (n_aims - 1) * (aim_width + aim_spacing) + aim_width + 0.5
        output_box = FancyBboxPatch(
            (output_x, 4.5), 2, 2,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor=COLORS['light_pink'], edgecolor=COLORS['danger'], linewidth=2
        )
        ax.add_patch(output_box)
        ax.text(output_x + 1, 5.5, '출력\nOutput', fontsize=10, fontweight='bold',
               ha='center', va='center')

        # 마지막 화살표
        last_aim_x = 3 + aim_spacing + (n_aims - 1) * (aim_width + aim_spacing) + aim_width
        ax.annotate('', xy=(output_x, 5.5), xytext=(last_aim_x, 5.5),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark_gray'], lw=2))

        # 하단 범례
        ax.text(7, 1, '각 Aim은 병렬/순차 수행 가능하며, 피드백 루프를 통해 반복 최적화',
               fontsize=9, ha='center', style='italic', color=COLORS['gray'])

        # 저장
        output_path = self.output_dir / "fig2_method_pipeline.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"  ✅ {output_path}")
        return output_path

    def generate_fig3_aim_workflow(self) -> Path:
        """Fig 3: Aim별 실험/데이터 흐름 + 성공기준"""
        print("\n📊 Fig 3: Aim별 실험 흐름 생성 중...")

        aims = self.config.get('aims', [
            {'name': 'Aim 1', 'description': '데이터 수집',
             'success_criteria': '데이터셋 N>1000'},
            {'name': 'Aim 2', 'description': '모델 개발',
             'success_criteria': 'AUC>0.85'},
            {'name': 'Aim 3', 'description': '임상 검증',
             'success_criteria': 'Sensitivity>90%'},
        ])

        n_aims = len(aims)
        fig, axes = plt.subplots(1, n_aims, figsize=(4 * n_aims, 8), dpi=150)

        if n_aims == 1:
            axes = [axes]

        aim_colors = [COLORS['primary'], COLORS['secondary'], COLORS['info']]

        for i, (ax, aim) in enumerate(zip(axes, aims)):
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 8)
            ax.axis('off')
            ax.set_facecolor('white')

            color = aim_colors[i % 3]

            # Aim 헤더
            header_box = FancyBboxPatch(
                (0.2, 7), 3.6, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color, edgecolor=color, linewidth=2
            )
            ax.add_patch(header_box)
            ax.text(2, 7.4, aim.get('name', f'Aim {i+1}'),
                   fontsize=12, fontweight='bold', ha='center', color='white')

            # 설명
            ax.text(2, 6.5, aim.get('description', ''), fontsize=10, ha='center')

            # 워크플로 단계들
            steps = aim.get('steps', ['데이터 수집', '전처리', '분석', '검증'])
            step_height = 4.5 / len(steps)

            for j, step in enumerate(steps):
                y = 5.5 - j * step_height

                step_box = FancyBboxPatch(
                    (0.5, y - 0.3), 3, 0.6,
                    boxstyle="round,pad=0.02,rounding_size=0.1",
                    facecolor=color + '30', edgecolor=color, linewidth=1.5
                )
                ax.add_patch(step_box)
                ax.text(2, y, step, fontsize=9, ha='center', va='center')

                # 화살표
                if j < len(steps) - 1:
                    ax.annotate('', xy=(2, y - 0.4), xytext=(2, y - step_height + 0.4),
                               arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

            # 성공 기준
            success_y = 1.2
            success_box = FancyBboxPatch(
                (0.3, success_y - 0.5), 3.4, 1,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=COLORS['light_green'], edgecolor=COLORS['success'], linewidth=2
            )
            ax.add_patch(success_box)
            ax.text(2, success_y + 0.2, '성공 기준', fontsize=9,
                   fontweight='bold', ha='center', color=COLORS['success'])
            ax.text(2, success_y - 0.2, aim.get('success_criteria', 'TBD'),
                   fontsize=9, ha='center')

        # 전체 제목
        fig.suptitle('Aim별 실험 흐름 및 성공 기준', fontsize=14, fontweight='bold', y=0.98)

        # 저장
        output_path = self.output_dir / "fig3_aim_workflow.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"  ✅ {output_path}")
        return output_path

    def generate_fig4_gantt_chart(self) -> Path:
        """Fig 4: Gantt 차트 + 마일스톤 + Go/No-Go"""
        print("\n📊 Fig 4: Gantt 차트 생성 중...")

        timeline_years = self.config.get('timeline', 3)
        aims = self.config.get('aims', [
            {'name': 'Aim 1', 'start': 0, 'end': 12},
            {'name': 'Aim 2', 'start': 6, 'end': 24},
            {'name': 'Aim 3', 'start': 18, 'end': 36},
        ])

        milestones = self.config.get('milestones', [
            {'month': 12, 'name': 'M1: 데이터 구축 완료'},
            {'month': 24, 'name': 'M2: 모델 개발 완료'},
            {'month': 36, 'name': 'M3: 검증 완료'},
        ])

        go_nogo = self.config.get('go_nogo', [
            {'month': 12, 'criteria': 'Go: 데이터 N>500'},
            {'month': 24, 'criteria': 'Go: AUC>0.8'},
        ])

        fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

        total_months = timeline_years * 12
        n_aims = len(aims)

        # Y축 설정
        ax.set_ylim(-1, n_aims + 2)
        ax.set_xlim(-2, total_months + 2)

        # X축 (월/년)
        ax.set_xlabel('연구 기간 (개월)', fontsize=11)

        # 년도 표시
        for year in range(timeline_years + 1):
            month = year * 12
            ax.axvline(x=month, color=COLORS['gray'], linestyle='--', alpha=0.3)
            ax.text(month, n_aims + 1.5, f'{year + 1}년차' if year < timeline_years else '종료',
                   ha='center', fontsize=10, fontweight='bold')

        # Aim 바
        aim_colors = [COLORS['primary'], COLORS['secondary'], COLORS['info'],
                     COLORS['success'], COLORS['warning']]

        for i, aim in enumerate(aims):
            y = n_aims - 1 - i
            start = aim.get('start', i * 6)
            end = aim.get('end', (i + 1) * 12)
            duration = end - start

            color = aim_colors[i % len(aim_colors)]

            # Aim 바
            bar = FancyBboxPatch(
                (start, y - 0.3), duration, 0.6,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color, edgecolor=color, linewidth=0
            )
            ax.add_patch(bar)

            # Aim 이름
            ax.text(-1, y, aim.get('name', f'Aim {i+1}'), fontsize=10,
                   fontweight='bold', ha='right', va='center')

            # 기간 표시
            ax.text(start + duration/2, y, f'{start}-{end}M',
                   fontsize=8, ha='center', va='center', color='white', fontweight='bold')

        # 마일스톤
        for ms in milestones:
            month = ms.get('month', 12)
            name = ms.get('name', 'Milestone')

            ax.plot(month, -0.5, 'v', markersize=15, color=COLORS['danger'])
            ax.text(month, -0.8, name, fontsize=8, ha='center', va='top',
                   color=COLORS['danger'], rotation=45)

        # Go/No-Go
        for gng in go_nogo:
            month = gng.get('month', 12)
            criteria = gng.get('criteria', 'Go/No-Go')

            ax.axvline(x=month, color=COLORS['success'], linestyle='-', linewidth=2, alpha=0.7)
            ax.plot(month, n_aims + 0.6, 's', markersize=10, color=COLORS['success'])
            ax.text(month + 0.5, n_aims + 0.5, criteria, fontsize=7,
                   ha='left', va='center', color=COLORS['success'])

        # 범례
        legend_elements = [
            mpatches.Patch(facecolor=COLORS['primary'], label='연구 수행 기간'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['danger'],
                  markersize=10, label='마일스톤'),
            Line2D([0], [0], color=COLORS['success'], linewidth=2, label='Go/No-Go 판단점'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # 제목
        ax.set_title(f'연구 추진 일정 ({timeline_years}년)', fontsize=14, fontweight='bold', pad=20)

        # 축 설정
        ax.set_yticks([])
        ax.set_xticks(range(0, total_months + 1, 6))
        ax.set_xticklabels([f'{m}M' for m in range(0, total_months + 1, 6)])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 저장
        output_path = self.output_dir / "fig4_gantt_chart.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        print(f"  ✅ {output_path}")
        return output_path

    def _wrap_text(self, text: str, max_chars: int) -> str:
        """텍스트 줄바꿈"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_chars:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    if not YAML_AVAILABLE:
        print("❌ pyyaml 미설치. 설치: pip install pyyaml")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_config_from_json(config_path: str) -> Dict[str, Any]:
    """JSON 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def interactive_config() -> Dict[str, Any]:
    """대화형 설정 입력"""
    print("\n" + "=" * 60)
    print("🎨 NRF 제안서 다이어그램 생성기 - 대화형 모드")
    print("=" * 60)

    config = {}

    # 기본 정보
    config['title'] = input("\n📋 과제명 (국문): ") or "생애주기 뇌영상-유전체 AI 모델"
    config['title_en'] = input("📋 과제명 (영문): ") or "LifeSpan Brain Imaging-Genomics AI"

    # 문제-갭-가설-기여
    print("\n--- Fig 1: 문제-갭-가설-기여 ---")
    config['problem'] = input("🔴 연구 문제: ") or "발달-노화 연속체의 뇌 변화 패턴 미규명"
    config['gap'] = input("🟡 기존 연구 갭: ") or "단면적 연구 중심, 종단/다중모달 통합 부재"
    config['hypothesis'] = input("🔵 핵심 가설: ") or "AI 기반 통합 모델로 뇌 발달-노화 예측 가능"
    config['contribution'] = input("🟢 기대 기여: ") or "생애주기 뇌 건강 예측 모델 및 바이오마커 발굴"

    # Aims
    print("\n--- Fig 2-3: 연구 Aims ---")
    n_aims = int(input("Aim 개수 (기본 3): ") or "3")

    config['aims'] = []
    for i in range(n_aims):
        print(f"\n  Aim {i + 1}:")
        aim = {
            'name': f"Aim {i + 1}",
            'description': input(f"    설명: ") or f"연구내용 {i + 1}",
            'success_criteria': input(f"    성공 기준: ") or f"지표 > 기준값",
            'start': i * 12,
            'end': (i + 1) * 12 + 6,
        }

        methods_input = input(f"    방법들 (쉼표 구분): ") or ""
        if methods_input:
            aim['methods'] = [m.strip() for m in methods_input.split(',')]

        steps_input = input(f"    단계들 (쉼표 구분): ") or ""
        if steps_input:
            aim['steps'] = [s.strip() for s in steps_input.split(',')]

        config['aims'].append(aim)

    # 타임라인
    print("\n--- Fig 4: 연구 일정 ---")
    config['timeline'] = int(input("연구 기간 (년, 기본 3): ") or "3")

    # 마일스톤
    milestones_input = input("마일스톤 (월:이름, 쉼표 구분): ") or ""
    if milestones_input:
        config['milestones'] = []
        for ms in milestones_input.split(','):
            parts = ms.strip().split(':')
            if len(parts) >= 2:
                config['milestones'].append({
                    'month': int(parts[0]),
                    'name': parts[1]
                })

    # 출력 폴더
    config['output_dir'] = input(f"\n📁 출력 폴더 (기본: {DEFAULT_OUTPUT_DIR}): ") or str(DEFAULT_OUTPUT_DIR)

    return config


def create_sample_config() -> Dict[str, Any]:
    """샘플 설정 생성"""
    return {
        'title': '생애주기 뇌영상-유전체 AI: 발달 패턴 기반 노화 예측 모델',
        'title_en': 'LifeSpan Brain Imaging-Genomics AI: Development-Based Aging Prediction',

        'problem': '발달-노화 연속체에서의 뇌 구조/기능 변화 패턴이 규명되지 않음',
        'gap': '기존 연구는 단면적·단일모달 분석 중심으로 종단적 다중모달 통합 연구 부재',
        'hypothesis': '뇌영상-유전체 다중모달 AI 모델로 발달 패턴 기반 노화 예측 가능',
        'contribution': '생애주기 뇌 건강 예측 바이오마커 발굴 및 임상 적용 기반 마련',

        'aims': [
            {
                'name': 'Aim 1',
                'description': '다중모달 데이터 통합 플랫폼 구축',
                'methods': ['MRI 전처리', '유전체 QC', '데이터 표준화'],
                'steps': ['데이터 수집', '품질 관리', '통합 DB 구축', '검증'],
                'success_criteria': 'N > 10,000, 결측 < 5%',
                'start': 0,
                'end': 18,
            },
            {
                'name': 'Aim 2',
                'description': '발달-노화 예측 AI 모델 개발',
                'methods': ['딥러닝 모델', '전이학습', '앙상블'],
                'steps': ['특성 추출', '모델 설계', '학습/최적화', '성능 평가'],
                'success_criteria': 'MAE < 3년, R² > 0.85',
                'start': 12,
                'end': 30,
            },
            {
                'name': 'Aim 3',
                'description': '임상 적용 및 바이오마커 검증',
                'methods': ['외부 검증', '임상 파일럿', '해석 가능성'],
                'steps': ['외부 데이터 검증', '임상 코호트 적용', '바이오마커 발굴', '가이드라인'],
                'success_criteria': '외부 AUC > 0.80, 임상 활용도 검증',
                'start': 24,
                'end': 36,
            },
        ],

        'timeline': 3,

        'milestones': [
            {'month': 12, 'name': 'M1: 데이터 플랫폼 구축'},
            {'month': 24, 'name': 'M2: AI 모델 v1.0'},
            {'month': 36, 'name': 'M3: 임상 검증 완료'},
        ],

        'go_nogo': [
            {'month': 12, 'criteria': 'Go: N>5000, QC통과'},
            {'month': 24, 'criteria': 'Go: MAE<5년'},
        ],

        'output_dir': str(DEFAULT_OUTPUT_DIR),
    }


def main():
    parser = argparse.ArgumentParser(
        description='NRF 중견연구 제안서 다이어그램 자동 생성기',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 대화형 모드
  python scripts/proposal_diagram_generator.py

  # 설정 파일 사용
  python scripts/proposal_diagram_generator.py --config proposal_config.yaml

  # 샘플 설정으로 테스트
  python scripts/proposal_diagram_generator.py --sample

  # 샘플 설정 파일 생성
  python scripts/proposal_diagram_generator.py --create-sample proposal_config.yaml
        """
    )

    parser.add_argument('--config', '-c', help='설정 파일 경로 (YAML 또는 JSON)')
    parser.add_argument('--sample', '-s', action='store_true', help='샘플 설정으로 테스트')
    parser.add_argument('--create-sample', help='샘플 설정 파일 생성')
    parser.add_argument('--output', '-o', help='출력 폴더')

    args = parser.parse_args()

    # 샘플 설정 파일 생성
    if args.create_sample:
        sample_config = create_sample_config()
        output_path = Path(args.create_sample)

        if output_path.suffix == '.yaml' or output_path.suffix == '.yml':
            if not YAML_AVAILABLE:
                print("❌ pyyaml 미설치. JSON으로 저장합니다.")
                output_path = output_path.with_suffix('.json')
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(sample_config, f, ensure_ascii=False, indent=2)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(sample_config, f, allow_unicode=True, default_flow_style=False)
        else:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_config, f, ensure_ascii=False, indent=2)

        print(f"✅ 샘플 설정 파일 생성됨: {output_path}")
        return

    # 설정 로드
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in ['.yaml', '.yml']:
            config = load_config_from_yaml(args.config)
        else:
            config = load_config_from_json(args.config)
    elif args.sample:
        config = create_sample_config()
        print("📋 샘플 설정 사용")
    else:
        config = interactive_config()

    # 출력 폴더 오버라이드
    if args.output:
        config['output_dir'] = args.output

    # 다이어그램 생성
    generator = ProposalDiagramGenerator(config)
    generated_files = generator.generate_all()

    print(f"\n🎉 총 {len(generated_files)}개 다이어그램 생성 완료!")


if __name__ == "__main__":
    main()
