#!/usr/bin/env python3
"""
NRF 중견연구 제안서 블록 통합기
==============================

5개 블록 + Figure 4종을 최종 제안서로 통합합니다.

블록 구조:
- Block 1: 연구과제의 필요성 (1p)
- Block 2: 연구과제의 목표 및 내용 (2-3p)
- Block 3: 추진전략·방법 및 추진체계 (3-3.5p)
- Block 4: 연구자의 연구 수행역량 (1.5-2p)
- Block 5: 활용방안 및 기대효과 (1p)

Usage:
    # 블록 디렉토리에서 통합
    poetry run python scripts/integrate_proposal_blocks.py \
        --blocks-dir "phase3_blocks/" \
        --figures-dir "phase3_figures/" \
        --output "integrated_proposal.md"

    # 개별 블록 파일 지정
    poetry run python scripts/integrate_proposal_blocks.py \
        --block1 "block1_necessity.md" \
        --block2 "block2_goals.md" \
        --block3 "block3_methods.md" \
        --block4 "block4_capability.md" \
        --block5 "block5_impact.md" \
        --figures-dir "figures/" \
        --output "proposal.md"

    # 템플릿 생성
    poetry run python scripts/integrate_proposal_blocks.py \
        --create-template \
        --output "proposal_template/"
"""

import argparse
import json
import re
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime

@dataclass
class BlockContent:
    """블록 콘텐츠"""
    number: int
    title: str
    content: str
    recommended_pages: str
    weight: str  # 평가 가중치

@dataclass
class FigureInfo:
    """그림 정보"""
    number: int
    title: str
    file_path: Optional[Path]
    placement: str  # 어느 블록에 배치할지

class ProposalIntegrator:
    """제안서 블록 통합기"""

    BLOCK_SPECS = [
        BlockContent(1, "연구과제의 필요성", "", "1p", "40%"),
        BlockContent(2, "연구과제의 목표 및 내용", "", "2-3p", "40%"),
        BlockContent(3, "연구과제의 추진전략·방법 및 추진체계", "", "3-3.5p", "30%"),
        BlockContent(4, "연구자의 연구 수행역량", "", "1.5-2p", "20%"),
        BlockContent(5, "연구과제의 활용방안 및 기대효과", "", "1p", "10%"),
    ]

    FIGURE_SPECS = [
        FigureInfo(1, "문제-갭-가설-기여 인포그래픽", None, "block1"),
        FigureInfo(2, "방법 파이프라인/워크플로", None, "block3"),
        FigureInfo(3, "Aim별 실험/데이터 흐름", None, "block3"),
        FigureInfo(4, "Gantt + 마일스톤 + Go/No-Go", None, "block3"),
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.blocks: List[BlockContent] = []
        self.figures: List[FigureInfo] = []

    def load_blocks_from_dir(self, blocks_dir: Path) -> bool:
        """디렉토리에서 블록 파일 로드"""
        if not blocks_dir.exists():
            print(f"Error: 블록 디렉토리가 없습니다: {blocks_dir}")
            return False

        self.blocks = []
        for i, spec in enumerate(self.BLOCK_SPECS, 1):
            # 다양한 파일명 패턴 시도
            patterns = [
                f"block{i}*.md",
                f"Block{i}*.md",
                f"{i}_*.md",
                f"*block{i}*.md",
            ]

            found = False
            for pattern in patterns:
                files = list(blocks_dir.glob(pattern))
                if files:
                    content = files[0].read_text(encoding='utf-8')
                    block = BlockContent(
                        number=i,
                        title=spec.title,
                        content=content,
                        recommended_pages=spec.recommended_pages,
                        weight=spec.weight
                    )
                    self.blocks.append(block)
                    found = True
                    if self.verbose:
                        print(f"  Block {i} 로드: {files[0].name}")
                    break

            if not found:
                # 빈 블록 생성
                block = BlockContent(
                    number=i,
                    title=spec.title,
                    content=f"[Block {i} 내용을 여기에 작성하세요]",
                    recommended_pages=spec.recommended_pages,
                    weight=spec.weight
                )
                self.blocks.append(block)
                if self.verbose:
                    print(f"  Block {i} 없음 - 플레이스홀더 생성")

        return True

    def load_blocks_from_files(self, block_files: Dict[int, Path]) -> bool:
        """개별 파일에서 블록 로드"""
        self.blocks = []

        for i, spec in enumerate(self.BLOCK_SPECS, 1):
            if i in block_files and block_files[i].exists():
                content = block_files[i].read_text(encoding='utf-8')
            else:
                content = f"[Block {i} 내용을 여기에 작성하세요]"

            block = BlockContent(
                number=i,
                title=spec.title,
                content=content,
                recommended_pages=spec.recommended_pages,
                weight=spec.weight
            )
            self.blocks.append(block)

        return True

    def load_figures(self, figures_dir: Path) -> bool:
        """그림 파일 로드"""
        if not figures_dir or not figures_dir.exists():
            self.figures = self.FIGURE_SPECS.copy()
            return True

        self.figures = []
        for spec in self.FIGURE_SPECS:
            # 다양한 파일명 패턴 시도
            patterns = [
                f"fig{spec.number}*.png",
                f"Fig{spec.number}*.png",
                f"figure{spec.number}*.png",
                f"*fig{spec.number}*.png",
            ]

            file_path = None
            for pattern in patterns:
                files = list(figures_dir.glob(pattern))
                if files:
                    file_path = files[0]
                    break

            # 다른 확장자 시도
            if not file_path:
                for ext in ['.jpg', '.jpeg', '.pdf', '.svg']:
                    for pattern in patterns:
                        p = pattern.replace('.png', ext)
                        files = list(figures_dir.glob(p))
                        if files:
                            file_path = files[0]
                            break
                    if file_path:
                        break

            fig = FigureInfo(
                number=spec.number,
                title=spec.title,
                file_path=file_path,
                placement=spec.placement
            )
            self.figures.append(fig)

            if self.verbose:
                if file_path:
                    print(f"  Fig {spec.number} 발견: {file_path.name}")
                else:
                    print(f"  Fig {spec.number} 없음")

        return True

    def integrate(self, output_path: Path, copy_figures: bool = True) -> str:
        """블록과 그림을 통합하여 최종 제안서 생성"""
        lines = []

        # 헤더
        lines.append("# 연구계획서(연구내용)")
        lines.append("")
        lines.append(f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("> 총 분량: 10페이지 이내")
        lines.append("")
        lines.append("---")
        lines.append("")

        # 각 블록 통합
        for block in self.blocks:
            # 블록 제목
            lines.append(f"## {block.number}. {block.title}")
            lines.append("")

            # 콘텐츠
            content = block.content.strip()

            # Block 1에 Fig 1 삽입
            if block.number == 1:
                fig1 = next((f for f in self.figures if f.number == 1), None)
                if fig1 and fig1.file_path:
                    content = self._insert_figure_ref(content, fig1, "end")

            # Block 3에 Fig 2, 3, 4 삽입
            elif block.number == 3:
                for fig_num in [2, 3, 4]:
                    fig = next((f for f in self.figures if f.number == fig_num), None)
                    if fig and fig.file_path:
                        content = self._insert_figure_ref(content, fig, "end")

            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        # 참고문헌 섹션 (분량 제외)
        lines.append("## 참고문헌")
        lines.append("")
        lines.append("[참고문헌은 페이지 수에 포함되지 않습니다]")
        lines.append("")
        lines.append("1. [참고문헌 목록을 여기에 작성하세요]")
        lines.append("")

        # 최종 텍스트
        final_text = "\n".join(lines)

        # 파일 저장
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_text, encoding='utf-8')

        # 그림 파일 복사
        if copy_figures:
            figures_output_dir = output_path.parent / "figures"
            figures_output_dir.mkdir(exist_ok=True)

            for fig in self.figures:
                if fig.file_path and fig.file_path.exists():
                    dest = figures_output_dir / fig.file_path.name
                    shutil.copy2(fig.file_path, dest)
                    if self.verbose:
                        print(f"  그림 복사: {fig.file_path.name} -> {dest}")

        return final_text

    def _insert_figure_ref(self, content: str, fig: FigureInfo, position: str = "end") -> str:
        """콘텐츠에 그림 참조 삽입"""
        fig_ref = f"\n\n![{fig.title}](figures/{fig.file_path.name})\n*Fig {fig.number}: {fig.title}*\n"

        if position == "end":
            return content + fig_ref
        elif position == "start":
            return fig_ref + content
        else:
            return content + fig_ref

    @staticmethod
    def create_template(output_dir: Path):
        """빈 템플릿 생성"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Block 템플릿 생성
        block_templates = {
            1: """# Block 1: 연구과제의 필요성

> 권장 분량: 1페이지 | 평가 가중치: 40% (창의성/도전성과 공유)

## 핵심 체크리스트
- [ ] 기존 접근의 한계가 명확히 기술됨
- [ ] 핵심 갭/미해결 문제가 1문장으로 정의됨
- [ ] "왜 지금/왜 나"가 3문장 이내로 설명됨
- [ ] Fig 1 (문제-갭-가설-기여 인포그래픽) 포함

---

## 작성 내용

### 연구 배경 및 현황
[연구 분야의 국내외 현황을 간략히 기술]

### 기존 연구의 한계
[기존 접근법의 구체적인 한계점 기술]

### 핵심 갭 (1문장)
> **[미해결 핵심 문제를 1문장으로 명확히 정의]**

### 왜 지금, 왜 이 연구인가
1. [시의성/필요성 근거 1]
2. [연구자의 적합성 근거]
3. [기대 기여 요약]

---

[Fig 1: 문제-갭-가설-기여 인포그래픽 삽입 위치]
""",

            2: """# Block 2: 연구과제의 목표 및 내용

> 권장 분량: 2-3페이지 | 평가 가중치: 40% (창의성/도전성)

## 핵심 체크리스트
- [ ] 최종 목표가 1문장으로 명확히 기술됨
- [ ] Aim이 2-3개로 구조화됨
- [ ] 각 Aim별 핵심 연구질문이 정의됨
- [ ] 선행연구 대비 차별성 도표 포함
- [ ] Layer 1(개념) + Layer 2(실행) 2층 차별성 구조

---

## 작성 내용

### 1) 연구과제의 최종 목표
> **[최종 목표를 1문장으로 명확히 기술]**

### 2) 연구과제의 내용

#### 선행연구 분석 및 차별성

| 구분 | 기존 연구 | 본 연구 | 차별점 |
|------|----------|---------|--------|
| 접근법 | | | |
| 데이터 | | | |
| 방법론 | | | |

#### 2-Layer 차별성 구조

**Layer 1 (개념적 차별성)**
> [기존 가정을 깨는 새로운 개념/접근]

**Layer 2 (실행적 차별성)**
> [데이터/모델/기술의 구체적 차별점]

---

### Aim 1: [Aim 1 제목]
- **목표**: [구체적 목표]
- **핵심 질문**: [해결할 핵심 연구질문]
- **주요 내용**: [수행할 연구 내용]

### Aim 2: [Aim 2 제목]
- **목표**: [구체적 목표]
- **핵심 질문**: [해결할 핵심 연구질문]
- **주요 내용**: [수행할 연구 내용]

### Aim 3: [Aim 3 제목] (선택)
- **목표**: [구체적 목표]
- **핵심 질문**: [해결할 핵심 연구질문]
- **주요 내용**: [수행할 연구 내용]
""",

            3: """# Block 3: 추진전략·방법 및 추진체계

> 권장 분량: 3-3.5페이지 | 평가 가중치: 30% (방법론 적합성)

## 핵심 체크리스트
- [ ] Aim별 세부 방법이 구체적으로 기술됨
- [ ] 각 실험/분석의 성공 기준(정량지표) 정의됨
- [ ] 핵심 리스크 + 대안 프로토콜 명시됨
- [ ] Fig 2 (방법 파이프라인) 포함
- [ ] Fig 3 (Aim별 실험/데이터 흐름) 포함
- [ ] Fig 4 (Gantt + Go/No-Go) 포함
- [ ] 추진체계(인력/장비/협력) 명시됨
- [ ] 연구기간/연구비 적정성 근거 제시됨

---

## 작성 내용

### 1) 연구과제의 추진전략·방법

[Fig 2: 방법 파이프라인 삽입]

#### Aim 1: [Aim 1 제목]

| 단계 | 세부 방법 | 사용 도구/기술 | 성공 기준 |
|------|----------|----------------|----------|
| 1.1 | | | |
| 1.2 | | | |

**성공 기준 (정량지표)**: [예: N > 10,000, QC 통과율 > 95%]

**핵심 리스크 및 대안**:
- 리스크: [예상 리스크]
- 대안: [대체 프로토콜/방법]

---

#### Aim 2: [Aim 2 제목]

| 단계 | 세부 방법 | 사용 도구/기술 | 성공 기준 |
|------|----------|----------------|----------|
| 2.1 | | | |
| 2.2 | | | |

**성공 기준 (정량지표)**: [예: MAE < 3년, R² > 0.85]

---

[Fig 3: Aim별 실험/데이터 흐름 삽입]

### 2) 연구과제의 추진체계

#### 연구 인력 구성
| 역할 | 인원 | 담당 업무 |
|------|------|----------|
| 책임연구원 | 1명 | 총괄, Aim 1-3 설계 |
| 연구원 | | |
| 학생연구원 | | |

#### 연구 장비 및 인프라
- [보유 장비/인프라 목록]

#### 협력 네트워크
- [국내외 협력 연구자/기관]

### 3) 연구기간 및 연구비 적정성

[Fig 4: Gantt 차트 + 마일스톤 삽입]

#### 연차별 목표 및 마일스톤

| 년차 | 주요 목표 | 마일스톤 | Go/No-Go 기준 |
|------|----------|----------|---------------|
| 1년차 | | M1: | |
| 2년차 | | M2: | |
| 3년차 | | M3: | |

#### 연구비 산정 근거
[연구비 적정성에 대한 설명]
""",

            4: """# Block 4: 연구자의 연구 수행역량

> 권장 분량: 1.5-2페이지 | 평가 가중치: 20%

## 핵심 체크리스트
- [ ] PI의 핵심 역량이 연구 주제와 직접 연결됨
- [ ] 대표 성과 1-2개가 구체적으로 기술됨
- [ ] 기확보 인프라/장비/데이터가 명시됨
- [ ] 협력 네트워크(국내외)가 제시됨
- [ ] 선행연구와의 연속성이 설명됨

---

## 작성 내용

### 연구책임자 핵심 역량

**[연구자 이름, 소속, 직위]**

본 연구과제와 직접 연관된 핵심 역량:
1. [역량 1]: [구체적 설명]
2. [역량 2]: [구체적 설명]
3. [역량 3]: [구체적 설명]

### 대표 연구 성과

#### 대표 성과 1: [제목]
- **내용**: [성과 내용 설명]
- **본 연구와의 연결**: [어떻게 본 연구에 기여하는지]

#### 대표 성과 2: [제목]
- **내용**: [성과 내용 설명]
- **본 연구와의 연결**: [어떻게 본 연구에 기여하는지]

### 연구 인프라 및 장비

| 항목 | 상세 내용 | 활용 계획 |
|------|----------|----------|
| 장비 | | |
| 데이터 | | |
| 소프트웨어 | | |

### 협력 네트워크

#### 국내 협력
- [협력 연구자/기관 1]
- [협력 연구자/기관 2]

#### 국제 협력
- [해외 협력 연구자/기관]

### 선행연구와의 연속성

본 연구는 연구책임자의 기존 연구 [선행연구 제목]의 연장선상에 있으며,
[선행연구에서 얻은 예비 결과/데이터/경험]을 바탕으로 수행됩니다.
""",

            5: """# Block 5: 활용방안 및 기대효과

> 권장 분량: 1페이지 | 평가 가중치: 10%

## 핵심 체크리스트
- [ ] 학문적 기여가 구체적으로 기술됨
- [ ] 기술적/응용적 확장 가능성 제시됨
- [ ] 후속 연구 방향이 명시됨
- [ ] 파급효과(학문/기술/인력/사회) 4영역 커버
- [ ] 성과확산 계획(논문/특허/기술이전 등) 포함

---

## 작성 내용

### 1) 연구과제의 활용방안

#### 학술적 활용
- [논문 발표 계획]
- [학회 발표 계획]

#### 기술적 활용
- [기술 이전 가능성]
- [특허 출원 계획]

#### 응용/확장 가능성
- [다른 분야/문제에 적용 가능성]

### 2) 연구과제의 기대효과

#### 학문적 기여
1. [학문적 기여 1]
2. [학문적 기여 2]

#### 기술적 기여
1. [기술적 기여 1]
2. [기술적 기여 2]

#### 인력 양성
- 박사 [N]명, 석사 [N]명 양성 예정
- [전문 인력 양성 계획]

#### 사회적 파급효과
- [사회/경제적 기여]

### 후속 연구 방향

본 연구의 결과를 바탕으로:
1. [후속 연구 방향 1]
2. [후속 연구 방향 2]

### 성과확산 계획

| 성과 유형 | 목표 | 연차별 계획 |
|----------|------|------------|
| SCI 논문 | [N]편 | 1년차: / 2년차: / 3년차: |
| 학회 발표 | [N]건 | |
| 특허 | [N]건 | |
| 기술 이전 | | |
"""
        }

        # 블록 파일 생성
        for block_num, content in block_templates.items():
            file_path = output_dir / f"block{block_num}_{['necessity', 'goals', 'methods', 'capability', 'impact'][block_num-1]}.md"
            file_path.write_text(content, encoding='utf-8')
            print(f"  생성: {file_path.name}")

        # 설정 파일 생성
        config = {
            "title": "연구 과제명",
            "pi_name": "연구책임자명",
            "organization": "소속기관",
            "duration": "3년",
            "budget": "연 1.5억원",
            "aims": [
                {"name": "Aim 1", "description": ""},
                {"name": "Aim 2", "description": ""},
                {"name": "Aim 3", "description": ""}
            ]
        }
        config_path = output_dir / "proposal_config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"  생성: {config_path.name}")

        # figures 디렉토리 생성
        (output_dir / "figures").mkdir(exist_ok=True)
        print(f"  생성: figures/")

        print(f"\n템플릿 생성 완료: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="NRF 중견연구 제안서 블록 통합기"
    )

    # 입력 옵션
    parser.add_argument("--blocks-dir", type=Path,
                       help="블록 파일 디렉토리")
    parser.add_argument("--block1", type=Path, help="Block 1 파일")
    parser.add_argument("--block2", type=Path, help="Block 2 파일")
    parser.add_argument("--block3", type=Path, help="Block 3 파일")
    parser.add_argument("--block4", type=Path, help="Block 4 파일")
    parser.add_argument("--block5", type=Path, help="Block 5 파일")
    parser.add_argument("--figures-dir", type=Path,
                       help="그림 파일 디렉토리")

    # 출력 옵션
    parser.add_argument("--output", "-o", type=Path,
                       default=Path("integrated_proposal.md"),
                       help="출력 파일 경로")

    # 동작 옵션
    parser.add_argument("--create-template", action="store_true",
                       help="빈 템플릿 생성")
    parser.add_argument("--no-copy-figures", action="store_true",
                       help="그림 파일 복사 안함")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세 출력")

    args = parser.parse_args()

    print("=" * 60)
    print("  NRF 중견연구 제안서 블록 통합기")
    print("=" * 60)

    # 템플릿 생성 모드
    if args.create_template:
        output_dir = args.output if args.output.suffix == "" else args.output.parent / "proposal_template"
        ProposalIntegrator.create_template(output_dir)
        return

    # 통합 모드
    integrator = ProposalIntegrator(verbose=args.verbose)

    # 블록 로드
    if args.blocks_dir:
        print(f"\n블록 디렉토리: {args.blocks_dir}")
        if not integrator.load_blocks_from_dir(args.blocks_dir):
            sys.exit(1)
    elif any([args.block1, args.block2, args.block3, args.block4, args.block5]):
        block_files = {}
        for i, arg in enumerate([args.block1, args.block2, args.block3, args.block4, args.block5], 1):
            if arg:
                block_files[i] = arg
        integrator.load_blocks_from_files(block_files)
    else:
        print("Error: --blocks-dir 또는 개별 --block[1-5] 파일을 지정하세요")
        print("       또는 --create-template으로 템플릿을 생성하세요")
        sys.exit(1)

    # 그림 로드
    if args.figures_dir:
        print(f"\n그림 디렉토리: {args.figures_dir}")
        integrator.load_figures(args.figures_dir)

    # 통합 실행
    print(f"\n통합 중...")
    result = integrator.integrate(
        args.output,
        copy_figures=not args.no_copy_figures
    )

    # 결과 출력
    print()
    print("-" * 60)
    print(f"통합 완료!")
    print(f"  출력 파일: {args.output}")
    print(f"  블록 수: {len(integrator.blocks)}")
    print(f"  그림 수: {sum(1 for f in integrator.figures if f.file_path)}")
    print(f"  총 문자 수: {len(result):,}")
    print("-" * 60)

    # 제출 체크리스트 안내
    print("\n다음 단계: 제출 전 체크리스트 검증")
    print(f"  poetry run python scripts/validate_submission_checklist.py --input {args.output}")


if __name__ == "__main__":
    main()
