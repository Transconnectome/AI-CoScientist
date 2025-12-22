#!/usr/bin/env python3
"""
NRF 중견연구 제안서 Figure 생성 스크립트 (v2)

발달-노화 궤적 예측 멀티모달 파운데이션 모델
- Figure 1: 문제-갭-가설-기여 개요도
- Figure 2: 300B LifeSpan-FM 아키텍처
- Figure 3: 데이터 파이프라인
- Figure 4: 5년 로드맵

사용법:
    python generate_figures_v2.py

의존성:
    pip install matplotlib numpy
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import matplotlib.patheffects as path_effects
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본 폰트
# plt.rcParams['font.family'] = 'NanumGothic'  # 한글 사용 시
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def draw_rounded_box(ax, x, y, w, h, text, facecolor, edgecolor='#333333',
                     fontsize=8, fontweight='bold', linewidth=1.5, alpha=0.9):
    """둥근 모서리 박스 그리기"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha
    )
    ax.add_patch(box)

    # 텍스트 추가
    lines = text.split('\n')
    line_height = 0.15
    start_y = y + h/2 + (len(lines)-1) * line_height / 2

    for i, line in enumerate(lines):
        ax.text(x + w/2, start_y - i*line_height, line,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight if i == 0 else 'normal')


def draw_arrow(ax, start, end, color='#37474f', style='->', lw=1.5):
    """화살표 그리기"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle="arc3,rad=0"))


def draw_dashed_arrow(ax, start, end, color='#9e9e9e'):
    """점선 화살표 그리기"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                               linestyle='--', connectionstyle="arc3,rad=0"))


# ============================================================================
# Figure 2: 300B LifeSpan-FM Architecture
# ============================================================================
def generate_figure2_architecture():
    """300B 멀티모달 파운데이션 모델 아키텍처"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # 색상 정의
    colors = {
        'input': '#e3f2fd',       # 파란색 계열
        'fmri': '#ffecb3',        # 노란색 계열
        'eeg': '#ffe0b2',         # 주황색 계열
        'genomic': '#fff8e1',     # 연노랑
        'fusion': '#f3e5f5',      # 보라색 계열
        'trajectory': '#c8e6c9',  # 초록색 (강조)
        'downstream': '#e8f5e9',  # 연초록
        'benchmark': '#e0f2f1',   # 청록색
    }

    # ===== INPUT 계층 =====
    ax.text(2.5, 9.5, 'INPUT: Multimodal Brain Data', fontsize=11,
            fontweight='bold', ha='center')
    draw_rounded_box(ax, 0.3, 8.5, 1.4, 0.8, 'fMRI/sMRI\n70,000+', colors['input'], '#1565c0')
    draw_rounded_box(ax, 1.9, 8.5, 1.4, 0.8, 'EEG\n30,000h', colors['input'], '#1565c0')
    draw_rounded_box(ax, 3.5, 8.5, 1.4, 0.8, 'Genomic\n50+ PRS', colors['input'], '#1565c0')

    # ===== ENCODER 계층 (300B) =====
    ax.text(7, 9.5, 'MODALITY-SPECIFIC FM (300B)', fontsize=11,
            fontweight='bold', ha='center')

    # fMRI FM 100B
    draw_rounded_box(ax, 5.5, 7.8, 2.2, 1.4, 'fMRI FM\n100B\n4D Swin-T\nMAE 75%',
                     colors['fmri'], '#ff8f00', fontsize=9)
    # EEG FM 100B
    draw_rounded_box(ax, 8.0, 7.8, 2.2, 1.4, 'EEG FM\n100B\nDIVER-XL\nChannel-Equivariant',
                     colors['eeg'], '#e65100', fontsize=9)
    # Genomic FM 100B
    draw_rounded_box(ax, 10.5, 7.8, 2.2, 1.4, 'Genomic FM\n100B\nPRS-Transformer\nContrast Learning',
                     colors['genomic'], '#ffa000', fontsize=9)

    # ===== FUSION 계층 (100M) =====
    draw_rounded_box(ax, 7.0, 5.8, 3.0, 1.2, 'LLM Fusion Layer (100M)\nCross-Modal Attention\nSemantic Alignment',
                     colors['fusion'], '#7b1fa2', fontsize=10)

    # Unified Representation
    draw_rounded_box(ax, 7.5, 4.5, 2.0, 0.7, 'Unified Rep\n2048-dim',
                     '#ede7f6', '#512da8', fontsize=9)

    # ===== DOWNSTREAM 태스크 =====
    ax.text(7, 3.8, 'DOWNSTREAM TASKS (8)', fontsize=11, fontweight='bold', ha='center')

    # Trajectory (Killer Task) - 강조
    draw_rounded_box(ax, 0.5, 2.0, 2.5, 1.4, 'TRAJECTORY\n(Killer Task)\nDev: r>0.5\nAging: r>0.5',
                     colors['trajectory'], '#2e7d32', fontsize=9, linewidth=2.5)

    # Diagnosis
    draw_rounded_box(ax, 3.3, 2.0, 1.8, 1.4, 'AD Diagnosis\nAUC > 0.90\n\nPD Diagnosis\nAUC > 0.85',
                     colors['downstream'], '#388e3c', fontsize=8)

    # Inference
    draw_rounded_box(ax, 5.4, 2.0, 1.8, 1.4, 'Zero-shot\nAcc > 70%\n\nFew-shot\nAUC > 0.75',
                     colors['downstream'], '#689f38', fontsize=8)

    # 치매 전환
    draw_rounded_box(ax, 7.5, 2.0, 1.8, 1.4, 'Dementia\nConversion\nHR > 2.0',
                     colors['downstream'], '#8bc34a', fontsize=8)

    # Brain Age (Benchmark)
    draw_rounded_box(ax, 9.6, 2.0, 1.8, 1.4, 'Brain Age\n(Benchmark)\nMAE < 2.5y',
                     colors['benchmark'], '#00897b', fontsize=8)

    # Korean Cohort
    draw_rounded_box(ax, 11.7, 2.0, 1.8, 1.4, 'Korean\nValidation\n13K Subjects',
                     '#fce4ec', '#c2185b', fontsize=8)

    # ===== 화살표 연결 =====
    # Input -> Encoder
    draw_arrow(ax, (1.0, 8.5), (6.0, 9.2))
    draw_arrow(ax, (2.6, 8.5), (9.0, 9.2))
    draw_arrow(ax, (4.2, 8.5), (11.5, 9.2))

    # Encoder -> Fusion
    draw_arrow(ax, (6.6, 7.8), (7.8, 7.0))
    draw_arrow(ax, (9.1, 7.8), (8.5, 7.0))
    draw_arrow(ax, (11.6, 7.8), (9.5, 7.0))

    # Fusion -> Unified
    draw_arrow(ax, (8.5, 5.8), (8.5, 5.2))

    # Unified -> Downstream
    draw_arrow(ax, (7.5, 4.5), (1.75, 3.4))
    draw_arrow(ax, (7.8, 4.5), (4.2, 3.4))
    draw_arrow(ax, (8.3, 4.5), (6.3, 3.4))
    draw_arrow(ax, (8.7, 4.5), (8.4, 3.4))
    draw_arrow(ax, (9.0, 4.5), (10.5, 3.4))
    draw_arrow(ax, (9.5, 4.5), (12.6, 3.4))

    # 제목
    ax.text(7, 0.5, 'Figure 2. LifeSpan-FM: 300B Multimodal Foundation Model Architecture',
            fontsize=12, fontweight='bold', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('fig2_model_architecture_v2.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('fig2_model_architecture_v2.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Figure 2 saved: fig2_model_architecture_v2.png/pdf")
    plt.close()


# ============================================================================
# Figure 1: Problem-Gap-Hypothesis-Contribution
# ============================================================================
def generate_figure1_overview():
    """문제-갭-가설-기여 개요도"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    colors = {
        'problem': '#ffebee',
        'gap': '#fff3e0',
        'hypothesis': '#e3f2fd',
        'contribution': '#e8f5e9',
    }

    # ===== PROBLEM =====
    ax.add_patch(Rectangle((0.2, 5.5), 3.2, 2.2, fill=True,
                           facecolor=colors['problem'], edgecolor='#c62828', linewidth=2))
    ax.text(1.8, 7.5, 'PROBLEM', fontsize=11, fontweight='bold', ha='center', color='#c62828')
    ax.text(1.8, 7.0, 'Brain Age Gap Limitations', fontsize=9, ha='center', style='italic')
    ax.text(1.8, 6.5, '- Single timepoint prediction', fontsize=8, ha='center')
    ax.text(1.8, 6.2, '- Statistical bias (eta2=0.006)', fontsize=8, ha='center')
    ax.text(1.8, 5.9, '- Cross-sectional training', fontsize=8, ha='center')

    # ===== GAP =====
    ax.add_patch(Rectangle((3.7, 5.5), 3.2, 2.2, fill=True,
                           facecolor=colors['gap'], edgecolor='#ef6c00', linewidth=2))
    ax.text(5.3, 7.5, 'GAP', fontsize=11, fontweight='bold', ha='center', color='#ef6c00')
    ax.text(5.3, 7.0, 'Missing Trajectory FM', fontsize=9, ha='center', style='italic')
    ax.text(5.3, 6.5, '- No trajectory prediction', fontsize=8, ha='center')
    ax.text(5.3, 6.2, '- Scale gap (<1B vs 300B+)', fontsize=8, ha='center')
    ax.text(5.3, 5.9, '- No Korean validation', fontsize=8, ha='center')

    # ===== HYPOTHESIS =====
    ax.add_patch(Rectangle((7.2, 5.5), 3.2, 2.2, fill=True,
                           facecolor=colors['hypothesis'], edgecolor='#1565c0', linewidth=2))
    ax.text(8.8, 7.5, 'HYPOTHESIS', fontsize=11, fontweight='bold', ha='center', color='#1565c0')
    ax.text(8.8, 7.0, 'Trajectory Prediction Paradigm', fontsize=9, ha='center', style='italic')
    ax.text(8.8, 6.5, '- Developmental trajectory', fontsize=8, ha='center')
    ax.text(8.8, 6.2, '- Aging/dementia trajectory', fontsize=8, ha='center')
    ax.text(8.8, 5.9, '- 8 downstream tasks', fontsize=8, ha='center')

    # ===== CONTRIBUTION =====
    ax.add_patch(Rectangle((10.7, 5.5), 3.0, 2.2, fill=True,
                           facecolor=colors['contribution'], edgecolor='#2e7d32', linewidth=2))
    ax.text(12.2, 7.5, 'CONTRIBUTION', fontsize=11, fontweight='bold', ha='center', color='#2e7d32')
    ax.text(12.2, 7.0, '300B Trajectory FM', fontsize=9, ha='center', style='italic')
    ax.text(12.2, 6.5, '- fMRI/EEG/Genomic 100B', fontsize=8, ha='center')
    ax.text(12.2, 6.2, '- Korean 13K longitudinal', fontsize=8, ha='center')
    ax.text(12.2, 5.9, '- Real-time inference', fontsize=8, ha='center')

    # 화살표
    draw_arrow(ax, (3.4, 6.5), (3.7, 6.5), lw=2)
    draw_arrow(ax, (6.9, 6.5), (7.2, 6.5), lw=2)
    draw_arrow(ax, (10.4, 6.5), (10.7, 6.5), lw=2)

    # ===== 하단: 핵심 차별화 =====
    ax.add_patch(Rectangle((0.5, 0.5), 13, 4.5, fill=True,
                           facecolor='#fafafa', edgecolor='#424242', linewidth=1))
    ax.text(7, 4.7, 'KEY DIFFERENTIATION vs Competitors', fontsize=11,
            fontweight='bold', ha='center')

    # 테이블 헤더
    headers = ['', 'NeuroSTORM', 'BrainLM', 'COMICAL', 'This Study']
    for i, h in enumerate(headers):
        ax.text(0.8 + i*2.6, 4.2, h, fontsize=9, fontweight='bold', ha='left')

    # 테이블 데이터
    rows = [
        ['Data', '28M frames', '6,700h', '40K', '70K MRI+30Kh'],
        ['Modality', 'fMRI only', 'fMRI only', 'MRI+Gene', '3-Modal'],
        ['Scale', 'Unknown', '~100M', 'Unknown', '300B'],
        ['Task', 'Age/Disease', 'Zero-shot', 'GWAS', 'Trajectory'],
        ['Korean', 'No', 'No', 'No', '13K Longitudinal'],
    ]

    for j, row in enumerate(rows):
        for i, cell in enumerate(row):
            color = '#2e7d32' if i == 4 else '#333333'
            weight = 'bold' if i == 4 else 'normal'
            ax.text(0.8 + i*2.6, 3.6 - j*0.6, cell, fontsize=8, ha='left',
                   color=color, fontweight=weight)

    # 제목
    ax.text(7, 0.2, 'Figure 1. Research Overview: Problem-Gap-Hypothesis-Contribution',
            fontsize=12, fontweight='bold', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('fig1_problem_gap_v2.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('fig1_problem_gap_v2.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Figure 1 saved: fig1_problem_gap_v2.png/pdf")
    plt.close()


# ============================================================================
# Figure 3: Data Pipeline
# ============================================================================
def generate_figure3_pipeline():
    """데이터 파이프라인 및 학습 흐름도"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = {
        'data': '#e3f2fd',
        'preprocess': '#fff3e0',
        'pretrain': '#f3e5f5',
        'trajectory': '#e8f5e9',
        'validate': '#fce4ec',
        'deploy': '#e0f2f1',
    }

    # ===== STAGE 1: DATA COLLECTION =====
    ax.add_patch(Rectangle((0.2, 7.5), 3.8, 2.2, fill=True,
                           facecolor=colors['data'], edgecolor='#1565c0', linewidth=2))
    ax.text(2.1, 9.5, 'DATA COLLECTION', fontsize=10, fontweight='bold', ha='center', color='#1565c0')
    ax.text(2.1, 9.1, '70K+ MRI, 30Kh EEG', fontsize=8, ha='center', style='italic')

    # Data sources
    sources = [
        ('UK Biobank', '50,000', '45-82y'),
        ('ABCD Study', '12,000', '9-14y'),
        ('HCP Dev+Aging', '2,500', '5-100y'),
        ('Temple EEG', '30,000h', 'All ages'),
        ('CHA Hospital', '3,000', '8-25y'),
        ('Chosun Univ', '10,000', '50-90y'),
    ]
    for i, (name, n, age) in enumerate(sources):
        row = i // 2
        col = i % 2
        ax.text(0.5 + col*1.9, 8.7 - row*0.5, f'{name}', fontsize=7, fontweight='bold', ha='left')
        ax.text(0.5 + col*1.9, 8.4 - row*0.5, f'{n}, {age}', fontsize=6, ha='left', color='#555')

    # ===== STAGE 2: PREPROCESSING =====
    ax.add_patch(Rectangle((4.3, 7.5), 2.5, 2.2, fill=True,
                           facecolor=colors['preprocess'], edgecolor='#ef6c00', linewidth=2))
    ax.text(5.55, 9.5, 'PREPROCESS', fontsize=10, fontweight='bold', ha='center', color='#ef6c00')

    preprocess = [
        ('MRI', 'FreeSurfer\nfMRIPrep'),
        ('EEG', 'Artifact\n200Hz'),
        ('Genomic', 'QC+Impute\nPRS'),
    ]
    for i, (mod, desc) in enumerate(preprocess):
        ax.text(4.5, 8.9 - i*0.7, mod, fontsize=8, fontweight='bold', ha='left')
        ax.text(5.3, 8.85 - i*0.7, desc, fontsize=6, ha='left', color='#555')

    # ===== STAGE 3: PRETRAINING (300B) =====
    ax.add_patch(Rectangle((7.1, 7.5), 3.5, 2.2, fill=True,
                           facecolor=colors['pretrain'], edgecolor='#7b1fa2', linewidth=2))
    ax.text(8.85, 9.5, 'PRETRAINING: 300B', fontsize=10, fontweight='bold', ha='center', color='#7b1fa2')

    pretrain = [
        ('fMRI FM', '100B', '4D Swin MAE'),
        ('EEG FM', '100B', 'DIVER-XL'),
        ('Genomic FM', '100B', 'PRS-T'),
        ('LLM Fusion', '100M', 'Cross-Modal'),
    ]
    for i, (name, size, arch) in enumerate(pretrain):
        ax.text(7.3, 9.0 - i*0.45, f'{name}', fontsize=7, fontweight='bold', ha='left')
        ax.text(8.5, 9.0 - i*0.45, size, fontsize=7, ha='left', color='#7b1fa2')
        ax.text(9.3, 9.0 - i*0.45, arch, fontsize=6, ha='left', color='#555')

    # ===== STAGE 4: TRAJECTORY LEARNING =====
    ax.add_patch(Rectangle((10.9, 7.5), 2.8, 2.2, fill=True,
                           facecolor=colors['trajectory'], edgecolor='#2e7d32', linewidth=2.5))
    ax.text(12.3, 9.5, 'TRAJECTORY', fontsize=10, fontweight='bold', ha='center', color='#2e7d32')
    ax.text(12.3, 9.1, '(Killer Task)', fontsize=8, ha='center', style='italic', color='#2e7d32')

    trajectory = [
        ('CNF Model', 'Individual Trajectory'),
        ('Dev Traj', 'CHA 3K Longitudinal'),
        ('Aging Traj', 'Chosun 10K Longitudinal'),
    ]
    for i, (name, desc) in enumerate(trajectory):
        ax.text(11.1, 8.6 - i*0.5, name, fontsize=7, fontweight='bold', ha='left')
        ax.text(11.1, 8.35 - i*0.5, desc, fontsize=6, ha='left', color='#555')

    # ===== STAGE 5: VALIDATION =====
    ax.add_patch(Rectangle((0.2, 4.5), 5.5, 2.5, fill=True,
                           facecolor=colors['validate'], edgecolor='#c2185b', linewidth=2))
    ax.text(3.0, 6.8, 'VALIDATION', fontsize=10, fontweight='bold', ha='center', color='#c2185b')

    # Internal validation
    ax.text(0.5, 6.3, 'Internal (UK Biobank 20%)', fontsize=8, fontweight='bold', ha='left')
    ax.text(0.5, 6.0, 'Hold-out test set, cross-validation', fontsize=7, ha='left', color='#555')

    # Korean validation
    ax.text(0.5, 5.5, 'Korean Validation (13K)', fontsize=8, fontweight='bold', ha='left', color='#c2185b')
    ax.text(0.5, 5.2, 'CHA 3K (Dev) + Chosun 10K (Aging)', fontsize=7, ha='left', color='#555')
    ax.text(0.5, 4.9, 'Domain adaptation, ethnicity bias check', fontsize=7, ha='left', color='#555')

    # Clinical pilot
    ax.text(3.2, 6.3, 'Clinical Pilot (2K)', fontsize=8, fontweight='bold', ha='left')
    ax.text(3.2, 6.0, 'Real-world deployment', fontsize=7, ha='left', color='#555')
    ax.text(3.2, 5.7, 'Clinician feedback loop', fontsize=7, ha='left', color='#555')

    # ===== STAGE 6: DEPLOYMENT =====
    ax.add_patch(Rectangle((6.0, 4.5), 4.0, 2.5, fill=True,
                           facecolor=colors['deploy'], edgecolor='#00695c', linewidth=2))
    ax.text(8.0, 6.8, 'DEPLOYMENT', fontsize=10, fontweight='bold', ha='center', color='#00695c')

    deploy = [
        ('Knowledge Distillation', '300B -> 3B'),
        ('INT8 Quantization', 'Optimized inference'),
        ('API Service', '<30sec, 8 tasks'),
    ]
    for i, (name, desc) in enumerate(deploy):
        ax.text(6.2, 6.2 - i*0.55, name, fontsize=8, fontweight='bold', ha='left')
        ax.text(6.2, 5.95 - i*0.55, desc, fontsize=7, ha='left', color='#555')

    # ===== DOWNSTREAM TASKS =====
    ax.add_patch(Rectangle((10.3, 4.5), 5.4, 2.5, fill=True,
                           facecolor='#fff8e1', edgecolor='#f57c00', linewidth=2))
    ax.text(13.0, 6.8, '8 DOWNSTREAM TASKS', fontsize=10, fontweight='bold', ha='center', color='#f57c00')

    tasks = [
        ['Dev Trajectory r>0.5', 'Aging Trajectory r>0.5'],
        ['AD Diagnosis AUC>0.90', 'PD Diagnosis AUC>0.85'],
        ['Zero-shot Acc>70%', 'Few-shot AUC>0.75'],
        ['Dementia Conv HR>2.0', 'Brain Age MAE<2.5y'],
    ]
    for i, row in enumerate(tasks):
        for j, task in enumerate(row):
            ax.text(10.5 + j*2.7, 6.25 - i*0.45, task, fontsize=7, ha='left')

    # ===== ARROWS =====
    # Data -> Preprocess
    draw_arrow(ax, (4.0, 8.5), (4.3, 8.5), lw=2)
    # Preprocess -> Pretrain
    draw_arrow(ax, (6.8, 8.5), (7.1, 8.5), lw=2)
    # Pretrain -> Trajectory
    draw_arrow(ax, (10.6, 8.5), (10.9, 8.5), lw=2)
    # Trajectory -> Validation
    draw_arrow(ax, (12.3, 7.5), (12.3, 7.2), lw=1.5)
    draw_arrow(ax, (12.3, 7.2), (5.5, 7.0), lw=1.5)
    # Validation -> Deploy
    draw_arrow(ax, (5.7, 5.7), (6.0, 5.7), lw=2)
    # Deploy -> Downstream
    draw_arrow(ax, (10.0, 5.7), (10.3, 5.7), lw=2)

    # ===== MILESTONES =====
    ax.add_patch(Rectangle((0.2, 0.8), 15.5, 3.2, fill=True,
                           facecolor='#fafafa', edgecolor='#616161', linewidth=1.5))
    ax.text(8.0, 3.8, 'MILESTONES & GO/NO-GO GATES', fontsize=10, fontweight='bold', ha='center')

    milestones = [
        ('M1 (Y2)', 'MAE<2.5y, Retrieval>80%', '#1565c0'),
        ('M2 (Y3)', 'Dev Traj r>0.5, AUC>0.80', '#2e7d32'),
        ('M3 (Y4)', 'Dementia AUC>0.85', '#7b1fa2'),
        ('M4 (Y5)', 'Latency<30s, Avg AUC>0.85', '#c2185b'),
    ]

    gates = [
        ('G1', 'N>50K, 10B done'),
        ('G2', 'MAE<3.0y'),
        ('G3', 'Dev Traj r>0.4'),
        ('G4', 'Aging AUC>0.80'),
        ('G5', 'Inference<60s'),
    ]

    ax.text(0.5, 3.3, 'Milestones:', fontsize=9, fontweight='bold', ha='left')
    for i, (name, desc, color) in enumerate(milestones):
        ax.text(0.5 + i*3.9, 2.9, name, fontsize=8, fontweight='bold', ha='left', color=color)
        ax.text(0.5 + i*3.9, 2.55, desc, fontsize=7, ha='left')

    ax.text(0.5, 2.0, 'Go/No-Go Gates:', fontsize=9, fontweight='bold', ha='left')
    for i, (name, crit) in enumerate(gates):
        ax.text(0.5 + i*3.1, 1.6, name, fontsize=8, fontweight='bold', ha='left', color='#d32f2f')
        ax.text(0.5 + i*3.1, 1.25, crit, fontsize=7, ha='left')

    # Title
    ax.text(8.0, 0.3, 'Figure 3. LifeSpan-FM: Data Pipeline and Training Workflow',
            fontsize=12, fontweight='bold', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('fig3_data_pipeline_v2.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('fig3_data_pipeline_v2.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Figure 3 saved: fig3_data_pipeline_v2.png/pdf")
    plt.close()


# ============================================================================
# Figure 4: 5-Year Roadmap (Gantt-style)
# ============================================================================
def generate_figure4_roadmap():
    """5년 연구 로드맵 (간트 차트 스타일)"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 62)  # 60 months + margins
    ax.set_ylim(0, 22)
    ax.axis('off')

    # Title
    ax.text(31, 21, 'LifeSpan-FM 5-Year Research Roadmap (2026-2031)',
            fontsize=14, fontweight='bold', ha='center')

    # Year labels
    years = ['2026', '2027', '2028', '2029', '2030', '2031']
    for i, year in enumerate(years):
        x = 2 + i*10
        ax.axvline(x=x, color='#e0e0e0', linestyle='-', linewidth=0.5, ymin=0.05, ymax=0.9)
        ax.text(x + 5, 19.5, year, fontsize=10, fontweight='bold', ha='center')

    # Month grid (quarterly)
    for i in range(25):  # 24 quarters
        x = 2 + i*2.5
        ax.axvline(x=x, color='#f5f5f5', linestyle=':', linewidth=0.3, ymin=0.05, ymax=0.9)

    colors = {
        'obj1': '#bbdefb',  # Blue - FM Training
        'obj2': '#c8e6c9',  # Green - Dev Trajectory
        'obj3': '#fff9c4',  # Yellow - Aging Trajectory
        'obj4': '#f8bbd0',  # Pink - Inference
        'milestone': '#7e57c2',  # Purple - Milestone
        'gate': '#ef5350',  # Red - Go/No-Go
    }

    def draw_task_bar(y, start_month, duration, label, color, sublabel=''):
        """Task bar 그리기 (start_month: 0=Jan 2026)"""
        x_start = 2 + start_month * (60/60)  # Scale to 60 months = 60 units
        width = duration * (60/60)
        bar = FancyBboxPatch((x_start, y-0.35), width, 0.7,
                             boxstyle="round,pad=0.01,rounding_size=0.1",
                             facecolor=color, edgecolor='#333', linewidth=0.8)
        ax.add_patch(bar)
        ax.text(x_start + width/2, y, label, fontsize=7, fontweight='bold',
                ha='center', va='center')
        if sublabel:
            ax.text(x_start + width + 0.3, y, sublabel, fontsize=6,
                    ha='left', va='center', color='#555')

    def draw_milestone(y, month, label, color='#7e57c2'):
        """Milestone diamond 그리기"""
        x = 2 + month * (60/60)
        diamond = plt.Polygon([[x, y-0.4], [x+0.4, y], [x, y+0.4], [x-0.4, y]],
                              facecolor=color, edgecolor='#333', linewidth=1)
        ax.add_patch(diamond)
        ax.text(x, y-0.8, label, fontsize=6, ha='center', va='top',
                fontweight='bold', color=color)

    # ===== OBJECTIVE 1: 300B FM (Y1-Y2) =====
    y_base = 17
    ax.text(0.5, y_base+0.8, 'Obj 1: 300B FM', fontsize=9, fontweight='bold', color='#1565c0')

    draw_task_bar(y_base, 2, 10, 'Data Integration 70K', colors['obj1'])
    draw_task_bar(y_base-1, 5, 12, 'fMRI FM 100B', colors['obj1'])
    draw_task_bar(y_base-2, 8, 10, 'EEG FM 100B', colors['obj1'])
    draw_task_bar(y_base-3, 12, 8, 'Genomic FM 100B', colors['obj1'])
    draw_task_bar(y_base-4, 17, 6, 'LLM Fusion 100M', colors['obj1'])
    draw_milestone(y_base-4, 19, 'M1: MAE<2.5y', colors['milestone'])

    # ===== OBJECTIVE 2: Dev Trajectory (Y2-Y3) =====
    y_base = 11.5
    ax.text(0.5, y_base+0.8, 'Obj 2: Dev Trajectory', fontsize=9, fontweight='bold', color='#2e7d32')
    ax.text(0.5, y_base+0.3, '(CHA Hospital)', fontsize=7, color='#555')

    draw_task_bar(y_base, 14, 6, 'CHA 3K Acquisition', colors['obj2'])
    draw_task_bar(y_base-1, 20, 9, 'CNF Model Dev', colors['obj2'])
    draw_task_bar(y_base-2, 24, 8, 'Dev Trajectory Train', colors['obj2'])
    draw_task_bar(y_base-3, 29, 6, 'Neurodevelopmental Pred', colors['obj2'])
    draw_milestone(y_base-3, 31, 'M2: r>0.5, AUC>0.80', colors['milestone'])

    # ===== OBJECTIVE 3: Aging Trajectory (Y3-Y4) =====
    y_base = 6
    ax.text(0.5, y_base+0.8, 'Obj 3: Aging Trajectory', fontsize=9, fontweight='bold', color='#ff8f00')
    ax.text(0.5, y_base+0.3, '(Chosun Univ)', fontsize=7, color='#555')

    draw_task_bar(y_base, 29, 6, 'Chosun 10K Acquisition', colors['obj3'])
    draw_task_bar(y_base-1, 35, 9, 'Aging Trajectory Model', colors['obj3'])
    draw_task_bar(y_base-2, 41, 8, 'Dementia Conversion Pred', colors['obj3'])
    draw_task_bar(y_base-3, 44, 6, 'Korean Domain Adaptation', colors['obj3'])
    draw_milestone(y_base-3, 49, 'M3: Dementia AUC>0.85', colors['milestone'])

    # ===== OBJECTIVE 4: Inference & Downstream (Y4-Y5) =====
    y_base = 17
    ax.text(42, y_base+0.8, 'Obj 4: Inference', fontsize=9, fontweight='bold', color='#c2185b')

    draw_task_bar(y_base, 48, 6, 'Distillation 300B->3B', colors['obj4'])
    draw_task_bar(y_base-1, 51, 4, 'INT8 Quantization', colors['obj4'])
    draw_task_bar(y_base-2, 53, 6, '8 Downstream Validation', colors['obj4'])
    draw_task_bar(y_base-3, 56, 6, 'Clinical Pilot 2K', colors['obj4'])
    draw_milestone(y_base-3.5, 61, 'M4: <30s, AUC>0.85', colors['milestone'])

    # ===== GO/NO-GO GATES =====
    ax.add_patch(Rectangle((1, 0.5), 60, 1.5, fill=True,
                           facecolor='#ffebee', edgecolor='#c62828', linewidth=1.5))
    ax.text(31, 1.8, 'GO/NO-GO GATES', fontsize=9, fontweight='bold', ha='center', color='#c62828')

    gates = [
        (13, 'G1: N>50K, 10B'),
        (25, 'G2: MAE<3.0y'),
        (31, 'G3: Dev r>0.4'),
        (49, 'G4: Aging AUC>0.80'),
        (55, 'G5: Inference<60s'),
    ]

    for month, label in gates:
        x = 2 + month * (60/60)
        ax.plot([x], [1.2], 'v', markersize=10, color='#c62828')
        ax.text(x, 0.8, label, fontsize=6, ha='center', va='top', color='#c62828')

    # Legend
    legend_items = [
        ('300B FM Training', colors['obj1']),
        ('Dev Trajectory', colors['obj2']),
        ('Aging Trajectory', colors['obj3']),
        ('Inference/Deploy', colors['obj4']),
        ('Milestone', colors['milestone']),
        ('Go/No-Go', colors['gate']),
    ]

    ax.text(2, 20.5, 'Legend:', fontsize=8, fontweight='bold')
    for i, (label, color) in enumerate(legend_items):
        x = 6 + i*9
        ax.add_patch(Rectangle((x, 20.3), 1.5, 0.4, facecolor=color, edgecolor='#333', linewidth=0.5))
        ax.text(x+2, 20.5, label, fontsize=7, ha='left', va='center')

    # Title at bottom
    ax.text(31, -0.3, 'Figure 4. 5-Year Research Roadmap with Milestones and Go/No-Go Gates',
            fontsize=12, fontweight='bold', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig('fig4_gantt_roadmap_v2.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('fig4_gantt_roadmap_v2.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Figure 4 saved: fig4_gantt_roadmap_v2.png/pdf")
    plt.close()


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating NRF Proposal Figures v2...")
    print("=" * 50)

    generate_figure1_overview()
    generate_figure2_architecture()
    generate_figure3_pipeline()
    generate_figure4_roadmap()

    print("=" * 50)
    print("All figures generated successfully!")
    print("\nGenerated files:")
    print("  - fig1_problem_gap_v2.png/pdf")
    print("  - fig2_model_architecture_v2.png/pdf")
    print("  - fig3_data_pipeline_v2.png/pdf")
    print("  - fig4_gantt_roadmap_v2.png/pdf")
