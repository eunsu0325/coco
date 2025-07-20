# framework/config.py - 프레임워크 설정 클래스들 (수정된 버전)
"""
COCONUT Framework Configuration Classes

DESIGN PHILOSOPHY:
- Comprehensive configuration management
- W2ML parameter integration
- Clear documentation of all settings
- 🔥 Model saving configuration support
"""

import dataclasses
from pathlib import Path
from typing import Optional

# === 적응 실험용 설정들 ===
@dataclasses.dataclass
class ReplayBufferConfig:
    """지능형 리플레이 버퍼 설정"""
    config_file: Path
    maximize_diversity: bool
    max_buffer_size: int
    similarity_threshold: float
    storage_path: str
    feature_extraction_for_diversity: bool
    # W2ML 관련 설정
    enable_smart_sampling: Optional[bool] = True
    hard_sample_ratio: Optional[float] = 0.3
    diversity_update_frequency: Optional[int] = 10
    # 모델 저장 경로
    model_save_path: Optional[str] = "./results/models/"

@dataclasses.dataclass
class ContinualLearnerConfig:
    """연속 학습기 설정"""
    config_file: Path
    adaptation: bool
    adaptation_epochs: int
    sync_frequency: int
    replay_weight: float
    new_data_weight: float
    # W2ML 전용 설정
    enable_w2ml: Optional[bool] = True
    hard_negative_weight: Optional[float] = 2.0
    hard_positive_weight: Optional[float] = 1.5
    similarity_threshold_negative: Optional[float] = 0.7
    similarity_threshold_positive: Optional[float] = 0.5
    adaptive_weight_scheduling: Optional[bool] = True
    # 🔥 중간 저장 빈도 설정
    intermediate_save_frequency: Optional[int] = 100

@dataclasses.dataclass  
class LossConfig:
    """손실 함수 설정"""
    config_file: Path
    temp: float
    type: Optional[str] = "SupConLoss"  # 수정: type 필드 추가
    # 사전 훈련용 (하이브리드 손실)
    weight1: Optional[float] = 0.8  # ArcFace 가중치
    weight2: Optional[float] = 0.2  # SupCon 가중치
    # W2ML 전용 설정 (온라인 적응용) - adapt_config.yaml과 일치하도록 수정
    w2ml_temperature: Optional[float] = 0.07
    hard_positive_weight: Optional[float] = 2.0
    hard_negative_weight: Optional[float] = 2.0
    similarity_threshold_pos: Optional[float] = 0.5
    similarity_threshold_neg: Optional[float] = 0.3
    negative_loss_weight: Optional[float] = 0.3
    enable_w2ml_logging: Optional[bool] = True
    # 기존 W2ML 파라미터 (호환성 유지)
    w2ml_alpha: Optional[float] = 2.0      # W2ML Eq.(6) 파라미터
    w2ml_beta: Optional[float] = 40.0      # W2ML Eq.(7) 파라미터
    w2ml_gamma: Optional[float] = 0.5      # W2ML 마진 파라미터
    difficulty_adaptation_rate: Optional[float] = 0.1
    max_difficulty_weight: Optional[float] = 5.0

@dataclasses.dataclass
class W2MLExperimentConfig:
    """W2ML 실험 전용 설정"""
    config_file: Path
    enable_logging: bool = True
    log_frequency: int = 100
    save_difficulty_history: bool = True
    analysis_window_size: int = 50
    hard_sample_threshold: float = 0.8
    mathematical_verification: bool = True
    # 추가된 필드들 (adapt_config.yaml과 일치)
    track_hard_negatives: Optional[bool] = True
    track_weight_amplification: Optional[bool] = True
    performance_monitoring: Optional[bool] = True
    stability_check: Optional[bool] = True
    progression_tracking: Optional[bool] = True

@dataclasses.dataclass
class ModelSavingConfig:
    """🔥 모델 저장 설정"""
    config_file: Path
    final_save_path: str = "/content/drive/MyDrive/CoCoNut_STAR"
    intermediate_save_frequency: int = 100
    enable_intermediate_save: bool = True
    include_timestamp: bool = True
    auto_generate_readme: bool = True

# === 사전 훈련용 설정들 ===
@dataclasses.dataclass
class TrainingConfig:
    """사전 훈련용 Training 설정"""
    config_file: Path
    batch_size: int
    epoch_num: int
    lr: float
    redstep: int
    gpu_id: int

@dataclasses.dataclass  
class PathsConfig:
    """사전 훈련용 경로 설정"""
    config_file: Path
    checkpoint_path: str
    results_path: str
    save_interval: int
    test_interval: int