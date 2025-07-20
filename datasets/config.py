# datasets/config.py - 데이터셋 설정
"""
COCONUT Dataset Configuration

DESIGN PHILOSOPHY:
- Flexible dataset configuration
- Support for various palmprint datasets
- Preprocessing parameters
"""

import dataclasses
from pathlib import Path
from typing import Optional

@dataclasses.dataclass
class DatasetConfig:
    """데이터셋 설정"""
    config_file: Path
    type: str
    height: int
    width: int
    use_angle_normalization: bool
    # 사전 훈련용
    train_set_file: Optional[str] = None
    test_set_file: Optional[str] = None
    # 온라인 적응용
    dataset_path: Optional[Path] = None