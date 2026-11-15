"""
Pydantic Request and Response Schemas for EEG Emotion Recognition API
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_architecture: str
    eeg_channels: int
    sampling_rate_hz: int
    target_dimensions: List[str]


class CircumplexInfo(BaseModel):
    quadrant: str
    emotional_state: str
    theme_color: str


class PredictionValues(BaseModel):
    Valence: float = Field(..., ge=1.0, le=9.0)
    Arousal: float = Field(..., ge=1.0, le=9.0)
    Dominance: float = Field(..., ge=1.0, le=9.0)
    Liking: float = Field(..., ge=1.0, le=9.0)


class TrajectoryData(BaseModel):
    window_indices: List[int]
    valence: List[float]
    arousal: List[float]
    dominance: List[float]
    liking: List[float]


class PredictionResponse(BaseModel):
    success: bool
    predictions: PredictionValues
    uncertainty_std: Dict[str, float]
    circumplex: CircumplexInfo
    windows_analyzed: int
    trajectory: Optional[TrajectoryData] = None
    disclaimer: str


class SampleTrialRequest(BaseModel):
    sample_filename: str = "s01.dat"
    trial_index: int = 0


class SampleItem(BaseModel):
    filename: str
    participant_id: str
    num_trials: int
    description: str


class SamplesListResponse(BaseModel):
    samples: List[SampleItem]
