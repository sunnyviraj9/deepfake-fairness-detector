"""
Frequency-Domain Deepfake Detector using 2D Fast Fourier Transform (FFT) and High-Pass Filtering.
Detects periodic synthesis artifacts, upsampling spectral anomalies, and high-frequency discrepancies.
"""

from typing import Optional, Tuple
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyExtractor(nn.Module):
    """
    Differentiable 2D Fast Fourier Transform and Spectral Filter Module.
    Converts RGB/Grayscale images to shifted Log-Magnitude Spectra and High-Pass filtered residuals.
    """

    def __init__(self, high_pass_cutoff: float = 0.1):
        """
        Args:
            high_pass_cutoff: Normalized frequency radius [0.0, 0.5] below which frequencies are attenuated.
        """
        super().__init__()
        self.high_pass_cutoff = high_pass_cutoff

    def compute_fft_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes centered 2D FFT log-magnitude spectrum.
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            Normalized log magnitude spectrum of shape (B, C, H, W).
        """
        # Compute 2D FFT
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        # Log magnitude spectrum: log(1 + |FFT|)
        magnitude = torch.abs(fft_shift)
        log_mag = torch.log1p(magnitude)

        # Batch-wise instance normalization for stable gradient flow
        mean = log_mag.mean(dim=(-2, -1), keepdim=True)
        std = log_mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_mag = (log_mag - mean) / std

        return norm_mag

    def apply_high_pass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies frequency-domain high-pass mask to attenuate low frequency face geometry and preserve artifacts.
        """
        B, C, H, W = x.shape
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))

        # Create radial frequency meshgrid
        Y, X = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=x.device),
            torch.linspace(-0.5, 0.5, W, device=x.device),
            indexing="ij",
        )
        radius = torch.sqrt(X**2 + Y**2)
        # High pass mask: 1 for high frequencies, 0 for DC/low frequencies
        hp_mask = (radius >= self.high_pass_cutoff).float().unsqueeze(0).unsqueeze(0)

        filtered_fft_shift = fft_shift * hp_mask
        filtered_fft = torch.fft.ifftshift(filtered_fft_shift, dim=(-2, -1))
        hp_residual = torch.fft.ifft2(filtered_fft, dim=(-2, -1)).real

        return hp_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns combined frequency representation:
        - 3 channels of Log-Magnitude Spectrum
        - 3 channels of High-Pass Spatial Residuals
        Total 6-channel frequency tensor: (B, 6, H, W)
        """
        log_mag = self.compute_fft_magnitude(x)
        hp_res = self.apply_high_pass_filter(x)
        return torch.cat([log_mag, hp_res], dim=1)


class FrequencyDetector(nn.Module):
    """
    Frequency-Domain Deepfake Classifier.
    Transforms spatial face inputs into spectral representations and processes them
    via a dedicated CNN/timm backbone to capture forgery artifacts.
    """

    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        num_classes: int = 1,
        dropout: float = 0.3,
        high_pass_cutoff: float = 0.1,
    ):
        """
        Args:
            backbone_name: timm backbone architecture for processing spectral inputs.
            pretrained: Whether to initialize with ImageNet weights (adapted for multi-channel input).
            num_classes: Output logits dimension (1 for binary).
            dropout: Dropout probability.
            high_pass_cutoff: High pass cutoff radius in frequency domain.
        """
        super().__init__()
        self.freq_extractor = FrequencyExtractor(high_pass_cutoff=high_pass_cutoff)

        # Backbone takes 6 input channels (3 log-mag + 3 high-pass residuals)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=6,
            drop_rate=dropout,
        )

        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout / 2.0),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts frequency embeddings of shape (B, feature_dim)."""
        freq_repr = self.freq_extractor(x)
        return self.backbone(freq_repr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input RGB face tensor of shape (B, 3, H, W).
        Returns:
            Logits of shape (B, 1).
        """
        freq_repr = self.freq_extractor(x)
        features = self.backbone(freq_repr)
        logits = self.head(features)
        return logits
