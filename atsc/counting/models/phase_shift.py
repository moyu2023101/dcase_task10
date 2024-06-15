# Copyright 2024 Robert Bosch GmbH.
# SPDX-License-Identifier: GPL-3.0-only

"""Generalized cross-correlation with phase attention."""

import torch
import torchaudio
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
class PhaseShift(nn.Module):
    """Extract GCC-Phat from multi-channel STFT."""

    def __init__(self, n_channels: int | None = None,unwrap: bool | None=None):
        """
        Initialize GCCPhat layer.

        Args:
            max_coeff: maximum number of coefficients, first max_coeff//2 and last max_coeff//2
        """
        super().__init__()
        self.n_channels = n_channels
        self.unwrap=unwrap

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward step for GCCPhat layer.

        Args:
            x: STFT
                STFT: N,ch,T,F complex64 tensor
                    N: batch size
                    ch: number of channels
                    T: time frames
                    F = nfft/2+1

        Returns: N,comb,S,T float32 tensor
                N: number of signals in the batch
                comb: number of channels combinations
                S: max_coeff or nfft
                T: time frames
        """
        N, ch, F, T = x.size()
        assert ch == self.n_channels, "Input channels do not match initialized number of channels."

        # Extract the phase of the complex STFT
        phase = torch.angle(x)  # Shape: (N, ch, F, T)

        # Calculate phase differences
        phase_diff_list = []
        for i in range(ch):
            phase_diff = np.unwrap((phase[:, i, :, :] - phase[:, 0, :, :]).cpu().numpy(),axis=1)
            #print(phase[:, i, :, :].shape)
            #print(phase_diff.shape)
            phase_diff_list.append(torch.tensor(phase_diff))

        # Stack the phase differences
        phase_diff_tensor = torch.stack(phase_diff_list, dim=1)  # Shape: (N, ch*ch, T, F)
        # self.plot_phase_shifts(phase_diff_tensor)
        return phase_diff_tensor

    def plot_phase_shifts(self, phase_shifts: torch.Tensor, fs: int=16000, duration: float=5):
        """
        Plot the phase shifts.

        Args:
            phase_shifts: N, ch, F, T tensor of phase shifts
            fs: sampling frequency
            duration: duration of the signal in seconds
        """
        N, ch, F, T = phase_shifts.shape

        frequencies = torch.linspace(0, fs // 2, steps=F)
        times = torch.linspace(0, duration, steps=T)

        plt.figure(figsize=(12, 12))
        for i in range(1, ch):
            plt.subplot(ch, 1, i)
            plt.pcolormesh(times, frequencies, phase_shifts[0, i, :, :].cpu().numpy(), shading='gouraud')
            plt.title(f'Phase Difference: Channel {i} - Channel 0')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Phase Difference (radians)')

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
class PhaseShift_1(nn.Module):
    """Extract GCC-Phat from multi-channel STFT."""

    def __init__(self, n_channels: int, unwrap: bool = True):
        """
        Initialize GCCPhat layer.

        Args:
            n_channels: number of channels
            unwrap: whether to unwrap the phase
        """
        super().__init__()
        self.n_channels = n_channels
        self.unwrap = unwrap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward step for GCCPhat layer.

        Args:
            x: STFT
                STFT: N, ch, F, T complex64 tensor
                    N: batch size
                    ch: number of channels
                    T: time frames
                    F: nfft/2+1

        Returns:
            Tensor of shape (N, comb, S, T)
                N: number of signals in the batch
                comb: number of channels combinations
                S: max_coeff or nfft
                T: time frames
        """
        N, ch, F, T = x.size()
        assert ch == self.n_channels, "Input channels do not match initialized number of channels."

        # Extract the phase of the complex STFT
        phase = torch.angle(x)  # Shape: (N, ch, F, T)

        # Calculate phase differences
        ref_phase = phase[:, 0, :, :].unsqueeze(1)  # Shape: (N, 1, F, T)
        phase_diff = phase - ref_phase  # Shape: (N, ch, F, T)

        if self.unwrap:
            phase_diff = self.phase_unwrap(phase_diff)
        #self.plot_phase_shifts(phase_diff)
        return phase_diff

    def phase_unwrap(self, phase_diff):
        """
        Unwrap the phase along the specified axis.

        Args:
            phase_diff: Tensor containing phase differences

        Returns:
            Unwrapped phase difference tensor
        """
        # Calculate the difference between adjacent elements
        delta = torch.diff(phase_diff, dim=2)
        # Find the phase jumps
        phase_jumps = torch.round(delta / (2 * torch.pi)) * 2 * torch.pi
        # Accumulate the phase jumps
        phase_jumps_accumulated = torch.cumsum(phase_jumps, dim=2)
        # Create a tensor to add the initial column of zeros back
        phase_jumps_accumulated = torch.cat((torch.zeros_like(phase_diff[:, :, :1]), phase_jumps_accumulated), dim=2)
        # Unwrap the phase by subtracting the accumulated phase jumps
        phase_unwrapped = phase_diff - phase_jumps_accumulated
        return phase_unwrapped
    def plot_phase_shifts(self, phase_shifts: torch.Tensor, fs: int=16000, duration: float=5):
        """
        Plot the phase shifts.

        Args:
            phase_shifts: N, ch, F, T tensor of phase shifts
            fs: sampling frequency
            duration: duration of the signal in seconds
        """
        N, ch, F, T = phase_shifts.shape

        frequencies = torch.linspace(0, fs // 2, steps=F)
        times = torch.linspace(0, duration, steps=T)

        plt.figure(figsize=(12, 12))
        for i in range(1, ch):
            plt.subplot(ch, 1, i)
            plt.pcolormesh(times, frequencies, phase_shifts[0, i, :, :].cpu().numpy(), shading='gouraud')
            plt.title(f'Phase Difference: Channel {i} - Channel 0')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(label='Phase Difference (radians)')

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    model = PhaseShift_1(n_channels=4,unwrap=True)
    # Create a random complex tensor to represent the STFT input
    batch_size=1
    n_channels=4
    audio,sr=torchaudio.load("/home/fanshitong/run/acoustic-traffic-simulation-counting-main/data_root/real_root/loc1/train/00001.flac")
    print(f"{sr} : {audio.unsqueeze(0)[:,:,:sr*5].shape}")
    audio_flat = audio.unsqueeze(0)[:,:,:sr*5].view(batch_size * n_channels, -1)


    # print(f"{sr} : {audio.unsqueeze(0).shape}")
    # audio_flat = audio.unsqueeze(0).view(batch_size * n_channels, -1)

    stft_param={'hop_length': 512, 'return_complex': True, 'n_fft': 1024, 'center': False}
    windows =torch.hann_window(stft_param["n_fft"])
    stft_flat = torch.stft(audio_flat,**stft_param,window=windows)
    n_time_frames = stft_flat.shape[-1]
    stft = stft_flat.view(batch_size, n_channels, -1, n_time_frames)
    print(stft.shape)
    # x = torch.randn(1, 4, 274, 5994, dtype=torch.complex64)  # (N=2, ch=4, T=100, F=257)
    # Pass the input through the model
    output = model(stft)

    print(output.shape)