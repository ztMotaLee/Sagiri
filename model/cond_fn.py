from typing import overload, Optional
import torch
from torch.nn import functional as F
from pytorch_msssim import ssim
class Guidance:

    def __init__(
        self,
        scale: float,
        t_start: int,
        t_stop: int,
        space: str,
        repeat: int
    ) -> "Guidance":
        """
        Initialize latent image guidance.
        
        Args:
            scale (float): Gradient scale (denoted as `s` in our paper). The larger the gradient scale, 
                the closer the final result will be to the output of the first stage model.
            t_start (int), t_stop (int): The timestep to start or stop guidance. Note that the sampling 
                process starts from t=1000 to t=0, the `t_start` should be larger than `t_stop`.
            space (str): The data space for computing loss function (rgb or latent).
            repeat (int): Repeat gradient descent for `repeat` times.

        Our latent image guidance is based on [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior).
        Thanks for their work!
        """
        self.scale = scale
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat
    
    def load_target(self, target: torch.Tensor) -> torch.Tensor:
        self.target = target

    def __call__(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Optional[torch.Tensor]:
        if self.t_stop < t and t < self.t_start:
            # print("sampling with classifier guidance")
            # avoid propagating gradient out of this scope
            pred_x0 = pred_x0.detach().clone()
            target_x0 = target_x0.detach().clone()
            return self.scale * self._forward(target_x0, pred_x0)
        else:
            return None
    
    @overload
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
        ...


class MSEGuidance(Guidance):
    
    def __init__(
        self,
        scale: float,
        t_start: int,
        t_stop: int,
        space: str,
        repeat: int
    ) -> "MSEGuidance":
        super().__init__(
            scale, t_start, t_stop, space, repeat
        )
    
    @torch.enable_grad()
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
        pred_x0.requires_grad_(True)

        # L2 Loss
        l2_loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()

        # FFT Loss
        fft_loss = torch.mean(torch.abs(torch.fft.fft2(pred_x0) - torch.fft.fft2(target_x0)))

        # SSIM Loss
        ssim_loss = 1 - ssim(pred_x0, target_x0, data_range=2, size_average=True)

        combined_loss = l2_loss + fft_loss + ssim_loss

        print(f"Combined loss = {combined_loss.item()}")
        return -torch.autograd.grad(combined_loss, pred_x0)[0]