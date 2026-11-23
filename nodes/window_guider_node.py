"""[N3] ShotVaultWindowGuider: <=30s ContextWindows (X0 anchoring) & 1st+2nd asymmetric H3 guider."""

from __future__ import annotations

from ..core.context_window_guider import build_window_guider_and_latent


class ShotVaultWindowGuider:
    """[N3] Joint in-step ContextWindows X0 anchoring + 1st+2nd asymmetric guides + sigma release."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "pass_stage": (["single_or_1st", "2nd_refine"], {"default": "single_or_1st"}),
                "window_mode": (["auto_over_260f", "auto_over_192f", "force_on", "off"], {"default": "auto_over_260f"}),
                "window_frames": ("INT", {"default": 260, "min": 124, "max": 362, "step": 17}),
                "window_overlap": ("INT", {"default": 39, "min": 22, "max": 107, "step": 17}),
                "continuation_frames": (
                    ["auto", "auto+auto", "auto+5", "auto+0", "22+5", "39+5", "22+22", "39+39", "22+0", "39+0"],
                    {"default": "auto"},
                ),
                "keyframe_until_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_feather_ticks": ("INT", {"default": 7, "min": 0, "max": 64, "step": 1}),
                "rebase_2nd_pass_audio": ("BOOLEAN", {"default": True}),
                "lock_target_audio": ("BOOLEAN", {"default": False}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "tapered_prev_images": ("IMAGE",),
                "prev_audio": ("AUDIO",),
                "prev_tail_latent": ("LATENT",),
                "shot_audio_guide": ("AUDIO",),
                "shot_pipe": ("SHOT_PIPE",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("GUIDER", "LATENT")
    RETURN_NAMES = ("guider", "latent")
    FUNCTION = "prepare_guider"
    CATEGORY = "Acheng-ShotVault/Signal"

    def prepare_guider(
        self,
        model,
        positive,
        latent,
        pass_stage: str,
        window_mode: str,
        window_frames: int,
        window_overlap: int,
        continuation_frames: str,
        keyframe_until_sigma: float,
        audio_feather_ticks: int = 7,
        rebase_2nd_pass_audio: bool = True,
        lock_target_audio: bool = False,
        cfg: float = 1.0,
        vae=None,
        audio_vae=None,
        tapered_prev_images=None,
        prev_audio=None,
        prev_tail_latent=None,
        shot_audio_guide=None,
        shot_pipe=None,
        negative=None,
    ):
        guider, out_latent = build_window_guider_and_latent(
            model=model,
            positive=positive,
            latent=latent,
            vae=vae,
            audio_vae=audio_vae,
            tapered_prev_images=tapered_prev_images,
            prev_audio=prev_audio,
            prev_tail_latent=prev_tail_latent,
            shot_audio_guide=shot_audio_guide,
            shot_pipe=shot_pipe,
            pass_stage=pass_stage,
            window_mode=window_mode,
            window_frames=window_frames,
            window_overlap=window_overlap,
            continuation_frames=continuation_frames,
            keyframe_until_sigma=keyframe_until_sigma,
            lock_target_audio=lock_target_audio,
            cfg=cfg,
            negative=negative,
            audio_feather_ticks=audio_feather_ticks,
            rebase_2nd_pass_audio=rebase_2nd_pass_audio,
        )
        return (guider, out_latent)
