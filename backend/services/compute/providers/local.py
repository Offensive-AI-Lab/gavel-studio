"""LocalProvider — in-process torch on the best available device
(CUDA → MPS → CPU). The always-present floor every other provider falls back to.

Delegates to the existing local code paths verbatim, so behavior is identical to
the pre-abstraction "local" branch.
"""
from typing import Callable, List, Optional

from ..base import Accelerator, Capabilities, ComputeProvider, InferenceSpec


class LocalProvider(ComputeProvider):
    name = "local"

    def _device_type(self) -> str:
        try:
            from utils.device import get_torch_device
            return get_torch_device().type  # "cuda" | "mps" | "cpu"
        except Exception:
            return "cpu"

    def capabilities(self) -> Capabilities:
        dev = self._device_type()
        acc = {"cuda": Accelerator.CUDA, "mps": Accelerator.MPS}.get(dev, Accelerator.CPU)
        detail = {
            "cuda": "Local CUDA GPU",
            "mps": "Local Apple GPU (MPS)",
            "cpu": "Local CPU (no GPU — slow)",
        }.get(dev, "Local")
        return Capabilities(
            name=self.name, accelerator=acc, is_local=True,
            supports_training=True, supports_inference=True, supports_realtime=True,
            max_realtime_sessions=1, detail=detail,
        )

    def is_available(self) -> bool:
        return True  # local torch is always there (CPU at worst)

    # --- inference (calibration + evaluation) ---
    def run_inference(self, spec: InferenceSpec, on_phase: Optional[Callable] = None,
                      on_submit: Optional[Callable] = None) -> List[dict]:
        from evaluation.inference import run_inference_on_dialogues

        if on_phase:
            on_phase(f"Running inference locally on {len(spec.dialogues)} conversations…")

        # inference_core emits ("inference", "Dialogue 120/500") periodically;
        # relay it to the live phase line (same wording as the old local path).
        def _infer_progress(stage, detail=""):
            if on_phase and detail:
                if stage and stage != "inference":
                    on_phase(f"Running inference locally — {detail.lower()} ({stage})")
                else:
                    on_phase(f"Running inference locally — {detail.lower()}")

        return run_inference_on_dialogues(
            spec.classifier_id, spec.dialogues, progress_callback=_infer_progress,
        )
