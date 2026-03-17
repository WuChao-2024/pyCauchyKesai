"""Type stubs for pyCauchyKesai module (Bayes / RDK X5)."""

from typing import List, Dict, Any
import numpy as np
import numpy.typing as npt

__version__: str
__author__: str
__date__: str
__doc__: str

class ModelSummary(Dict[str, Any]):
    """
    Model summary dictionary with formatted string representation.

    Keys:
        model_path: str - Path to the model file
        model_names: List[Dict[str, Any]] - List of model names with selection status
        n_task: int - Number of concurrent task slots
        memory_mb: float - Total allocated memory in MB
        inputs: List[Dict[str, Any]] - Input tensor specifications
        outputs: List[Dict[str, Any]] - Output tensor specifications
    """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class BenchmarkResult(Dict[str, Any]):
    """
    Benchmark result dictionary with formatted string representation.

    Keys:
        time_us: float - Inference time in microseconds
        time_ms: float - Inference time in milliseconds
        time_s: float - Inference time in seconds
        time_min: float - Inference time in minutes
    """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class CauchyKesai:
    """
    AI Native inference interface for RDK X5 (Bayes) BPU platform.

    Wraps libDNN (hb_dnn.h / hb_sys.h) for RDK X5.
    Python interface is identical to the Nash (S100) implementation.

    Args:
        model_path: Path to the .bin model file
        n_task: Maximum number of concurrent tasks (1-32, default: 1)
        model_cnt_select: Model index within a packed model (default: 0)

    Raises:
        RuntimeError: If model file does not exist or initialization fails
        UserWarning: If n_task or model_cnt_select are clamped to valid range

    Notes (X5 vs S100 differences):
        - input_tensors are flat (alignedByteSize,) uint8 arrays, not shaped by validShape,
          because X5 validShape may differ from actual memory layout (e.g. NV12 inputs).
        - Uses hbSysAllocCachedMem / hbSysFreeMem / hbSysFlushMem instead of hbUCP* APIs.
        - Uses hbDNNInfer + hbDNNInferCtrlParam instead of hbDNNInferV2 + hbUCPSchedParam.

    Examples:
        >>> model = CauchyKesai("/path/to/model.bin", n_task=4)
        >>> print(model)
        >>> outputs = model([input_array])
    """

    input_tensors: List[List[npt.NDArray[np.uint8]]]
    """
    ION memory views for input tensors.
    Shape: [n_task][input_count]
    Each array is flat (alignedByteSize,) uint8 — zero-copy access to BPU-accessible memory.
    Note: Unlike Nash, shape is NOT validShape due to X5 memory layout differences.
    """

    output_tensors: List[List[npt.NDArray[Any]]]
    """
    ION memory views for output tensors.
    Shape: [n_task][output_count]
    Zero-copy access to BPU output memory, shaped by validShape.
    """

    input_names: List[str]
    """List of input tensor names."""

    output_names: List[str]
    """List of output tensor names."""

    def __init__(
        self,
        model_path: str,
        n_task: int = 1,
        model_cnt_select: int = 0
    ) -> None: ...

    def s(self) -> ModelSummary:
        """Return model summary as a structured dict (also pretty-prints in terminal)."""
        ...

    def t(self) -> BenchmarkResult:
        """Perform a single dirty inference run and return benchmark timing."""
        ...

    def is_busy(self, task_id: int = 0) -> bool:
        """Return True if the given task slot is currently running inference."""
        ...

    def start(
        self,
        inputs: List[npt.NDArray[Any]],
        task_id: int = 0,
        priority: int = 0
    ) -> None:
        """
        Submit an inference task asynchronously.

        Args:
            inputs: List of input numpy arrays. Pass [] for zero-copy mode
                    (pre-write data to input_tensors[task_id][i]).
            task_id: Task slot ID (default: 0)
            priority: BPU scheduling priority 0-255 (default: 0=lowest)
        """
        ...

    def wait(self, task_id: int = 0) -> List[npt.NDArray[Any]]:
        """Wait for inference to complete and return output arrays (zero-copy)."""
        ...

    def inference(
        self,
        inputs: List[npt.NDArray[Any]],
        task_id: int = 0,
        priority: int = 0
    ) -> List[npt.NDArray[Any]]:
        """Validate inputs, then start + wait. Returns output arrays."""
        ...

    def __call__(
        self,
        inputs: List[npt.NDArray[Any]],
        task_id: int = 0,
        priority: int = 0
    ) -> List[npt.NDArray[Any]]:
        """Shorthand for inference()."""
        ...

    def __repr__(self) -> str: ...
