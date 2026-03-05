"""Type stubs for pyCauchyKesai module."""

from typing import List, Dict, Any, Union
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
    AI Native inference interface for Horizon Robotics BPU platform.

    This class provides a Python interface to load and run .hbm neural network
    models on edge AI chips such as Nash-e and Nash-m.

    Args:
        model_path: Path to the .hbm model file
        n_task: Maximum number of concurrent tasks (1-32, default: 1)
        model_cnt_select: Model index within a packed model (default: 0)

    Raises:
        RuntimeError: If model file does not exist or initialization fails
        UserWarning: If n_task or model_cnt_select are clamped to valid range

    Examples:
        >>> model = CauchyKesai("/path/to/model.hbm", n_task=4)
        >>> print(model)  # Shows model info
        >>> outputs = model([input1, input2])  # Run inference
    """

    input_tensors: List[List[npt.NDArray[Any]]]
    """
    ION memory views for input tensors.
    Shape: [n_task][input_count]
    Zero-copy access to BPU-accessible physical memory.
    """

    output_tensors: List[List[npt.NDArray[Any]]]
    """
    ION memory views for output tensors.
    Shape: [n_task][output_count]
    Zero-copy access to BPU output memory.
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
        """
        Return model summary as a structured dict.

        The returned ModelSummary object can be:
        - Printed directly for formatted terminal output
        - Accessed as a dict for programmatic use

        Returns:
            ModelSummary containing model metadata, input/output specs, and memory info

        Examples:
            >>> info = model.s()
            >>> print(info)  # Formatted output
            >>> input_shape = info['inputs'][0]['shape']  # Dict access
        """
        ...

    def t(self) -> BenchmarkResult:
        """
        Perform a single dirty inference run and return benchmark result.

        Uses random data for inputs. The returned BenchmarkResult can be:
        - Printed directly for formatted terminal output
        - Accessed as a dict for programmatic use

        Returns:
            BenchmarkResult containing inference timing in various units

        Examples:
            >>> bench = model.t()
            >>> print(bench)  # Formatted output
            >>> latency_ms = bench['time_ms']  # Dict access
        """
        ...

    def is_busy(self, task_id: int = 0) -> bool:
        """
        Check if a task slot is currently running inference.

        Thread-safe query using atomic operations.

        Args:
            task_id: Task slot ID to query (default: 0)

        Returns:
            True if the task slot is busy, False otherwise

        Raises:
            IndexError: If task_id is out of range [0, n_task-1]

        Examples:
            >>> if not model.is_busy(0):
            ...     model.start(inputs, task_id=0)
        """
        ...

    def start(
        self,
        inputs: List[npt.NDArray[Any]],
        task_id: int = 0,
        priority: int = 0
    ) -> None:
        """
        Submit an inference task asynchronously and return immediately.

        Releases GIL during BPU submission for true multi-threaded concurrency.

        Args:
            inputs: List of input numpy arrays. Pass empty list [] for zero-copy mode
                   (data must be pre-written to input_tensors)
            task_id: Task slot ID (0 to n_task-1, default: 0)
            priority: BPU scheduling priority (0-255, 255=highest, default: 0)

        Raises:
            IndexError: If task_id or priority out of range
            RuntimeError: If BPU submission fails

        Examples:
            >>> # Regular mode
            >>> model.start([y, uv], task_id=0, priority=128)
            >>>
            >>> # Zero-copy mode
            >>> np.copyto(model.input_tensors[0][0], my_data)
            >>> model.start([], task_id=0)
        """
        ...

    def wait(self, task_id: int = 0) -> List[npt.NDArray[Any]]:
        """
        Wait for inference task to complete and return results.

        Releases GIL during wait, allowing other Python threads to run.
        Returns zero-copy views of BPU output memory.

        Args:
            task_id: Task slot ID to wait for (default: 0)

        Returns:
            List of output numpy arrays (zero-copy, pointing to ION memory)

        Raises:
            RuntimeError: If wait or task release fails

        Examples:
            >>> model.start([y, uv], task_id=0)
            >>> outputs = model.wait(task_id=0)
            >>> print(outputs[0].shape)
        """
        ...

    def inference(
        self,
        inputs: List[npt.NDArray[Any]],
        task_id: int = 0,
        priority: int = 0
    ) -> List[npt.NDArray[Any]]:
        """
        Perform synchronous inference: validate + start + wait.

        Validates all inputs before submission. Releases GIL during BPU execution.
        Equivalent to calling start() followed by wait().

        Args:
            inputs: List of input numpy arrays matching model requirements
            task_id: Task slot ID (0 to n_task-1, default: 0)
            priority: BPU scheduling priority (0-255, default: 0)

        Returns:
            List of output numpy arrays

        Raises:
            IndexError: If task_id or priority out of range
            ValueError: If input count, dtype, ndim, or shape mismatch
            RuntimeError: If task slot is already in use or BPU call fails

        Examples:
            >>> y = np.random.randint(0, 255, (1, 640, 640, 1), dtype=np.uint8)
            >>> uv = np.random.randint(0, 255, (1, 320, 320, 2), dtype=np.uint8)
            >>> outputs = model.inference([y, uv])
            >>> print(outputs[0].shape)
        """
        ...

    def __call__(
        self,
        inputs: List[npt.NDArray[Any]],
        task_id: int = 0,
        priority: int = 0
    ) -> List[npt.NDArray[Any]]:
        """
        Shorthand for inference(). Allows calling model object directly.

        Args:
            inputs: List of input numpy arrays
            task_id: Task slot ID (default: 0)
            priority: BPU scheduling priority (default: 0)

        Returns:
            List of output numpy arrays

        Examples:
            >>> outputs = model([y, uv])  # Same as model.inference([y, uv])
        """
        ...

    def __repr__(self) -> str:
        """
        Return concise model representation showing key info and busy status.

        Examples:
            >>> print(model)
            CauchyKesai(model='yolov8n_640x640_nv12', inputs=2, outputs=6, n_task=4)
        """
        ...
