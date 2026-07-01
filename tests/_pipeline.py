"""测试内部用的多模型零拷贝串联。

原 ``pyCauchyKesai.Pipeline``，已从主包移除（仅测试覆盖需要，仓库内无用户代码引用）。
底层能力由 ``IONArray.memory`` / ``IONArray.from_memory()`` 提供，
用户如需多模型零拷贝可直接基于这两个 API 手动组合。

被以下测试使用：
  - ``tests/test_ai_native.py``（TestPipelineSingle / TestPipelineDual）
  - ``tests/dimensions/_paths.py::run_pipeline``（D4 推理路径一致性矩阵的一列）
"""
from pyCauchyKesai import IONArray


class Pipeline:
    """多模型串联: 上游 output IONArray 的 IONMemory 共享给下游 input。

    中间步用 m.outputs[task_id] 拿 IONArray (含 IONMemory 共享),
    通过 IONArray.from_memory(up.memory, 0, tpl) 构造下游 input。
    """

    def __init__(self, models):
        """构造 Pipeline。

        Args:
            models: list[CauchyKesai], 至少一个模型。
        """
        self.models = list(models)
        if not self.models:
            raise ValueError("Pipeline needs at least one model")

    def pipe(self, inputs, task_id=0):
        """执行多模型串联推理。

        Args:
            inputs: list[np.ndarray], 第一阶段模型的输入。
            task_id: int, 所有模型使用同一个 task slot (默认 0)。
        Returns:
            list[np.ndarray]: 最终模型的输出。
        Raises:
            RuntimeError: 上下游 input/output 不匹配。
        """
        cur_inputs = inputs      # 第一阶段用原始 numpy inputs
        upstream_ions = None
        for i, model in enumerate(self.models):
            if i > 0 and upstream_ions is not None:
                # 零拷贝: 上游 output IONArray 的 IONMemory 共享给下游 input
                n_down_inputs = len(model.input_names)
                n_up_outputs = len(upstream_ions)
                if n_down_inputs != n_up_outputs:
                    raise RuntimeError(
                        f"Pipeline model[{i}]: input count ({n_down_inputs}) != "
                        f"upstream output count ({n_up_outputs}). "
                        f"Ensure models have compatible input/output signatures.")
                inter = []
                for j in range(n_down_inputs):
                    up = upstream_ions[j]
                    tpl = model.input_descs[j]        # 下游 input 模板 desc
                    ion = IONArray.from_memory(up.memory, 0, tpl)
                    inter.append(ion)
                for j, ion in enumerate(inter):
                    model.inputs[task_id][j] = ion   # 纯赋值绑定(零拷贝)
            # i==0: Auto memcpy(start 传 inputs); i>0: 零拷贝(start 不传 inputs 用 bound)
            if i == 0:
                model.start(cur_inputs, task_id=task_id)
            else:
                model.start(task_id=task_id)
            model.wait(task_id=task_id)
            upstream_ions = model.outputs[task_id]    # IONArray (含 IONMemory 共享)
        last = self.models[-1]
        outs = last.outputs[task_id]
        return [o.numpy() for o in outs]
