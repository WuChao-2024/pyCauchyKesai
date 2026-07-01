"""文档「信息查询 summary/benchmark/只读属性」节细节测试。

覆盖:
  - summary() 返原生 dict,字段齐全(model_path/model_names/n_task/memory_mb/
    compile_config/bpu_core_num/bpu_fw_version/scheduled_cores/scheduled_priority/
    inputs/outputs/platform)
  - input/output dict 字段齐全(index/name/dtype/tensorType/shape/alignedByteSize/
    quantiType/quantizeAxis/scale/zero_point/stride/desc)
  - summary() 可 json.dumps(AI-Native 契约,numpy 转原生)
  - compile_config dict|None
  - s()/t() 返 None(内部 print)
  - benchmark() 返 dict,time_*>0
  - 只读属性 input_names/output_names/bpu_core_num/n_task/input_count/output_count/platform
  - bpu_core_num == 编译 yaml 的 core
  - platform 子 dict
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import conftest as C

SUMMARY_KEYS = {"model_path", "model_names", "n_task", "memory_mb", "compile_config",
                "bpu_core_num", "bpu_fw_version", "scheduled_cores", "scheduled_priority",
                "inputs", "outputs", "platform"}
IO_KEYS = {"index", "name", "dtype", "tensorType", "shape", "alignedByteSize",
           "quantiType", "quantizeAxis", "scale", "zero_point", "stride", "desc"}


@pytest.mark.docvar
class TestMetadata:
    def test_summary_keys_complete(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        s = m.summary()
        missing = SUMMARY_KEYS - set(s.keys())
        assert not missing, f"summary 缺字段: {missing}"

    def test_summary_json_serializable(self, factory, bpu_free):
        # 文档:summary() numpy 已转原生,可 json.dumps
        m = factory(C.ENTRIES[0])
        json.dumps(m.summary())  # 不抛即通过

    def test_io_dict_keys_complete(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        s = m.summary()
        for io in s["inputs"] + s["outputs"]:
            missing = IO_KEYS - set(io.keys())
            assert not missing, f"IO dict 缺字段: {missing}"
            assert io["quantiType"] in (0, 1)

    def test_compile_config_dict_or_none(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        cc = m.summary()["compile_config"]
        assert cc is None or isinstance(cc, dict)

    def test_s_t_return_none(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        assert m.s() is None
        assert m.t() is None

    def test_benchmark_dict_positive(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        b = m.benchmark()
        for k in ("time_us", "time_ms", "time_s", "time_min"):
            assert k in b
        assert b["time_us"] > 0
        assert b["time_ms"] > 0

    def test_benchmark_big_timeout_no_raise(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        b = m.benchmark(timeout_ms=60000)
        assert b["time_us"] > 0

    def test_readonly_attrs_match_summary(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        s = m.summary()
        assert m.input_names == [i["name"] for i in s["inputs"]]
        assert m.output_names == [o["name"] for o in s["outputs"]]
        assert m.bpu_core_num == s["bpu_core_num"]
        assert m.n_task == s["n_task"]
        assert m.input_count == len(s["inputs"])
        assert m.output_count == len(s["outputs"])
        assert m.platform is not None

    def test_bpu_core_num_matches_yaml_all_entries(self, factory, bpu_free):
        for e in C.ENTRIES:
            m = factory(e)
            assert m.bpu_core_num == e["yaml"].get("core", 1)

    def test_platform_subdict(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        p = m.summary()["platform"]
        assert "soc_name" in p
        assert "bpu_type" in p
        assert "physical_core_num" in p

    def test_memory_mb_positive(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        assert m.summary()["memory_mb"] > 0

    def test_model_names_structure(self, factory, bpu_free):
        m = factory(C.ENTRIES[0])
        mns = m.summary()["model_names"]
        assert len(mns) >= 1
        for mn in mns:
            assert "index" in mn and "name" in mn and "selected" in mn
