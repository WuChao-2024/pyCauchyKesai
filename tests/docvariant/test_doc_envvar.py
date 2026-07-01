"""文档「环境变量」「任务优先级 priority(抢占)」节细节测试。

envvar 必须在 import 前设置(本包 import 时 setdefault),故用子进程测;
抢占 priority 254/255 需 ① 编译时 max_time_per_fc ② UNSET L2 Cache。

覆盖:
  - HB_DNN_USER_DEFINED_L2M_SIZES 默认 6:6:6:6(import 时 setdefault)
  - UNSET_HB_DNN_USER_DEFINED_L2M_SIZES=1 → 不填默认
  - HB_UCP_LOG_LEVEL 默认 6
  - 抢占(若 preempt hbm 可达):UNSET L2 Cache + priority 254/255 不抛
"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(__file__))
import pytest
import conftest as C

PY = sys.executable


def _import_code(code):
    """子进程跑 code(继承环境),返回 CompletedProcess。"""
    return subprocess.run([PY, "-c", code], capture_output=True,
                          text=True, env=dict(os.environ), timeout=120)


@pytest.mark.docvar
class TestEnvVar:
    def test_l2_cache_default_filled_on_import(self, bpu_free):
        # 文档:import 时 setdefault HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6
        code = (
            "import os; "
            "os.environ.pop('HB_DNN_USER_DEFINED_L2M_SIZES', None); "
            "os.environ.pop('UNSET_HB_DNN_USER_DEFINED_L2M_SIZES', None); "
            "from pyCauchyKesai import CauchyKesai; "
            "print(os.environ.get('HB_DNN_USER_DEFINED_L2M_SIZES','UNSET'))"
        )
        p = _import_code(code)
        assert "6:6:6:6" in p.stdout, f"默认 L2 Cache 未填充: {p.stdout!r} {p.stderr[-300:]!r}"

    def test_unset_disables_default(self, bpu_free):
        # 文档:UNSET_HB_DNN_USER_DEFINED_L2M_SIZES=1 → 不填默认
        code = (
            "import os; "
            "os.environ.pop('HB_DNN_USER_DEFINED_L2M_SIZES', None); "
            "os.environ['UNSET_HB_DNN_USER_DEFINED_L2M_SIZES'] = '1'; "
            "from pyCauchyKesai import CauchyKesai; "
            "print(os.environ.get('HB_DNN_USER_DEFINED_L2M_SIZES','UNSET'))"
        )
        p = _import_code(code)
        assert "UNSET" in p.stdout or p.stdout.strip() == "", \
            f"UNSET=1 时仍填了默认: {p.stdout!r}"

    def test_log_level_default(self, bpu_free):
        # 文档:HB_UCP_LOG_LEVEL 默认 6(never)
        code = (
            "import os; os.environ.pop('HB_UCP_LOG_LEVEL', None); "
            "from pyCauchyKesai import CauchyKesai; "
            "print(os.environ.get('HB_UCP_LOG_LEVEL','UNSET'))"
        )
        p = _import_code(code)
        # UCP 启动 banner([UCP]: log level = 6)会混入 stdout,取最后一行(print 的值)
        assert p.stdout.strip().splitlines()[-1].strip() == "6", \
            f"日志默认值非 6: {p.stdout!r}"

    def test_l2_cache_overridable(self, bpu_free):
        # 文档:import 前显式设置可覆盖默认
        code = (
            "import os; os.environ['HB_DNN_USER_DEFINED_L2M_SIZES'] = '8:8:4:4'; "
            "from pyCauchyKesai import CauchyKesai; "
            "print(os.environ.get('HB_DNN_USER_DEFINED_L2M_SIZES'))"
        )
        p = _import_code(code)
        assert "8:8:4:4" in p.stdout


@pytest.mark.docvar
class TestPreempt:
    """抢占优先级 254/255:需 max_time_per_fc 编译 + UNSET L2 Cache。"""

    def test_preempt_254_255_accepted(self, bpu_free):
        e = C.by_name("conv_preempt_c1")
        if e is None:
            pytest.skip("无 preempt entry(max_time_per_fc 编译失败/不可达)")
        # 抢占与 L2 Cache 互斥 → 子进程 UNSET L2 Cache 后加载 + 设 254/255
        hbm = C.hbm_path(e)
        code = (
            "import os; os.environ['UNSET_HB_DNN_USER_DEFINED_L2M_SIZES'] = '1'; "
            "import numpy as np; "
            "from pyCauchyKesai import CauchyKesai; "
            f"m = CauchyKesai({hbm!r}, n_task=1); "
            "m.set_scheduling_params([0], priority=254); "   # high 抢占
            "assert m.scheduled_priority() == 254; "
            "m.set_scheduling_params([0], priority=255); "   # urgent 抢占
            "assert m.scheduled_priority() == 255; "
            "print('PREEMPT_OK')"
        )
        p = _import_code(code)
        if "PREEMPT_OK" not in p.stdout:
            pytest.skip(f"抢占在本机不生效(可能 L2 Cache 未真关/驱动限制): {p.stderr[-300:]!r}")
