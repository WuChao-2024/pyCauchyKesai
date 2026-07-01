# Platform — 平台与 BPU 信息查询

## 概述

「这块板是什么 / BPU 现在什么状态 / CPU 内存温度频率多少」的一组平台原子能力整合在一个类里：Platform（src/pyCauchyKesai/platform.py）。机型无关、不依赖加载模型——`Platform()` 即可独立使用；CauchyKesai 构造时持有一份 `m.platform`。


```python
from pyCauchyKesai import Platform, CauchyKesai

p = Platform()
p.soc_name            # 'D-Robotics RDK S600 MCB V0p2'
p.bpu_type            # 'nash-p'
p.physical_core_num   # 4
p.bpu_rate()          # [0, 0, 0, 0]   result[i]=核 i 占用率，0-100
p.temperature()       # {'pvt_bpu_pvtc1_t1': 43.5, …}  传感器名:°C；传名只读该条
p.voltage()           # {'VDD_CPU': 900, …}  电源轨名:mV；传名只读该条
p.ion_memory()        # {'ion_cma': {'used':..,'total':..}, …}  字节；÷1048576 得 MB
p.s()                 # 美观打印平台信息（内部 print，返回 None）

m = CauchyKesai("model.hbm")
m.platform.soc_name        
m.summary()["platform"]  
```

---

## 原子能力一览（15 项）

### 静态属性

只读 `@property`，构造时缓存一次。

| 属性 | 返回 | 数据源 | 实现 |
|---|---|---|---|
| soc_name | str | hbUCPGetSocName()（芯片直读） | C 桥 |
| bpu_type | str | 由 soc_name 推断（nash-e/nash-m/nash-p，未知 ''） | Python |
| dnn_version | str | hbDNNGetVersion() | C 桥 |
| bpu_version | str | hbUCPGetVersion()（BPU 运行时版本） | C 桥 |
| physical_core_num | int | /sys/devices/system/bpu/core_num（回退数 bpu* 目录）；读不到 -1 | Python |
| cpu_model | str | /proc/cpuinfo model name（如 Cortex-A78AE） | Python |
| cpu_count | int | /proc/cpuinfo processor 段数 | Python |
| mem_total_mb | int | /proc/meminfo MemTotal（kB→MB）；读不到 -1 | Python |

### 实时方法

每次现读 sysfs / debugfs。均无异常——读失败时返回空列表或默认值（占 0、-1、空 list、空 dict），不抛。

```python
def bpu_rate(self) -> list[int]:
def bpu_freq(self) -> list[int]:
def cpu_freq(self) -> list[int]:
def temperature(self, name: str = "") -> dict:
def voltage(self, name: str = "") -> dict:
def ion_memory(self, name: str = "") -> dict:
def ucp_library_path(self) -> str:
```

| 方法 | 返回 | 数据源 |
|---|---|---|
| bpu_rate() | list[int] | bpu*/ratio，按核号索引，0-100（0=空闲） |
| bpu_freq() | list[int] MHz | bpu*/devfreq/*/cur_freq（Hz→MHz），按核号索引 |
| cpu_freq() | list[int] MHz | cpu*/cpufreq/scaling_cur_freq（kHz→MHz），按 CPU 号索引 |
| temperature(name="") | dict | thermal_zone*/{type,temp}，{传感器名:°C}（毫度÷1000）；name 命中只读该条，未命中 {name:-1} |
| voltage(name="") | dict | hwmon*/in*_label+in*_input，{电源轨名:mV}，跳过 label 为 NULL 的轨；name 同 temperature |
| ion_memory(name="") | dict | debugfs /sys/kernel/debug/ion/heaps/{ion_cma,cma_reserved,carveout}，{heap:{used,total}}（字节）；name 命中只读该条，未命中 {name:-1} |
| ucp_library_path() | str | /proc/self/maps，当前进程加载的 libhbucp.so 绝对路径（未加载 / 不可读 ''） |

### 辅助方法

```python
def summary(self) -> dict:
def s(self) -> None:            # 美观打印（内部 print）
def __repr__(self) -> str:
```

| 方法 | 返回 | 说明 |
|---|---|---|
| summary() | dict | 静态属性 + 实时测量完整快照（每次现读 sysfs）；始终返回 dict |
| s() | None | 美观打印平台信息（静态 + Realtime 段，内部 print） |
| __repr__() | str | 如 Platform(soc='...', bpu_type='nash-p', bpu_cores=4, cpu='Cortex-A78AE' x18, mem=43503MB) |

> 15 项里 C 桥 3 个（soc_name / dnn_version / bpu_version），其余 12 项纯 Python。

---

## 平台对照表（实测）

| 板卡 | soc_name | bpu_type | 物理核数 |
|---|---|---|---|
| S100 | D-Robotics RDK S100 V0P5 | nash-e | 1 |
| S100P | D-Robotics RDK S100P V1P0 | nash-m | 1 |
| S600 | D-Robotics RDK S600 MCB V0p2 | nash-p | 4 |

板卡↔架构↔核数对应固定（OE 文档 key_concept.md），故 bpu_type / physical_core_num 在同一块板上恒定。

---

## 注意事项

1. 机型无关、不依赖模型：`Platform()` 未 load 任何 .hbm 时即可用（3 个 SDK 查询无参、芯片直读）。CauchyKesai 构造时持有一份 `m.platform`。
2. `bpu_type` 是推断值（运行时无 SDK 直读 march）：由 soc_name 按板卡映射推断（S600→nash-p、S100→nash-e、S100P→nash-m）。权威机型信息看 soc_name。
3. `ratio` 是内核周期性统计（非瞬时）：亚毫秒级小模型常读到 0；观察负载需持续打一会。
4. `bpu_core_num` ≠ `physical_core_num`：前者是模型编译核数（张量并行宽度，在 CauchyKesai 上）；后者才是板子物理核数。
5. 单核板（S100/S100P）上 `bpu_rate()` / `bpu_freq()` 返回长度 1 的列表。
6. `bpu_fw_version` 在 CauchyKesai 上（固件首次加载模型时才初始化）：`m.summary()['bpu_fw_version']` 或模块级 `read_bpu_fw_version()`（读 bpu0/fw_version，如 1.1.26）。Platform 不暴露。
7. `temperature(name)` / `voltage(name)`：均返回 `{名: 值}` 扁平 dict；`name=""` 返回全部，命中返回单条 `{名: 值}`，未命中 `{名: -1}`。temperature 毫度÷1000→°C，voltage 已是 mV（不做名称校验）。
8. `ion_memory(name)`：读 debugfs（需 root，部署环境即 root），固定三块 heap（ion_cma/cma_reserved/carveout），返回字节（÷1048576 得 MB），name 语义同上。注意 S600 上 BPU 推理 buffer 实际落在 ion_uncache heap（不在本组三块内），监控 BPU 显存需另读该 heap。
9. `summary()` / `s()` 含实时项（不只是静态）：`summary()` 返回 8 个静态属性 + 7 个实时方法（`bpu_rate`/`bpu_freq`/`cpu_freq`/`temperature`/`voltage`/`ion_memory`/`ucp_library_path`）的输出 + `mem_available_mb`；`s()` 据此渲染——静态段（含 `UCP Library`）+ Realtime 段（BPU 占用率、温度仅 `pvt_bpu_pvtc1_t1`、ION used/total (pct)、内存 used/total (pct)）。BPU/CPU 频率、电压、静态 Memory 总量在 `s()` 不显示，但 dict 里仍有全量（温度/电压分别约 19/28 项）。每次调用现读 sysfs/proc，数 ms 级，非热路径可放心调用。
10. `ucp_library_path()`：读 `/proc/self/maps` 给出当前进程加载的 `libhbucp.so` 绝对路径。诊断库来源——指向 `site-packages/pyCauchyKesai/lib/` = 包自带库；指向系统 `/usr/lib*/` 则是被 OE SDK 替换过的（配合 cn_install.md「OE SDK 替换」判断）。
