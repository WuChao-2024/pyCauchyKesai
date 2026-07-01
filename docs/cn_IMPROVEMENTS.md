# pyCauchyKesai 改进待办 (IMPROVEMENTS)

记录尚未落地、计划统一处理的改进项。每条含：现状 / 方案 / 影响面 / 验证 / 注意。

落地一条就把它的「状态」改成已完成（或直接删条目）。

## 待办列表

| 编号 | 标题 | 优先级 | 状态 |
|------|------|--------|------|
| IMPRO-001 | RDK_CHECK_SUCCESS 异常带出底层错误码 | 中 | 待办 |


## IMPRO-001 — RDK_CHECK_SUCCESS 异常带出底层错误码

**现状**

`RDK_CHECK_SUCCESS` 宏（`csrc/nash/include/cauchykesai/pycauchykesai.h:18-27`）捕获了底层调用的返回码 `ret_code`，但只判 `!= 0` 就抛**固定**的 errmsg，ret_code 被丢弃：

```cpp
auto ret_code = value;                 // 捕获到码（如 -200002）
if (ret_code != 0) {
    throw std::runtime_error(errmsg);  // errmsg 是写死的串，码被丢
}
```

后果：Python 侧看到的是通用 `RuntimeError`，**无法区分超时和其他失败**。例如 `wait` 超时，底层返回 `HB_UCP_TASK_TIMEOUT`(-200002)，但 Python 只看到 `RuntimeError: hbUCPWaitTaskDone failed`，和 run failed(-200003)、mem flush fail(-400003) 完全一样。

涉及 14 个调用点（全在 `csrc/nash/cauchy_kesai.cpp`）：模型加载、`hbDNNGetInputCount` / `hbDNNGetInputName` / `hbDNNGetOutputCount` / `hbDNNGetOutputName`、`hbUCPSubmitTask`、`hbUCPWaitTaskDone`、`hbUCPReleaseTask`。

> 注：`docs/cn_CauchyKesai_Auto.md` 的 wait 节目前写「返回 HB_UCP_TASK_TIMEOUT（-200002）」——这条表述**待本改进落地后才成立**（落地前 Python 侧不可见具体码）。落地时一并修正措辞。

**方案**

宏里把 `ret_code` 拼进 errmsg，并调 SDK 的 `hbUCPGetErrorDesc(ret_code)`（`csrc/nash/include/hobot/hb_ucp_status.h:80`，返回自然语言描述）拿可读名。改后形如：

```
RuntimeError: hbUCPWaitTaskDone failed (ret=-200002, task timeout ...)
```

示意：

```cpp
#define RDK_CHECK_SUCCESS(value, errmsg)                          \
  do {                                                            \
    auto ret_code = (value);                                      \
    if (ret_code != 0) {                                          \
      throw std::runtime_error(                                   \
          std::string(errmsg) + " (ret=" + std::to_string(ret_code) \
          + ", " + hbUCPGetErrorDesc(ret_code) + ")");            \
    }                                                             \
  } while (0);
```

**影响面**

14 个调用点统一改善。按项目约定（内部更新、不照顾历史版本），改异常消息无兼容负担。

**验证**

落地后重跑超时测试：用小模型（如 siglip，t_inf≈17ms），`start` 后 `wait(timeout_ms=1)`（1ms << t_inf），断言抛出的 `RuntimeError` 消息里**含 -200002**。这一步同时：
- 确认超时确实返回 `HB_UCP_TASK_TIMEOUT`(-200002)（之前文档里的码是推断、未在 Python 侧验证过）。
- 证明改进生效。

**注意**

宏也用于 `hbDNN*` 函数，需确认 `hbUCPGetErrorDesc` 对 hbDNN 返回码也能给出合理描述。`hb_dnn_status.h:29` 里 `HB_DNN_TIMEOUT = HB_UCP_TASK_TIMEOUT`，码空间共用，应可；实现时验证一下未知码的回退（`hbUCPGetErrorDesc` 对陌生码返回什么，避免拼出空串或乱码）。



# CauchyKesai 实例化太慢，速度有些慢


# Plantform改为纯C实现(待定), 避免每一个 CauchyKesai 都持有一份



# IONArrayDesc实例化 IONArray 也改 RAII 引用，避免重复持有