# Qwen3-VL 视频理解项目 - 改动记录与踩雷总结

## 一、项目概述

基于 Gradio + Qwen3-VL 的视频理解应用，上传视频自动生成 SRT 字幕。

## 二、改动清单

### 2.1 Bug 修复（3个）

#### Bug 1：上下文管理器重复调用 `__exit__`

**文件**：`core/video_processor.py`

**问题**：`_convert_video_to_compatible_mp4` 手动调用 `__enter__` 并返回上下文管理器对象，`process_video_messages` 的 `finally` 块再次调用 `__exit__`，导致临时目录被重复清理。

**修复**：重构 `_convert_video_to_compatible_mp4` 接收 `tmp_dir` 参数，调用方用 `with` 语句管理生命周期。

```python
# 修复前
converted_tmp = None
try:
    ...
except Exception:
    converted_video_path, converted_tmp = self._convert_video_to_compatible_mp4(video_path)
    ...
finally:
    if converted_tmp:
        converted_tmp.__exit__(None, None, None)  # 重复调用

# 修复后
try:
    ...
except Exception:
    with temp_video_dir() as tmp_dir:
        converted_video_path = self._convert_video_to_compatible_mp4(video_path, tmp_dir)
        ...
```

---

#### Bug 2：函数内动态 import

**文件**：`core/subtitle_generator.py:177`

**问题**：使用 `__import__("torch")` 在函数体内动态导入，违反 Python 编码规范。

**修复**：改为模块顶部 `import torch`。

---

#### Bug 3：临时文件泄漏

**文件**：`app.py` 的 `download_srt` 函数

**问题**：`NamedTemporaryFile(delete=False)` 创建的临时 SRT 文件永远不会被清理。

**修复**：
- 添加 `atexit` 注册清理函数
- 每次下载前清理上一个临时文件
- 用列表跟踪已创建的临时文件

---

### 2.2 UI 改进（6个）

| 改进 | 之前 | 之后 |
|------|------|------|
| 历史记录展示 | 原始 JSON 文本框 | 格式化 Markdown 表格 |
| 清空历史 | 无 | 新增按钮 |
| 分析按钮状态 | 无反馈 | 分析中禁用，完成后恢复 |
| SRT 输出框 | 可编辑 | 只读 `interactive=False` |
| 模型状态 | 无 | 顶部实时显示已加载模型 |
| 清除视频 | 无 | 新增按钮 |

---

### 2.3 长视频分段分析

**文件**：`core/video_splitter.py`（新增）

**功能**：视频 >60 秒时自动分段处理，保持上下文一致性。

**流程**：
```
视频 > 60秒？
  ├─ 否 → 正常两步推理
  └─ 是 → ffmpeg 切割为 60 秒片段
         ├─ 第1段：正常分析 → 提取人物/场景信息
         ├─ 第2段：带上下文分析（保持角色名一致）
         ├─ 第3段：继续传递上下文...
         ├─ 合并所有 SRT（时间轴自动偏移）
         └─ 清理所有临时分段文件
```

**关键实现**：
- `get_video_duration()`：用 ffprobe 获取视频时长
- `split_video()`：用 ffmpeg 切割视频
- `merge_srt_segments()`：合并多段 SRT，自动偏移时间轴
- `_extract_context()`：从分析结果中提取人物名称、场景信息
- `cleanup_split_files()`：清理临时文件

---

### 2.4 MPS→CPU 降级

**文件**：`core/model_manager.py`

**问题**：macOS 上 MPS 内存不足时（如分配 27.6GB 连续内存）会直接崩溃。

**修复**：`model.to("mps")` 外包 try/except，失败时保持在 CPU。

```python
if torch.backends.mps.is_available():
    try:
        model = model.to("mps")
    except Exception as e:
        logger.warning(f"MPS allocation failed ({e}), keeping on CPU")
```

---

### 2.5 Gradio 6.0 兼容

**文件**：`app.py`

**问题**：Gradio 6.0 把 `css` 参数从 `Blocks` 构造函数移到了 `launch()` 方法。

**修复**：`css` 参数移至 `launch()` 调用。

---

### 2.6 代码清理

- 移除未使用的 `ANALYSIS_PROMPT` 导入
- 移除未使用的 `glob` 导入
- CSS 标题渐变美化

---

## 三、踩雷总结（重点）

### 雷 1：单次推理生成字幕质量极差

**尝试**：合并分析+字幕为单次推理调用（`DIRECT_SRT_PROMPT`），一步到位生成 SRT。

**结果**：输出垃圾内容——重复、格式混乱（`1 --> 01 --> 00:00:00,000`）、完全不符合视频内容。

**原因**：Qwen3-VL 需要先理解视频内容，再基于理解生成字幕。跳过"理解"步骤直接生成字幕，模型不知道该写什么，只能重复填充。

**结论**：**两步推理不能合并**。速度优化不能以牺牲质量为代价。

---

### 雷 2：`repetition_penalty` 过度导致内容贫乏

**尝试**：添加 `repetition_penalty=1.2` 和 `no_repeat_ngram_size=3` 防止重复。

**结果**：字幕不再重复，但内容质量大幅下降，远不如原始版本。

**原因**：惩罚参数过于激进，限制了模型的正常表达能力。

**结论**：**不要随意添加生成参数**。原始的 `do_sample=False` + `max_new_tokens=1024` 是经过验证的配置。

---

### 雷 3：`max_new_tokens` 减半导致分析不完整

**尝试**：将分析阶段的 `max_new_tokens` 从 1024 减到 512 以加速。

**结果**：分析被截断，后续字幕生成质量下降。

**结论**：**token 数量影响输出质量**，不能随意减少。

---

### 雷 4：stream copy 分割可能不兼容

**尝试**：视频分割用 `-c copy`（不重编码）代替 `-c:v libx264`。

**结果**：部分视频格式不兼容，分割失败。

**结论**：**stream copy 不是万能的**，需要 fallback 到重编码。最终回退到原始重编码方案。

---

### 雷 5：MPS 内存预加载崩溃

**尝试**：启动时预加载默认模型到 MPS。

**结果**：MPS 内存分配失败，应用启动即崩溃。

**原因**：macOS MPS 后端对大模型内存分配有限制。

**结论**：**MPS 预加载不可靠**，改为仅 CUDA 环境预加载，MPS 环境延迟加载。

---

### 雷 6：MPS→CPU 降级不完整

**尝试**：捕获 MPS 异常后设置 `dtype = torch.float32`。

**结果**：模型仍处于不一致状态（部分在 MPS），后续推理崩溃。

**正确做法**：捕获异常后不做任何操作，让模型保持在 CPU。

```python
# 错误
except Exception as e:
    dtype = torch.float32  # 没有实际移动模型

# 正确
except Exception as e:
    logger.warning(f"MPS failed: {e}, keeping on CPU")
    # 模型已经在 CPU 上，无需操作
```

---

## 四、最终保留的优化

| 优化 | 状态 | 说明 |
|------|------|------|
| Bug 修复 | ✅ 保留 | 上下文管理器、torch import、临时文件 |
| UI 改进 | ✅ 保留 | 历史表格、按钮状态、只读输出 |
| 长视频分段 | ✅ 保留 | 自动切割 + 上下文传递 + SRT 合并 |
| MPS→CPU 降级 | ✅ 保留 | 防止内存不足崩溃 |
| 单次推理 | ❌ 回退 | 质量太差 |
| repetition_penalty | ❌ 回退 | 内容贫乏 |
| max_new_tokens 减半 | ❌ 回退 | 分析不完整 |
| stream copy 分割 | ❌ 回退 | 兼容性问题 |
| MPS 预加载 | ❌ 回退 | 内存崩溃 |

## 五、经验教训

1. **性能优化不能以牺牲质量为代价**——用户宁可慢一点也要质量好
2. **LLM 的推理步骤有其内在逻辑**——跳过理解直接生成必然失败
3. **生成参数需要谨慎调整**——默认参数通常是经过验证的最优解
4. **平台差异要考虑周全**——macOS MPS 和 CUDA 行为差异很大
5. **先跑通再优化**——原始版本能用，优化反而引入新问题
