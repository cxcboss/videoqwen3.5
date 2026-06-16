# Qwen3-VL 视频理解项目 - 开发文档

> 最后更新：2026-06-15
> 本项目基于 Gradio + Qwen3-VL 实现视频自动生成 SRT 字幕

---

## 一、项目结构

```
视频理解千问3.5_副本/
├── app.py                    # 主入口，Gradio UI + 业务逻辑
├── config.py                 # 配置管理（模型、视频参数、Prompt 模板）
├── core/
│   ├── model_manager.py      # 模型加载、缓存、GPU 内存管理
│   ├── video_processor.py    # 视频处理、格式转换、消息构建
│   ├── subtitle_generator.py # SRT 字幕生成、时间轴标准化
│   ├── history_manager.py    # 分析历史记录（JSON 文件存储）
│   └── video_splitter.py     # 长视频分段、SRT 合并、上下文传递
├── utils/
│   ├── file_utils.py         # 临时文件管理、视频路径标准化
│   ├── formatters.py         # 时间格式化（SRT 时间、耗时显示）
│   ├── system_monitor.py     # CPU/内存/磁盘监控
│   └── logger.py             # 日志配置
├── tests/                    # 59 个测试用例
├── requirements.txt          # 依赖清单
└── CHANGELOG.md              # 改动记录与踩雷总结
```

---

## 二、核心技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| Python | 3.12 | 运行时 |
| Gradio | ≥4.0 (实际 6.x) | Web UI 框架 |
| PyTorch | ≥2.0 | 模型推理 |
| Transformers | ≥4.57 | Qwen3-VL 模型加载 |
| qwen-vl-utils | ≥0.0.14 | 视频处理工具 |
| torchcodec | ≥0.1.0 | 视频解码后端 |
| psutil | ≥5.9 | 系统监控 |
| ffmpeg | 系统安装 | 视频分割/转换 |

---

## 三、关键架构设计

### 3.1 两步推理流程（核心）

```
视频输入
  → Pass 1: model.generate() 生成视频分析文本
  → Pass 2: model.generate() 基于分析结果生成 SRT 字幕
```

**为什么不能合并为一步？**
- Qwen3-VL 需要先"理解"视频内容，再基于理解生成字幕
- 单步推理会导致输出垃圾内容（重复、格式混乱）
- 详见 `CHANGELOG.md` 雷 1

### 3.2 长视频分段机制

```python
# video_splitter.py 核心流程
if video_duration > 60s:
    segments = split_video(video_path)  # ffmpeg 切割
    for i, seg in enumerate(segments):
        analysis = analyze(seg, context=prev_context)  # 带上下文
        srt = generate_srt(analysis)
        prev_context = extract_context(analysis)  # 提取人物/场景
    merged_srt = merge_srt_segments(all_srt)  # 时间轴偏移合并
    cleanup_split_files()  # 清理临时文件
```

### 3.3 模型管理架构

```python
# model_manager.py 单例模式
model_manager = ModelManager()  # 全局实例

# 缓存机制
self._cache: dict[str, Tuple[model, processor]] = {}

# 内存管理
- 同时只保留一个模型
- 切换模型时自动释放旧模型
- MPS 内存不足自动降级到 CPU
```

---

## 四、文件详解

### 4.1 app.py（主入口，~950 行）

**职责**：UI 构建 + 业务编排

**关键函数**：
| 函数 | 作用 |
|------|------|
| `analyze_video()` | 主分析入口，处理短/长视频分支 |
| `_analyze_single_segment()` | 单段分析（两步推理） |
| `_extract_context()` | 从分析结果提取人物/场景用于下一段 |
| `build_demo()` | 构建 Gradio UI |
| `download_srt()` | 生成临时 SRT 文件供下载 |
| `load_model_action()` | 模型加载操作 |
| `download_model_action()` | 模型下载操作 |

**UI 组件**：
- 3 个 Tab：视频分析、历史记录、系统状态
- 系统状态 Tab 包含自动刷新 Timer（3 秒）

### 4.2 config.py（配置）

**模型映射**：
```python
MODELS = {
    "2B": ModelConfig("2B", "Qwen/Qwen3-VL-2B-Instruct", 1.0),
    "4B": ModelConfig("4B", "Qwen/Qwen3-VL-4B-Instruct", 1.5),
    "8B": ModelConfig("8B", "Qwen/Qwen3-VL-8B-Instruct", 2.0),
}
```

**视频参数**：
```python
VideoConfig:
    default_fps: 1.0
    short_video_fps: 2.0      # <10s 视频
    long_video_fps: 0.5       # >60s 视频
    base_pixels: 360 * 420    # 基础分辨率
```

**Prompt 模板**：
- `ANALYSIS_PROMPT`：视频分析提示词
- `SUBTITLE_PROMPT_TEMPLATE`：字幕生成模板（含 `{analysis_result}` 占位符）

### 4.3 model_manager.py

**MPS 降级逻辑**（重要）：
```python
if torch.backends.mps.is_available():
    try:
        model = model.to("mps")
    except Exception as e:
        logger.warning(f"MPS failed: {e}, keeping on CPU")
        # 不做任何操作，模型保持在 CPU
```

**模型下载**（使用 huggingface_hub）：
```python
from huggingface_hub import snapshot_download
snapshot_download(model_id, ignore_patterns=["*.md", "*.txt"])
```

**本地缓存路径**：
```
~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/
```

### 4.4 video_splitter.py

**分割方式**：ffmpeg libx264 重编码（非 stream copy，兼容性更好）

**SRT 合并**：自动偏移时间轴 + 重新编号

**上下文提取**：正则匹配人物/场景/时间线关键词

### 4.5 system_monitor.py

- 使用 `psutil` 获取 CPU/内存/磁盘
- 格式化为 Markdown 进度条
- 每 3 秒自动刷新（Gradio Timer）

---

## 五、环境变量

```bash
FORCE_QWENVL_VIDEO_READER=torchcodec  # 强制使用 torchcodec 解码
PYTORCH_ENABLE_MPS_FALLBACK=1         # MPS 兼容性
```

---

## 六、运行命令

```bash
# 安装依赖
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 启动服务
python app.py

# 运行测试
python -m pytest tests/ -v
```

默认端口：`7861`（config.py → ServerConfig）

---

## 七、踩坑大全（重点）

### 坑 1：单次推理生成字幕质量极差 ❌

**现象**：输出重复内容、格式混乱（`1 --> 01 --> 00:00:00,000`）

**原因**：跳过"理解"步骤直接生成字幕，模型不知道该写什么

**结论**：两步推理不能合并，速度优化不能牺牲质量

---

### 坑 2：repetition_penalty 过度导致内容贫乏 ❌

**尝试**：`repetition_penalty=1.2, no_repeat_ngram_size=3`

**现象**：字幕不重复了，但内容质量大幅下降

**结论**：不要随意添加生成参数，原始 `do_sample=False` 是最优解

---

### 坑 3：max_new_tokens 减半导致分析不完整 ❌

**尝试**：分析阶段从 1024 减到 512

**现象**：分析被截断，后续字幕质量下降

**结论**：token 数量影响输出质量，不能随意减少

---

### 坑 4：stream copy 分割不兼容 ❌

**尝试**：`-c copy` 代替 `-c:v libx264`

**现象**：部分视频格式分割失败

**结论**：stream copy 兼容性差，回退到重编码方案

---

### 坑 5：MPS 内存预加载崩溃 ❌

**尝试**：启动时预加载模型到 MPS

**现象**：MPS 分配 27.6GB 内存失败，应用崩溃

**结论**：MPS 预加载不可靠，改为仅 CUDA 环境预加载

---

### 坑 6：MPS→CPU 降级不完整 ❌

**错误写法**：
```python
except Exception as e:
    dtype = torch.float32  # 只改了 dtype，没移动模型
```

**正确写法**：
```python
except Exception as e:
    logger.warning(f"MPS failed: {e}, keeping on CPU")
    # 模型已在 CPU，无需操作
```

---

### 坑 7：Gradio 6.0 API 变更

**变化**：`css` 参数从 `Blocks()` 移到 `launch()`

**修复**：
```python
# 错误
gr.Blocks(css=CUSTOM_CSS)

# 正确
demo.launch(css=CUSTOM_CSS)
```

---

### 坑 8：Gradio Timer 自动刷新

**注意**：`gr.Timer(3)` 在切换 Tab 时会停止，这是 Gradio 的行为

**解决方案**：Timer 放在 Tab 内部，或使用 `every=` 参数

---

### 坑 9：临时文件泄漏

**问题**：`NamedTemporaryFile(delete=False)` 创建的文件不会自动清理

**解决**：`atexit` 注册清理函数 + 下载前清理旧文件

---

### 坑 10：上下文管理器重复调用

**问题**：手动调用 `__enter__` 后在 `finally` 中调用 `__exit__`，导致重复清理

**解决**：用 `with` 语句管理生命周期，不要手动调用 dunder 方法

---

## 八、扩展指南

### 8.1 添加新模型

1. 在 `config.py` 的 `MODELS` 字典中添加配置
2. 确保模型 ID 格式正确（`org/model-name`）

### 8.2 修改视频分割时长

修改 `video_splitter.py` 中的 `SEGMENT_DURATION` 常量（默认 60 秒）

### 8.3 添加新的 UI Tab

在 `build_demo()` 函数中，在 `gr.Tabs()` 内添加新的 `gr.TabItem()`

### 8.4 修改 Prompt

编辑 `config.py` 中的 `ANALYSIS_PROMPT` 或 `SUBTITLE_PROMPT_TEMPLATE`

---

## 九、测试覆盖

当前 59 个测试用例，覆盖：
- 历史管理器（8 个）
- 字幕生成器（14 个）
- 工具函数（15 个）
- 视频分割器（8 个）
- 其他（14 个）

运行：`python -m pytest tests/ -v`

---

## 十、已知限制

1. **MPS 兼容性**：macOS 上大模型可能内存不足，自动降级到 CPU
2. **视频格式**：依赖 ffmpeg，需要系统安装
3. **网络依赖**：首次下载模型需要网络连接
4. **并发限制**：同时只支持一个分析任务（模型独占）
5. **最长视频**：无硬性限制，但超长视频分段后分析时间很长
