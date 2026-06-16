# Qwen3-VL 视频理解演示

一个最简单的视频理解演示项目：上传视频，选择 Qwen3-VL 2B / 4B / 8B 模型，点击按钮后生成标准 SRT 旁白字幕。

页面只包含：

- 视频上传框
- 模型选择
- 风格学习文案输入框
- 开始分析按钮
- 识别耗时区域
- 输出结果区域

## 功能

输出结果包含：

- AI 生成的标准 SRT 旁白字幕
- 从开始识别到完成输出的耗时
- 如果输入风格样本文案，字幕文案会学习参考文案的表达风格
- 如果视频格式无法直接解码，会自动转为兼容的 H.264 MP4 后重试

## 环境要求

- Python 3.10 - 3.12
- 建议使用 NVIDIA GPU。CPU 也能启动项目，但模型推理会非常慢。
- 第一次运行会从 Hugging Face 下载模型权重，需要网络和足够磁盘空间。

模型越大显存要求越高。建议先用 `2B` 跑通流程。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果系统默认 Python 太新，例如 Python 3.14，建议改用 Python 3.10、3.11 或 3.12 创建虚拟环境。

## 一键启动

```bash
python app.py
```

如果你的系统只有 `python3` 命令，也可以使用：

```bash
python3 app.py
```

启动后打开：

```text
http://localhost:7860
```

## 使用方式

1. 上传一个视频文件。
2. 选择模型：`2B`、`4B` 或 `8B`。
3. 可选：粘贴一篇或多篇参考视频文案到“风格学习文案”。
4. 点击“开始分析”。
5. 等待输出完整 SRT 字幕和识别耗时。

## 模型对应关系

| 选择 | Hugging Face 模型 |
| --- | --- |
| 2B | `Qwen/Qwen3-VL-2B-Instruct` |
| 4B | `Qwen/Qwen3-VL-4B-Instruct` |
| 8B | `Qwen/Qwen3-VL-8B-Instruct` |

## 说明

项目使用 Qwen3-VL 官方推荐的视频输入方式：

- 使用 `qwen_vl_utils.process_vision_info`
- 对 Qwen3-VL 设置 `image_patch_size=16`
- 设置 `return_video_kwargs=True`
- 设置 `return_video_metadata=True`
- 将 `video_metadata` 传给 processor
- 因为 `qwen-vl-utils` 已处理视频尺寸，所以 processor 设置 `do_resize=False`
- 使用 `torchcodec` 作为视频解码后端，避免新版 `torchvision` 缺少 `read_video` 的问题
- 对解码失败的视频使用 `ffmpeg` 自动转换为 H.264 / AAC / yuv420p MP4 后重试

输出流程为：

```text
上传视频
-> Qwen3-VL 理解视频
-> Qwen 根据理解结果和可选风格样本生成标准 SRT 旁白字幕
-> 只返回完整 SRT 字幕
```

为了优先跑通，本项目没有登录、数据库、历史记录和任务队列。
