"""Qwen3-VL Video Understanding Demo - Main Application."""

import os
import time
import tempfile
import atexit
from pathlib import Path
from typing import Any

import gradio as gr
import torch

# Set environment variables before importing other modules
os.environ["FORCE_QWENVL_VIDEO_READER"] = "torchcodec"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from config import MODELS, get_config
from core.model_manager import model_manager
from core.video_processor import video_processor
from core.subtitle_generator import subtitle_generator
from core.history_manager import history_manager
from core.video_splitter import (
    get_video_duration,
    split_video,
    merge_srt_segments,
    cleanup_split_files,
    SEGMENT_DURATION,
)
from utils.file_utils import normalize_video_path
from utils.formatters import format_elapsed, format_duration
from utils.system_monitor import get_system_status_display, get_system_status, format_bytes
from utils.logger import get_logger

logger = get_logger(__name__)

# Custom CSS - Minimalist Modern Style with Dark Mode Support
CUSTOM_CSS = """
/* ===== Hide Gradio branding ===== */
.gradio-container .footer {
    display: none !important;
}
.gradio-container .api-docs {
    display: none !important;
}
.gradio-container [data-testid="api-link"] {
    display: none !important;
}
.gradio-container .gr-button-link {
    display: none !important;
}
footer {
    display: none !important;
}

/* ===== Global Reset ===== */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
}

/* ===== Header ===== */
.app-header {
    text-align: center;
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--border-color-primary, #e5e7eb);
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 1.8em !important;
    font-weight: 700 !important;
    margin: 0 0 6px !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-header p {
    color: var(--body-text-color-subdued, #6b7280) !important;
    font-size: 0.95em !important;
    margin: 0 !important;
}

/* ===== Tabs ===== */
.tabs > .tab-nav {
    border-bottom: 2px solid var(--border-color-primary, #e5e7eb) !important;
    gap: 0 !important;
}
.tabs > .tab-nav > button {
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: all 0.2s !important;
}
.tabs > .tab-nav > button.selected {
    border-bottom-color: #6366f1 !important;
    color: #6366f1 !important;
    font-weight: 600 !important;
}

/* ===== Cards / Sections ===== */
.section-card {
    background: var(--background-fill-primary, #ffffff);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.85em !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: var(--body-text-color-subdued, #6b7280) !important;
    margin-bottom: 12px !important;
}

/* ===== Status Box ===== */
.status-box {
    padding: 8px 14px !important;
    border-radius: 8px !important;
    background: var(--background-fill-secondary, #f9fafb) !important;
    font-size: 0.9em !important;
    border: 1px solid var(--border-color-primary, #e5e7eb) !important;
}

/* ===== Model Status ===== */
.model-status {
    text-align: center;
    padding: 8px 16px;
    font-size: 0.85em;
    color: var(--body-text-color-subdued, #6b7280);
    margin-bottom: 16px;
}

/* ===== Buttons ===== */
.analyze-btn {
    width: 100% !important;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
}
.analyze-btn:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
.secondary-btn {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ===== History ===== */
.history-item {
    padding: 12px 16px;
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 8px;
    margin-bottom: 8px;
    background: var(--background-fill-secondary, #f9fafb);
}
.history-item:hover {
    border-color: #6366f1;
}

/* ===== Dark Mode ===== */
@media (prefers-color-scheme: dark) {
    .app-header {
        border-bottom-color: #374151 !important;
    }
    .app-header p {
        color: #9ca3af !important;
    }
    .status-box {
        background: #1f2937 !important;
        border-color: #374151 !important;
    }
    .tabs > .tab-nav {
        border-bottom-color: #374151 !important;
    }
    .history-item {
        background: #1f2937 !important;
        border-color: #374151 !important;
    }
}

/* Also support Gradio's dark class */
.dark .app-header {
    border-bottom-color: #374151 !important;
}
.dark .app-header p {
    color: #9ca3af !important;
}
.dark .status-box {
    background: #1f2937 !important;
    border-color: #374151 !important;
}
.dark .tabs > .tab-nav {
    border-bottom-color: #374151 !important;
}
.dark .history-item {
    background: #1f2937 !important;
    border-color: #374151 !important;
}
"""

# Track temp SRT files for cleanup
_srt_temp_files: list[str] = []


def _cleanup_srt_temp_files():
    """Remove leftover temp SRT files."""
    for path in _srt_temp_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
    _srt_temp_files.clear()


atexit.register(_cleanup_srt_temp_files)
atexit.register(cleanup_split_files)


def get_model_status() -> str:
    """Get formatted model status for display."""
    info = model_manager.get_model_info()
    loaded = info.get("loaded_models", [])
    
    if not loaded:
        return "当前无已加载模型（首次分析时自动加载）"
    
    parts = []
    for model_id in loaded:
        short_name = model_id.split("/")[-1].replace("Qwen3-VL-", "").replace("-Instruct", "")
        parts.append(f"`{short_name}`")
    
    return f"已加载模型: {', '.join(parts)}"


def get_model_management_display() -> str:
    """Get formatted model management info."""
    info = model_manager.get_model_info()
    lines = []
    
    for key in ["2B", "4B", "8B"]:
        model_info = info["local_models"].get(key, {})
        model_id = model_info.get("model_id", "")
        downloaded = model_info.get("downloaded", False)
        short_name = model_id.split("/")[-1].replace("Qwen3-VL-", "").replace("-Instruct", "")
        
        # Check if loaded in memory
        loaded = model_id in info.get("loaded_models", [])
        
        # Get disk size
        disk_size = model_manager.get_model_disk_size(key)
        
        if loaded:
            status = "🟢 已加载"
        elif downloaded:
            status = "📥 已下载"
        else:
            status = "⚪ 未下载"
        
        lines.append(
            f"**{key}** (`{short_name}`)\n"
            f"{status} | {disk_size}"
        )
    
    return "\n\n".join(lines)


def load_model_action(model_key: str) -> tuple[str, str]:
    """Load a model into memory."""
    if model_key not in MODELS:
        return "请选择一个有效的模型", get_model_management_display()
    
    model_id = MODELS[model_key].model_id
    try:
        model_manager.load_model(model_id)
        return f"模型 {model_key} 已加载", get_model_management_display()
    except Exception as e:
        return f"加载失败: {e}", get_model_management_display()


def unload_model_action(model_key: str) -> tuple[str, str]:
    """Unload a model from memory."""
    if model_key not in MODELS:
        return "请选择一个有效的模型", get_model_management_display()
    
    model_id = MODELS[model_key].model_id
    result = model_manager.unload_model(model_id)
    return result, get_model_management_display()


def delete_model_action(model_key: str) -> tuple[str, str]:
    """Delete model from local cache."""
    if model_key not in MODELS:
        return "请选择一个有效的模型", get_model_management_display()
    
    result = model_manager.delete_local_model(model_key)
    return result, get_model_management_display()


def download_model_action(model_key: str) -> tuple[str, str]:
    """Download model from HuggingFace."""
    if model_key not in MODELS:
        return "请选择一个有效的模型", get_model_management_display()
    
    model_id = MODELS[model_key].model_id
    
    # Check if already downloaded
    local_path = model_manager._get_local_model_path(model_id)
    if local_path:
        return f"模型 {model_key} 已下载", get_model_management_display()
    
    try:
        # Use snapshot_download for reliable download
        from huggingface_hub import snapshot_download
        logger.info(f"开始下载模型 {model_key}: {model_id}")
        snapshot_download(
            model_id,
            ignore_patterns=["*.md", "*.txt", "*.gitignore"],
        )
        logger.info(f"模型 {model_key} 下载完成")
        return f"模型 {model_key} 下载完成", get_model_management_display()
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return f"下载失败: {e}", get_model_management_display()


def open_model_dir_action(model_key: str) -> str:
    """Open model directory in file manager."""
    if model_key not in MODELS:
        return "请选择一个有效的模型"
    
    model_id = MODELS[model_key].model_id
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name
    
    if not model_cache_path.exists():
        return f"模型 {model_key} 本地缓存不存在"
    
    import subprocess
    subprocess.run(["open", str(model_cache_path)])
    return f"已打开: {model_cache_path}"


def _analyze_single_segment(
    video_path: str,
    model: Any,
    processor: Any,
    model_size: str,
    duration_seconds: float | None,
    previous_context: str | None,
) -> tuple[str, str]:
    """Analyze a single video segment with two-pass inference.
    
    First pass: analyze video content.
    Second pass: generate SRT from analysis (more reliable than single-pass).
    
    Args:
        video_path: Path to video segment
        model: Loaded model
        processor: Model processor
        model_size: Model size
        duration_seconds: Segment duration
        previous_context: Analysis result from previous segment (for consistency)
        
    Returns:
        Tuple of (analysis_result, srt_subtitle)
    """
    messages, images, videos, video_kwargs, video_metadatas, seg_duration = (
        video_processor.process_video_messages(video_path, model_size)
    )
    
    if seg_duration is None:
        seg_duration = duration_seconds
    
    # Build prompt with context from previous segment
    user_text = "请分析这个视频"
    if previous_context:
        user_text = (
            f"请分析这个视频。注意：这是连续视频的一部分。"
            f"之前的片段中已出现的人物、场景信息如下，请保持一致：\n"
            f"{previous_context}"
        )
    
    messages[0]["content"][1]["text"] = user_text
    
    # Pass 1: Analyze video
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs,
    )
    inputs = inputs.to(model.device)
    
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )
    
    analysis_result = model_manager.decode_generation(processor, inputs, generated_ids)
    
    # Pass 2: Generate SRT from analysis
    subtitle_result = subtitle_generator.generate_subtitle(
        model=model,
        processor=processor,
        analysis_result=analysis_result,
        duration_seconds=seg_duration,
        style_text=None,
    )
    
    return analysis_result, subtitle_result


def analyze_video(
    video: Any,
    model_size: str,
    style_text: str | None,
    progress: gr.Progress = gr.Progress()
) -> tuple[str, str, str, gr.update, str]:
    """Analyze video and generate SRT subtitles.
    
    For long videos (>60s), automatically splits into segments, analyzes each
    with context from previous segments for consistency, then merges into
    a unified SRT output.
    
    Args:
        video: Video input (file path or Gradio video object)
        model_size: Model size (2B, 4B, 8B)
        style_text: Optional style reference text
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (SRT subtitle text, elapsed time string, history display, button state, model status)
    """
    started_at = time.perf_counter()
    
    video_path = normalize_video_path(video)
    if not video_path:
        return "请先上传一个视频。", format_elapsed(time.perf_counter() - started_at), history_manager.get_history_display(), gr.update(interactive=True), get_model_status()
    
    model_config = MODELS[model_size]
    model_id = model_config.model_id
    
    try:
        progress(0.05, desc="正在检测视频时长...")
        logger.info(f"开始分析视频: {video_path}, 模型: {model_size}")
        
        # Ensure video backend is initialized
        video_processor.ensure_torchcodec_backend()
        
        # Check video duration
        total_duration = get_video_duration(video_path)
        
        # Load model
        progress(0.1, desc="正在加载模型...")
        logger.info(f"加载模型: {model_id}")
        model, processor = model_manager.load_model(model_id)
        
        # Decide: split or direct analysis
        if total_duration and total_duration > SEGMENT_DURATION:
            # Long video: split and analyze segments
            logger.info(f"视频时长 {total_duration:.1f}s > {SEGMENT_DURATION}s，进行分段分析")
            progress(0.15, desc=f"视频较长({total_duration:.0f}秒)，正在分割...")
            
            segment_paths, total_dur = split_video(video_path, SEGMENT_DURATION)
            num_segments = len(segment_paths)
            logger.info(f"视频已分为 {num_segments} 段")
            
            srt_segments = []
            segment_durations = []
            previous_context = None
            
            for i, seg_path in enumerate(segment_paths):
                seg_progress = 0.15 + (0.75 * i / num_segments)
                progress(seg_progress, desc=f"正在分析第 {i+1}/{num_segments} 段...")
                logger.info(f"分析第 {i+1}/{num_segments} 段: {seg_path}")
                
                # Get segment duration
                seg_dur = get_video_duration(seg_path)
                if seg_dur is None:
                    seg_dur = SEGMENT_DURATION
                segment_durations.append(seg_dur)
                
                # Analyze segment with context
                analysis_result, subtitle_result = _analyze_single_segment(
                    video_path=seg_path,
                    model=model,
                    processor=processor,
                    model_size=model_size,
                    duration_seconds=seg_dur,
                    previous_context=previous_context,
                )
                
                srt_segments.append(subtitle_result)
                
                # Extract key context for next segment (character names, scenes)
                previous_context = _extract_context(analysis_result)
                
                logger.info(f"第 {i+1} 段分析完成")
            
            progress(0.92, desc="正在合并字幕...")
            logger.info("合并所有段落字幕...")
            
            # Merge SRT segments
            subtitle_result = merge_srt_segments(srt_segments, segment_durations)
            
            # Clean up split files
            cleanup_split_files()
            logger.info("临时分段文件已清理")
            
        else:
            # Short video: direct analysis
            progress(0.3, desc="正在处理视频...")
            logger.info("处理视频中...")
            
            messages, images, videos, video_kwargs, video_metadatas, duration_seconds = (
                video_processor.process_video_messages(video_path, model_size)
            )
            
            # Pass 1: Analyze video
            progress(0.5, desc="正在生成分析...")
            logger.info("生成视频分析...")
            
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            inputs = processor(
                text=text,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            )
            inputs = inputs.to(model.device)
            
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            
            analysis_result = model_manager.decode_generation(processor, inputs, generated_ids)
            
            # Pass 2: Generate SRT from analysis
            progress(0.75, desc="正在生成字幕...")
            logger.info("生成字幕中...")
            
            subtitle_result = subtitle_generator.generate_subtitle(
                model=model,
                processor=processor,
                analysis_result=analysis_result,
                duration_seconds=duration_seconds,
                style_text=style_text,
            )
        
        elapsed = time.perf_counter() - started_at
        
        # Save to history
        video_name = os.path.basename(video_path)
        history_manager.add_record(
            video_name=video_name,
            model_size=model_size,
            srt_content=subtitle_result,
            elapsed_time=elapsed,
            duration_seconds=total_duration,
        )
        
        progress(1.0, desc="完成！")
        logger.info(f"分析完成，耗时: {elapsed:.2f}秒")
        
        history_display = history_manager.get_history_display()
        
        return subtitle_result, format_elapsed(elapsed), history_display, gr.update(interactive=True), get_model_status()
        
    except Exception as exc:
        elapsed = time.perf_counter() - started_at
        error_msg = f"分析失败：{exc}"
        logger.error(error_msg)
        cleanup_split_files()
        return error_msg, format_elapsed(elapsed), history_manager.get_history_display(), gr.update(interactive=True), get_model_status()


def _extract_context(analysis_result: str) -> str:
    """Extract key context from analysis for next segment.
    
    Extracts character names, scene descriptions, and key elements
    to maintain consistency across segments.
    
    Args:
        analysis_result: Analysis text from current segment
        
    Returns:
        Context string for next segment
    """
    context_parts = []
    
    # Extract character names (look for patterns like "人物：", "角色：", etc.)
    import re
    name_patterns = [
        r'(?:人物|角色|主角|人物名称?)[：:]\s*(.+?)(?:\n|$)',
        r'(?:出现了?|有)\s*(.+?)(?:等人物|等人|，|。)',
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, analysis_result)
        for match in matches:
            names = match.strip()
            if names and len(names) < 100:
                context_parts.append(f"人物/角色: {names}")
    
    # Extract scene descriptions
    scene_patterns = [
        r'(?:场景|环境|地点)[：:]\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in scene_patterns:
        matches = re.findall(pattern, analysis_result)
        for match in matches:
            scene = match.strip()
            if scene and len(scene) < 200:
                context_parts.append(f"场景: {scene}")
    
    # Extract timeline key points
    timeline_patterns = [
        r'(?:时间线|时间顺序|关键变化)[：:]\s*(.+?)(?:\n\n|\Z)',
    ]
    
    for pattern in timeline_patterns:
        matches = re.findall(pattern, analysis_result, re.DOTALL)
        for match in matches:
            timeline = match.strip()[:300]
            if timeline:
                context_parts.append(f"时间线要点: {timeline}")
    
    if not context_parts:
        # Fallback: use first 300 chars of analysis as context
        return analysis_result[:300] if analysis_result else ""
    
    return "\n".join(context_parts)


def download_srt(subtitle_text: str) -> str:
    """Create a temporary SRT file for download.
    
    Args:
        subtitle_text: SRT content
        
    Returns:
        Path to temporary SRT file
    """
    if not subtitle_text or subtitle_text.startswith("请先上传") or subtitle_text.startswith("分析失败"):
        return None
    
    # Clean up previous temp files
    _cleanup_srt_temp_files()
    
    # Create temporary file
    tmp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.srt',
        delete=False,
        encoding='utf-8'
    )
    tmp_file.write(subtitle_text)
    tmp_file.close()
    
    _srt_temp_files.append(tmp_file.name)
    logger.info(f"SRT 文件已创建: {tmp_file.name}")
    return tmp_file.name


def build_demo() -> gr.Blocks:
    """Build Gradio demo interface with minimalist modern design.
    
    Returns:
        Gradio Blocks interface
    """
    config = get_config()
    
    with gr.Blocks(
        title="Qwen3-VL 视频理解",
    ) as demo:
        # Header
        gr.HTML("""
        <div class="app-header">
            <h1>Qwen3-VL 视频理解</h1>
            <p>上传视频，AI 自动生成 SRT 字幕</p>
        </div>
        """)
        
        # Model status
        model_status_display = gr.Markdown(
            value=get_model_status(),
            elem_classes="model-status",
        )
        
        with gr.Tabs():
            # Tab 1: Main Analysis
            with gr.TabItem("视频分析", id="analysis"):
                with gr.Row():
                    # Left column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("**输入**", elem_classes="section-title")
                        
                        video_input = gr.Video(
                            label="上传视频",
                            height=240,
                        )
                        
                        with gr.Row():
                            clear_video_btn = gr.Button(
                                "清除",
                                variant="secondary",
                                size="sm",
                                elem_classes="secondary-btn",
                            )
                        
                        model_select = gr.Radio(
                            choices=list(MODELS.keys()),
                            value=config.default_model,
                            label="模型",
                        )
                        
                        style_input = gr.Textbox(
                            label="风格文案（可选）",
                            lines=2,
                            placeholder="粘贴参考文案，AI 将学习其表达风格...",
                        )
                        
                        analyze_button = gr.Button(
                            "开始分析",
                            variant="primary",
                            size="lg",
                            elem_classes="analyze-btn",
                        )
                    
                    # Right column - Output
                    with gr.Column(scale=1):
                        gr.Markdown("**结果**", elem_classes="section-title")
                        
                        elapsed_output = gr.Textbox(
                            label="耗时",
                            lines=1,
                            interactive=False,
                            elem_classes="status-box",
                        )
                        
                        output = gr.Textbox(
                            label="SRT 字幕",
                            lines=14,
                            interactive=False,
                        )
                        
                        with gr.Row():
                            download_btn = gr.Button(
                                "下载 SRT",
                                variant="secondary",
                                elem_classes="secondary-btn",
                            )
                            srt_file = gr.File(
                                visible=False,
                            )
            
            # Tab 2: History
            with gr.TabItem("历史记录", id="history"):
                with gr.Row():
                    with gr.Column(scale=4):
                        history_display = gr.Markdown(
                            value="暂无记录。",
                        )
                    with gr.Column(scale=1, min_width=100):
                        clear_history_btn = gr.Button(
                            "清空",
                            variant="secondary",
                            size="sm",
                            elem_classes="secondary-btn",
                        )
            
            # Tab 3: System Status
            with gr.TabItem("系统状态", id="system"):
                # Auto-refresh timer (every 3 seconds)
                refresh_timer = gr.Timer(3)
                
                with gr.Row():
                    # System info column
                    with gr.Column(scale=3):
                        gr.Markdown("**系统资源**", elem_classes="section-title")
                        system_status_display = gr.Markdown(
                            value=get_system_status_display(),
                        )
                    
                    # Model management column
                    with gr.Column(scale=2):
                        gr.Markdown("**模型管理**", elem_classes="section-title")
                        model_management_display = gr.Markdown(
                            value=get_model_management_display(),
                        )
                        
                        model_action_select = gr.Radio(
                            choices=["2B", "4B", "8B"],
                            label="选择模型",
                        )
                        
                        with gr.Row():
                            download_model_btn = gr.Button(
                                "下载",
                                variant="primary",
                                size="sm",
                            )
                            load_model_btn = gr.Button(
                                "加载",
                                variant="secondary",
                                size="sm",
                            )
                            unload_model_btn = gr.Button(
                                "卸载",
                                variant="secondary",
                                size="sm",
                            )
                        
                        with gr.Row():
                            delete_model_btn = gr.Button(
                                "删除缓存",
                                variant="stop",
                                size="sm",
                            )
                            open_dir_btn = gr.Button(
                                "打开目录",
                                variant="secondary",
                                size="sm",
                            )
                        
                        model_action_output = gr.Textbox(
                            label="操作结果",
                            lines=1,
                            interactive=False,
                            elem_classes="status-box",
                        )
        
        # Event handlers
        analyze_button.click(
            fn=analyze_video,
            inputs=[video_input, model_select, style_input],
            outputs=[output, elapsed_output, history_display, analyze_button, model_status_display],
        )
        
        download_btn.click(
            fn=download_srt,
            inputs=[output],
            outputs=[srt_file],
        )
        
        clear_history_btn.click(
            fn=lambda: (history_manager.clear_history(), history_manager.get_history_display()),
            outputs=[history_display],
        )
        
        clear_video_btn.click(
            fn=lambda: None,
            outputs=[video_input],
        )
        
        # System status auto-refresh
        refresh_timer.tick(
            fn=get_system_status_display,
            outputs=[system_status_display],
        )
        
        download_model_btn.click(
            fn=download_model_action,
            inputs=[model_action_select],
            outputs=[model_action_output, model_management_display],
        )
        
        load_model_btn.click(
            fn=load_model_action,
            inputs=[model_action_select],
            outputs=[model_action_output, model_management_display],
        )
        
        unload_model_btn.click(
            fn=unload_model_action,
            inputs=[model_action_select],
            outputs=[model_action_output, model_management_display],
        )
        
        delete_model_btn.click(
            fn=delete_model_action,
            inputs=[model_action_select],
            outputs=[model_action_output, model_management_display],
        )
        
        open_dir_btn.click(
            fn=open_model_dir_action,
            inputs=[model_action_select],
            outputs=[model_action_output],
        )
    
    return demo


if __name__ == "__main__":
    config = get_config()
    logger.info("启动 Qwen3-VL 视频理解演示应用")
    
    # Display model status
    model_info = model_manager.get_model_info()
    logger.info("模型状态:")
    for model_key, model_status in model_info['local_models'].items():
        status = "已下载" if model_status['downloaded'] else "未下载"
        logger.info(f"  {model_key}: {status}")
    
    # Preload default model for faster first analysis (only if model is small enough)
    # Skipped on MPS to avoid memory issues with large models
    if not torch.backends.mps.is_available():
        default_model_id = MODELS[config.default_model].model_id
        if model_info['local_models'].get(config.default_model, {}).get('downloaded'):
            logger.info(f"预加载默认模型: {config.default_model}")
            model_manager.preload_model(default_model_id)
    
    demo = build_demo()
    demo.launch(
        server_name=config.server.host,
        server_port=config.server.port,
        prevent_thread_lock=False,
        css=CUSTOM_CSS,
    )
