"""
VL 多模态客户端：封装 qwen-vl-plus 进行图像、视频、混合输入分析。

功能：
  - 图像分析：将图片文件编码为 base64，调用 qwen-vl-plus 进行分析
  - 视频分析：通过 ffmpeg 提取关键帧，对帧序列进行多模态分析
  - 混合输入：文本 + 图片组合分析

用法：
    from src.tools.vl_media.vl_client import VLClient

    client = VLClient()
    # 分析图片
    result = client.analyze_image("path/to/image.jpg", "请描述这张图片")
    # 分析视频
    result = client.analyze_video("path/to/video.mp4", "视频中发生了什么？")
    # 混合分析
    result = client.analyze_mixed(
        texts=["用户描述：这是一张产品图"],
        image_paths=["path/to/product.jpg"],
        prompt="请结合用户描述分析这张图片",
    )
"""

import os
import base64
import json
import logging
import subprocess
import tempfile
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================
# 图像编码工具
# ============================================================

def _image_to_base64(image_path: str) -> str:
    """将图片文件编码为 base64 字符串，带 MIME 前缀。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def _get_image_size_mb(image_path: str) -> float:
    """获取图片文件大小（MB）。"""
    return os.path.getsize(image_path) / (1024 * 1024)


# ============================================================
# 视频帧提取
# ============================================================

class VideoFrameExtractor:
    """视频帧提取器：使用 ffmpeg 或 OpenCV 从视频中提取关键帧。"""

    # 最大帧数（避免超过 VL 模型的输入限制）
    MAX_FRAMES = 10
    # 单帧最大尺寸（像素，宽高中较大者）
    MAX_FRAME_SIZE = 1024

    def __init__(self, max_frames: int = 10, max_frame_size: int = 1024):
        self.max_frames = max_frames
        self.max_frame_size = max_frame_size

    def extract_frames(self, video_path: str) -> List[str]:
        """
        从视频中提取关键帧，返回临时图片文件路径列表。
        
        策略：
          - 短视频（≤30秒）：均匀采样，每秒 1 帧，最多 max_frames 帧
          - 长视频：均匀采样，总共 max_frames 帧
          - 优先使用 ffmpeg，失败则回退到 OpenCV
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        duration = self._get_duration(video_path)
        frame_count = min(self.max_frames, max(1, int(duration)))

        return self._extract_ffmpeg(video_path, frame_count)

    def _get_duration(self, video_path: str) -> float:
        """获取视频时长（秒）。"""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning("ffprobe 获取时长失败: %s", e)
        # 回退：使用 OpenCV
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0 and total_frames > 0:
                return total_frames / fps
        except Exception:
            pass
        return 30.0  # 默认 30 秒

    def _extract_ffmpeg(self, video_path: str, frame_count: int) -> List[str]:
        """使用 ffmpeg 提取帧。"""
        import cv2
        frame_paths = []
        temp_dir = tempfile.mkdtemp()

        try:
            # 先用 ffmpeg 提取帧
            output_pattern = os.path.join(temp_dir, "frame_%04d.jpg")
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", f"fps={frame_count}/30,scale='min({self.max_frame_size},iw)':min'({self.max_frame_size},ih)':force_original_aspect_ratio=decrease",
                "-q:v", "2",
                "-frames:v", str(frame_count),
                "-y", output_pattern,
            ]
            subprocess.run(cmd, capture_output=True, timeout=120)

            # 收集生成的帧
            for i in range(1, frame_count + 1):
                fp = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                if os.path.exists(fp):
                    frame_paths.append(fp)

            # 如果 ffmpeg 没有生成帧，回退到 OpenCV
            if not frame_paths:
                return self._extract_opencv(video_path, frame_count)

            return frame_paths

        except Exception as e:
            logger.warning("ffmpeg 提取帧失败: %s，回退到 OpenCV", e)
            return self._extract_opencv(video_path, frame_count)

    def _extract_opencv(self, video_path: str, frame_count: int) -> List[str]:
        """使用 OpenCV 提取帧（回退方案）。"""
        import cv2
        frame_paths = []
        temp_dir = tempfile.mkdtemp()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return frame_paths

        # 均匀采样
        step = max(1, total_frames // frame_count)
        frame_idx = 0

        for i in range(0, total_frames, step):
            if frame_idx >= frame_count:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # 缩放
                h, w = frame.shape[:2]
                scale = min(self.max_frame_size / max(w, h), 1.0)
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h))
                fp = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(fp, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_paths.append(fp)
                frame_idx += 1

        cap.release()
        return frame_paths

    def cleanup(self, frame_paths: List[str]):
        """清理临时帧文件。"""
        dirs = set()
        for fp in frame_paths:
            try:
                os.remove(fp)
                dirs.add(os.path.dirname(fp))
            except Exception:
                pass
        for d in dirs:
            try:
                os.rmdir(d)
            except Exception:
                pass


# ============================================================
# VL 多模态客户端
# ============================================================

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_VL_MODEL = "qwen-vl-plus"


class VLClient:
    """VL 多模态客户端，封装 qwen-vl-plus 进行图像/视频/混合输入分析。

    纯文本分析自动回退到本地 practiceLLM，不经过 VL 模型。
    """

    # VL 分析默认提示模板
    IMAGE_ANALYSIS_PROMPT = "请详细描述这张图片的内容，包括：主体、场景、颜色、文字、布局等可视化元素。"
    VIDEO_ANALYSIS_PROMPT = "这是从视频中提取的连续帧序列。请分析视频中发生了什么，描述场景、动作、人物和事件发展。"
    MIXED_ANALYSIS_PROMPT = "请结合提供的文本描述和图片，给出综合分析。"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        """初始化 VL 客户端。

        Args:
            model: VL 模型名，默认 qwen-vl-plus
            api_key: DashScope API Key，默认从 DASHSCOPE_API_KEY 环境变量读取
            base_url: API 端点，默认 DashScope OpenAI 兼容模式
            temperature: 生成温度
            max_tokens: 最大生成 Token 数
        """
        self.model = model or os.getenv("VL_MODEL_ID", DEFAULT_VL_MODEL)
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        base_url = base_url or os.getenv("VL_BASE_URL", DASHSCOPE_BASE_URL)

        if not api_key:
            raise ValueError(
                "需要 DASHSCOPE_API_KEY 环境变量来调用 VL 模型。"
                "请在 .env 文件中配置 DASHSCOPE_API_KEY。"
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._frame_extractor = VideoFrameExtractor()

    # ============================================================
    # 底层 API 调用
    # ============================================================

    def _call_vl(
        self,
        content: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """调用 VL 模型。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self._temperature,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error("VL API 调用失败: %s", e)
            return f"分析失败: {e}"

    # ============================================================
    # 图片分析
    # ============================================================

    def analyze_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """分析单张图片。

        Args:
            image_path: 图片文件路径
            prompt: 分析提示（可选，默认使用通用描述提示）
            system_prompt: 系统提示词
        Returns:
            分析结果文本
        """
        prompt = prompt or self.IMAGE_ANALYSIS_PROMPT

        # 检查文件大小（VL API 通常有 10MB 限制）
        size_mb = _get_image_size_mb(image_path)
        if size_mb > 10:
            return f"图片文件过大（{size_mb:.1f}MB），超过 API 限制（10MB）。请压缩后重试。"

        image_b64 = _image_to_base64(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]
        return self._call_vl(content, system_prompt=system_prompt)

    def analyze_images(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """分析多张图片。

        Args:
            image_paths: 图片文件路径列表
            prompt: 分析提示
            system_prompt: 系统提示词
        Returns:
            分析结果文本
        """
        if not image_paths:
            return "没有提供图片文件。"

        prompt = prompt or self.IMAGE_ANALYSIS_PROMPT
        content = [{"type": "text", "text": prompt}]

        for path in image_paths:
            size_mb = _get_image_size_mb(path)
            if size_mb > 10:
                return f"图片文件 {path} 过大（{size_mb:.1f}MB），超过 API 限制（10MB）。"
            image_b64 = _image_to_base64(path)
            content.append({"type": "image_url", "image_url": {"url": image_b64}})

        return self._call_vl(content, system_prompt=system_prompt)

    # ============================================================
    # 视频分析
    # ============================================================

    def analyze_video(
        self,
        video_path: str,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_frames: int = 10,
    ) -> str:
        """分析视频内容。

        流程：
          1. 提取视频关键帧（均匀采样）
          2. 将所有关键帧作为图片序列发送给 VL 模型
          3. 返回分析结果

        Args:
            video_path: 视频文件路径
            prompt: 分析提示
            system_prompt: 系统提示词
            max_frames: 最大提取帧数
        Returns:
            分析结果文本
        """
        prompt = prompt or self.VIDEO_ANALYSIS_PROMPT

        # 提取帧
        self._frame_extractor.max_frames = max_frames
        frame_paths = self._frame_extractor.extract_frames(video_path)
        if not frame_paths:
            return "无法从视频中提取帧，请检查视频文件格式。"

        # 构建多模态内容
        content = [{"type": "text", "text": prompt}]
        for fp in frame_paths:
            try:
                image_b64 = _image_to_base64(fp)
                content.append({"type": "image_url", "image_url": {"url": image_b64}})
            except Exception as e:
                logger.warning("跳过帧 %s: %s", fp, e)

        # 调用 VL 模型
        result = self._call_vl(content, system_prompt=system_prompt)

        # 清理临时帧
        self._frame_extractor.cleanup(frame_paths)

        return result

    # ============================================================
    # 混合输入分析
    # ============================================================

    def analyze_mixed(
        self,
        texts: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """分析混合输入（文本 + 图片）。

        Args:
            texts: 文本内容列表（作为上下文提供给 VL 模型）
            image_paths: 图片文件路径列表
            prompt: 综合分析提示
            system_prompt: 系统提示词
        Returns:
            分析结果
        """
        texts = texts or []
        image_paths = image_paths or []
        prompt = prompt or self.MIXED_ANALYSIS_PROMPT

        # 纯文本 → 不需要 VL 模型
        if not image_paths:
            return "没有提供图片，混合分析需要至少一张图片。"

        content = [{"type": "text", "text": prompt}]

        # 添加上下文文本
        for text in texts:
            if text.strip():
                content.append({"type": "text", "text": text.strip()})

        # 添加图片
        for path in image_paths:
            size_mb = _get_image_size_mb(path)
            if size_mb > 10:
                self._frame_extractor.cleanup([])
                return f"图片文件 {path} 过大（{size_mb:.1f}MB），超过 API 限制（10MB）。"
            image_b64 = _image_to_base64(path)
            content.append({"type": "image_url", "image_url": {"url": image_b64}})

        return self._call_vl(content, system_prompt=system_prompt)