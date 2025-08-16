# AI 图像与视频神器

炫酷的 AI 工具，支持图像/视频生成、AI 消除、画质增强、补帧，霓虹风格界面！

## 功能
- **图像生成**: 文本描述生成图像，支持辅助图片、描述扩充、艺术风格。
- **视频生成**: 文本生成视频，支持音频、15秒以上分段生成。
- **AI 消除**: 画笔/橡皮移除对象，实时预览。
- **画质增强**: 图像/视频增强至 2K/4K。
- **补帧**: 视频帧率提升至 60/90/120 FPS。

## 环境要求
- **OS**: Windows 11
- **GPU**: NVIDIA RTX 4060 (8GB, CUDA 13.0)
- **Python**: 3.13
- **磁盘**: 50GB

## 安装步骤
1. **安装 Python 3.13**
   - 下载: [python.org](https://www.python.org/downloads/release/python-3130/)
   - 勾选“Add Python to PATH”
   - 验证: `python --version`

2. **配置 CUDA 13.0**
   - 你的 RTX 4060 已支持，验证: `nvidia-smi`

3. **安装 FFmpeg** (音频处理)
   - 下载: [ffmpeg.org](https://ffmpeg.org/download.html)
   - 添加到 PATH: `C:\Program Files\ffmpeg\bin`

4. **创建虚拟环境**
   ```bash
   python -m venv env
   env\Scripts\activate
