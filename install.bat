@echo off
echo 正在配置 AI Toolkit 环境...

:: 检查 Python 是否安装
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.13（下载：https://www.python.org/downloads/）
    pause
    exit /b
)

:: 修复 pip
echo 正在修复 pip...
python -m ensurepip --upgrade
python -m pip install --upgrade pip
pip --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误: pip 安装失败，请检查 Python 安装
    pause
    exit /b
)

:: 清理 pip 缓存
python -m pip cache purge

:: 定义国内镜像列表
set MIRRORS= ^
    https://pypi.tuna.tsinghua.edu.cn/simple ^
    https://mirrors.aliyun.com/pypi/simple ^
    https://pypi.doubanio.com/simple ^
    https://mirrors.huaweicloud.com/pypi/simple

:: 安装依赖
echo 正在安装依赖...
for %%m in (%MIRRORS%) do (
    echo 尝试使用镜像: %%m
    python -m pip install -r requirements.txt -i %%m --trusted-host pypi.tuna.tsinghua.edu.cn --trusted-host mirrors.aliyun.com --trusted-host pypi.doubanio.com --trusted-host mirrors.huaweicloud.com
    if %ERRORLEVEL% equ 0 (
        echo 依赖安装成功！
        goto :install_success
    ) else (
        echo 镜像 %%m 安装失败，尝试下一个镜像...
    )
)

echo 所有镜像均安装失败，请检查网络或 requirements.txt
pause
exit /b

:install_success

:: 检查 FFmpeg
ffmpeg -version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 正在下载 FFmpeg...
    powershell -Command "Invoke-WebRequest -Uri https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip -OutFile ffmpeg.zip"
    powershell -Command "Expand-Archive -Path ffmpeg.zip -DestinationPath 'C:\Program Files\ffmpeg'"
    setx PATH "%PATH%;C:\Program Files\ffmpeg\ffmpeg-master-latest-win64-gpl\bin"
    del ffmpeg.zip
    echo FFmpeg 已安装并添加到 PATH
)

:: 创建必要文件夹
mkdir weights cache data logs
echo. > logs\error.log

:: 验证 CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
if %ERRORLEVEL% neq 0 (
    echo CUDA 配置可能有问题，请检查 NVIDIA 驱动和 CUDA 13.0
)

echo 安装完成！运行 run.bat 启动程序
pause
