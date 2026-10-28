"""Start the delivered local image localization application."""
import argparse
import importlib.util
import sys
from threading import Timer
import webbrowser

sys.dont_write_bytecode = True

def main():
    parser = argparse.ArgumentParser(description="无人机视觉目标定位")
    parser.add_argument("--multi", action="store_true", help="启动独立多帧界面")
    parser.add_argument("--port", type=int, help="指定本地端口")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()
    missing = [name for name in ("flask", "PIL", "numpy") if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(1, "缺少运行依赖，请先双击 安装依赖.cmd，或执行 python -m pip install -r requirements.txt\n")
    if args.multi:
        from multi_view_app import app
    else:
        from photo_localizer_app import app
    port = args.port or (5051 if args.multi else 5050)
    if not 1 <= port <= 65535:
        parser.error("端口须在1至65535之间")
    url = f"http://127.0.0.1:{port}"
    print(f"本地访问地址：{url}；按 Ctrl+C 停止服务。", flush=True)
    if not args.no_browser:
        timer = Timer(1.0, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
