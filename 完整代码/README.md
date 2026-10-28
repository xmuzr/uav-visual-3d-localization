# 无人机视觉目标高精度3D定位算法

## 快速启动

推荐 Windows 64位、Python 3.13。本地交付包可含该环境的离线依赖包；GitHub公开版本通过联网安装依赖。
首次使用双击“安装依赖.cmd”，成功后双击“启动.cmd”；浏览器自动打开 http://127.0.0.1:5050 。
独立多帧界面双击“启动多帧.cmd”，地址 http://127.0.0.1:5051 。关闭终端或按 Ctrl+C 停止服务。

程序默认加载同级“数据集/原始照片”，元数据读取“数据集/dji.md”。整个项目目录可以一起移动。
电脑须安装Python并将python加入PATH；安装脚本与启动脚本使用相同的python命令。

## 命令行

在“完整代码”中打开终端：

```powershell
python -m pip install --no-index --find-links dependency_wheels -r requirements.txt
python -B run.py
python -B run.py --multi
python -B locate_cli.py --config input_plane_pose_example.json --output result.json
python -B locate_cli.py --config input_planar_gcps_example.json --output result.csv
python -B multi_view_cli.py --config input_multi_view_intersection_example.json --output multi.json
python -B multi_view_cli.py --config input_lrf_height_plane_example.json --output plane.json
```

Python 3.10—3.12需联网安装对应版本依赖：`python -m pip install -r requirements.txt`。
没有dependency_wheels目录时执行 `python -m pip install -r requirements.txt`。安装依赖.cmd会按离线包是否存在自动选择安装方式。
本地附带离线wheel仅供Windows x64 / CPython 3.13使用。Python 3.14及以上不在本包支持范围。
服务端口占用时执行 `python -B run.py --port 5060`；指定 `--no-browser` 可关闭自动打开网页。
两份真实照片配置的相对图像路径以配置文件目录为基准，支持从其他工作目录使用绝对配置路径启动。

## 操作与文件

- 单图：加载照片，点选像素，核对自动填入的内参、位置、姿态与高度，点击“定位”。
- 多帧：选取至少两张照片，分别记录同一静止物理点，点击交会计算。主界面也提供多照片模式。
- 核心库：image_localization.py、multi_view_localization.py。
- 接口：locate_cli.py、multi_view_cli.py、photo_localizer_app.py、multi_view_app.py。
- templates和static为界面必需资源；dependency_wheels为离线运行依赖。
- 四份input_*_example.json是输入模板；平面位姿和GCP模板含演示参数，多帧中心像素需按实际同名点修改。

本地交付包的详细原理、参数、接口和数据说明见项目根目录“技术文档.docx”。公开版本见根目录README.md及docs目录；不附带原始照片、元数据与验收文档。无照片演示执行 `python -B ../examples/synthetic_demo.py`。
实验图中的点间距离和交会残差用于描述内部一致性；独立真值误差需要另外评价。
