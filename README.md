# 清华大学脑电数据分析与可视化项目

## 1. 项目概述

本项目旨在提供一个完整的工作流程，用于处理从 Intan RHD 文件格式记录的神经电生理（脑电）数据。整个流程覆盖了从原始数据转换、预处理、事件提取到最终可视化（原始信号轨迹图和时频谱图）的全过程。

该项目利用 MNE-Python 库进行核心数据处理，并通过一个中央配置文件 (`config.py`) 来管理所有参数，实现了代码逻辑与实验参数的分离，具有高度的灵活性和可扩展性。

## 2. 代码框架结构

本项目的代码结构经过精心设计，以实现模块化和易于维护：

```
.
├── config/
│   └── config.py               # 核心配置文件，管理所有路径、参数和设置
│
├── data/
│   ├── fif_data/               # (自动生成) 存放预处理后的FIF格式数据
│   └── plots/                  # (自动生成) 存放生成的可视化图表
│
├── scripts/
│   ├── process_rhd_to_fif.py   # 主处理脚本：读取RHD -> 预处理 -> 保存为FIF
│   ├── plot_raw_trace.py       # 可视化脚本：绘制单个试次的原始信号轨迹
│   └── plot_spectrogram_grid.py # 可视化脚本：按脑区绘制所有通道的时频谱图
│
└── utils/
    ├── importrhdutilities.py   # Intan官方提供的RHD文件读取工具
    ├── rhd_event_extractor.py  # 用于从RHD数字通道中提取事件的工具
    └── preprocess.py           # 包含预处理流程函数的模块
```

-   **`config/`**: 包含项目的中央配置文件。**所有实验相关的参数修改都应在此处进行。**
-   **`data/`**: 用于存放所有生成的数据和图表，已被添加到 `.gitignore` 中，不会被Git追踪。
-   **`scripts/`**: 包含项目的主要执行脚本。
-   **`utils/`**: 包含项目所需的辅助函数和工具模块。

## 3. 使用流程

请严格按照以下步骤执行分析流程。

### 步骤 0: 准备原始数据

在开始之前，请确保您的原始 `.rhd` 文件按照以下层级结构进行组织。处理脚本 (`process_rhd_to_fif.py`) 依赖于此结构来自动识别被试、范式和试次。

```
<您的原始数据根目录>/
└── <日期文件夹>/             (例如: 20250612)
    └── <范式文件夹>/         (例如: visual, beard_fast)
        └── <试次文件夹>/     (例如: visual_250613_162743)
            ├── file_1.rhd
            ├── file_2.rhd
            └── ...
```

### 步骤 1: 修改配置文件

在运行任何脚本之前，打开 `config/config.py` 文件，根据您的实验设置和电脑环境，仔细检查并修改以下关键参数：

1.  **目录设置 (`Directory Settings`)**:
    * `RAW_DATA_DIR`: **（必须修改）** 将此路径设置为您在步骤0中组织的原始数据所在的根目录。
    * `PROCESSED_DATA_DIR`: 预处理后的 `.fif` 文件输出目录，默认为 `./data/fif_data`，通常无需修改。
    * `PLOTS_DIR`: 可视化图表的输出目录，默认为 `./data/plots`，通常无需修改。

2.  **被试与文件命名 (`Subject & File Naming`)**:
    * `SUBJECT_MAPPING`: **（必须修改）** 将您的 `<日期文件夹>` 名称映射到唯一的被试ID（例如 `'20250612': 'mouse1'`）。

3.  **数据处理设置 (`Data Processing Settings`)**:
    * `NOTCH_FREQS`, `BANDPASS_L_FREQ`, `BANDPASS_H_FREQ`, `RESAMPLE_FREQ`: 检查这些预处理参数是否符合您的分析需求。

### 步骤 2: 运行数据预处理

此步骤将遍历您 `RAW_DATA_DIR` 中的所有数据，执行预处理，并生成 MNE Python 可用的 `.fif` 文件。

打开终端，确保您位于项目的根目录下，然后执行：

```bash
python scripts/process_rhd_to_fif.py
```

执行完毕后，您会在 `data/fif_data/` 目录下看到按被试和范式组织的、经过预处理的 `_raw.fif` 和 `_events.tsv` 文件。

### 步骤 3: 运行可视化脚本

现在您可以使用干净的 `.fif` 文件来生成图表。

#### 3.1 绘制原始信号轨迹图 (Raw Trace)

1.  **配置目标**: 打开 `config/config.py` 文件。在参数区，设置您想查看的**具体目标**：
    ```python
    TARGET_SUBJECT = 'mouse1'
    TARGET_PARADIGM = 'visual'
    TARGET_TRIAL = 1

    # 设置要显示的时间窗口 (单位: 秒)
    PLOT_T_MIN = 20.0
    PLOT_T_MAX = 25.0
    ```

2.  **执行脚本**:
    ```bash
    python scripts/plot_raw_trace.py
    ```
    脚本将生成一张按脑区布局的原始信号轨迹图，并保存在 `data/plots/` 目录下。

#### 3.2 绘制时频谱图 (Spectrogram)

1.  **配置目标**: 如果再绘制原始信号轨迹图的时候已经设置过了，这里可以跳过。
    打开 `config/config.py` 文件。与上一步类似，在参数区设置您想分析的**具体目标**：
    ```python
    TARGET_SUBJECT = 'mouse1'
    TARGET_PARADIGM = 'visual'
    TARGET_TRIAL = 1

    # 设置要显示的时间窗口 (单位: 秒)
    PLOT_T_MIN = 20.0
    PLOT_T_MAX = 25.0
    ```
    您也可以在 `config/config.py` 中调整STFT的相关参数（如 `STFT_WINDOW_SIZE`）来优化时频分析的分辨率。

2.  **执行脚本**:
    ```bash
    python scripts/plot_spectrogram_grid.py
    ```
    脚本将生成一张包含所有通道、按脑区布局的时频谱图网格，并保存在 `data/plots/` 目录下。