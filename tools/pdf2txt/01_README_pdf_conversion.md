# PDF 转文本转换说明

## 现状

`scratch/c228.txt` 是从 `docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf` 转换而来的纯文本文件，用于方便 AI 分析和参考。

**文件信息**：
- 原始 PDF：3.4 MB，71 页
- 转换文本：138 KB，3,265 行
- 创建时间：2026/9/4 9:21:51
- 编码：UTF-8（但中文显示为乱码，可能是转换环境编码问题）

**转换特征**：
- 每页末尾插入换页符 `\f`（ASCII 12）
- 保留原始排版和换行
- 符合 Poppler `pdftotext` 的典型输出格式

**项目中不存在原始转换脚本**，推测是通过临时命令行或在线工具完成。

---

## 如何重新转换

### 方法 1：使用本项目提供的脚本（推荐）

```powershell
# 安装依赖（三选一）
pip install pypdf      # 轻量级，纯 Python
pip install pdfplumber # 功能强大，依赖 Pillow
pip install PyMuPDF    # 最快，但依赖 C 库

# 运行转换
python scratch/convert_pdf_to_txt.py
```

**脚本功能**：
- 自动检测已安装的 PDF 库（pypdf > pdfplumber > PyMuPDF）
- 保留分页控制符（与原始 c228.txt 格式一致）
- UTF-8 编码输出
- 生成到 `scratch/c228_converted.txt`（避免覆盖原文件）

---

### 方法 2：使用 Poppler pdftotext（命令行）

**Windows 安装 Poppler**：
```powershell
# 使用 Chocolatey
choco install poppler

# 或下载预编译版本
# https://github.com/oschwartz10612/poppler-windows/releases
```

**转换命令**：
```powershell
pdftotext -layout -enc UTF-8 "docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf" "scratch/c228_new.txt"
```

参数说明：
- `-layout`：保留原始排版
- `-enc UTF-8`：强制 UTF-8 输出（避免中文乱码）

---

### 方法 3：使用 WSL（如果已安装）

```bash
# WSL 中通常已包含 poppler-utils
sudo apt update && sudo apt install poppler-utils

# 转换（注意路径转换）
pdftotext -layout -enc UTF-8 \
  "/mnt/c/Users/GZC/Desktop/CUM/260901/2023cumcm-C/docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf" \
  "/mnt/c/Users/GZC/Desktop/CUM/260901/2023cumcm-C/scratch/c228_new.txt"
```

---

## 编码问题修复

如果转换后中文显示为乱码（如原 `c228.txt` 所示），可能原因：

1. **转换时未指定 UTF-8**：
   - 解决：添加 `-enc UTF-8` 参数（pdftotext）
   - 或在 Python 脚本中明确 `encoding='utf-8'`

2. **Windows 终端编码问题**：
   ```powershell
   # 临时设置终端为 UTF-8
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   ```

3. **PDF 本身字体嵌入问题**：
   - 某些 PDF 使用特殊字体编码，pdftotext 无法正确识别
   - 解决：尝试其他工具（如 pdfplumber、Adobe Acrobat）

---

## 验证转换质量

```powershell
# 检查行数
(Get-Content scratch/c228_new.txt).Count

# 检查换页符数量（应为 70 左右）
([IO.File]::ReadAllBytes('scratch/c228_new.txt') | Where-Object { $_ -eq 12 }).Count

# 检查中文显示
Get-Content scratch/c228_new.txt -Encoding UTF8 | Select-Object -First 20
```

预期结果：
- 行数：3,200-3,300 行
- 换页符：70 个（71 页 PDF = 70 个分页符）
- 中文正常显示（无乱码）

---

## 注意事项

1. **不要覆盖原文件**：转换后的文件命名为 `c228_converted.txt` 或 `c228_new.txt`，保留原 `c228.txt` 作为历史参考

2. **scratch/ 目录不会提交到 Git**：该目录已在 `.gitignore` 中排除，适合存放临时文件

3. **如果只需要阅读**：可以直接用 PDF 阅读器打开原始文件，无需转换

4. **如果需要 AI 分析**：现代 AI 工具（如 Claude、GPT-4）通常可以直接读取 PDF，无需预先转换为文本

---

## 相关文件

- 原始 PDF：`docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf`
- 现有文本：`scratch/c228.txt`（乱码版本）
- 转换脚本：`scratch/convert_pdf_to_txt.py`（本次新建）
- 转换方法：`scratch/pdf_to_txt_methods.md`（详细方法对比）
