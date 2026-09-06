# PDF 转 TXT 工具包

本目录包含将 PDF 文件（特别是含中文的 C228 参考论文）转换为可读文本的工具、脚本和测试记录。

## 📁 目录内容

| 文件 | 类型 | 说明 |
|------|------|------|
| `convert_pdf_to_txt.py` | Python 脚本 | PDF 转 TXT 转换脚本（支持 PyMuPDF / pdfplumber / pypdf） |
| `c228_clean.txt` | 文本 | 转换后的清洁文本文件（UTF-8，中文可读，137 KB） |
| `c228.txt` | 文本 | 原始乱码文本文件（pdftotext 缺参导致，仅作历史参考） |
| `README.md` | 文档 | 本说明文件（入口索引） |
| `01_README_pdf_conversion.md` | 文档 | PDF 转换说明（含原 c228.txt 来源分析） |
| `02_pdf_to_txt_methods.md` | 文档 | PDF 转 TXT 方法详细对比 |
| `CONVERSION_TEST_RESULT.md` | 文档 | 转换质量测试报告与对比 |

---

## 🚀 快速开始

### 一键转换

```powershell
# 进入项目根目录
cd "C:\Users\GZC\Desktop\CUM\260901\2023cumcm-C"

# 安装依赖（建议使用 PyMuPDF）
pip install PyMuPDF

# 运行转换
python scratch\pdf2txt\convert_pdf_to_txt.py
```

### 转换结果

- **输入**：`docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf`
- **输出**：`scratch/pdf2txt/c228_clean.txt`

---

## ✅ 转换效果对比

| 指标 | `c228.txt`（原始乱码） | `c228_clean.txt`（新） |
|------|---------------------|----------------------|
| 中文可读性 | ❌ 乱码 | ✅ 完全可读 |
| 文件大小 | 138,436 字节（135 KB） | 137,565 字节（134 KB） |
| 字符数 | 96,989 | 96,118 |
| 换页符数 | 70 | 69 |
| 编码 | UTF-8（错误解码） | UTF-8（正确解码） |

详见 `CONVERSION_TEST_RESULT.md`。

---

## 🔧 技术细节

### 转换库优先级（自动选择）

| 优先级 | 库 | 优点 | 适用场景 |
|--------|----|------|---------|
| 1 | **PyMuPDF** (fitz) | 最快，中文支持好 | 一般论文/书籍 |
| 2 | **pdfplumber** | 表格提取强 | 财务报表等含表格的 PDF |
| 3 | **pypdf** | 纯 Python，无 C 依赖 | 备用 / 容器环境 |

### 安装依赖

```powershell
# 全部安装（约 50 MB）
pip install PyMuPDF pdfplumber pypdf

# 按需安装
pip install PyMuPDF        # 推荐
pip install pdfplumber     # 含表格时
pip install pypdf          # 轻量备用
```

### 脚本核心逻辑

```python
import fitz  # PyMuPDF (primary)
import pdfplumber  # fallback 1
from pypdf import PdfReader  # fallback 2

# 1. PyMuPDF: open + 逐页 get_text("text", sort=True)
# 2. pdfplumber: with pdfplumber.open() as pdf: page.extract_text()
# 3. pypdf: PdfReader(path).pages[i].extract_text()
```

完整源码见 `convert_pdf_to_txt.py`。

---

## 🎯 适用场景

1. **中文 PDF 文档提取** —— 解决 `pdftotext` 的中文乱码问题
2. **论文内容分析** —— 提取论文全文供 AI/人工阅读
3. **数据采集** —— 从 PDF 中提取文字用于进一步处理
4. **全文检索** —— 将 PDF 转为可搜索的纯文本

---

## ⚠️ 常见问题

### Q1: 为什么 `c228.txt` 是乱码？

原始文件使用 `pdftotext -layout` 转换，**未指定 `-enc UTF-8`** 参数，导致中文解码为默认编码（GBK）的字节序列，再用 UTF-8 读取时变成乱码。

**对比示例**：
```
c228.txt:         ڼ۸Ե߲ƷԶ벹
c228_clean.txt:   基于价格弹性的蔬菜类商品自动定价与补货决策
```

修复方法：见 `01_README_pdf_conversion.md` 中的"编码问题修复"章节。

### Q2: 转换后中文仍是乱码？

检查 PDF 是否为扫描件（含图片而非文字）。若是，需要先 OCR（`pytesseract`）再转换。

### Q3: 怎样提取表格？

在脚本中改用 `pdfplumber.extract_tables()`，详见 `02_pdf_to_txt_methods.md` 的方法 3。

### Q4: 中文显示 `?????` 或 ``？

一般是 PowerShell 终端编码问题。运行：
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```
文件本身通常是正常的，用编辑器（VS Code / Notepad++）打开时设置编码为 UTF-8 即可。

---

## 📚 文档导航

| 想了解的内容 | 查看文件 |
|------------|---------|
| 现有 `c228.txt` 乱码的修复方法 | `01_README_pdf_conversion.md` |
| 各 PDF 库的能力与对比 | `02_pdf_to_txt_methods.md` |
| 转换质量验证结果 | `CONVERSION_TEST_RESULT.md` |
| 转换脚本源码 | `convert_pdf_to_txt.py` |

---

## 🔄 更新日志

- **2026/9/4 上午** — 通过 `pdftotext -layout`（未指定编码）生成乱码版 `c228.txt`
- **2026/9/4 下午** — 安装 PyMuPDF，运行 `convert_pdf_to_txt.py`，成功生成 `c228_clean.txt`，中文完全可读
- **2026/9/4 下午** — 编写 `CONVERSION_TEST_RESULT.md`，对比乱码/清洁版
- **2026/9/5 下午** — 整理文件到 `scratch/pdf2txt/` 目录，更新 README 索引

---

## 🔗 相关路径

- **原始 PDF**：`docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf`
- **项目根目录**：`C:\Users\GZC\Desktop\CUM\260901\2023cumcm-C`
