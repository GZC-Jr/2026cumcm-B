# PDF 转 TXT 方法记录

## 背景

`scratch/c228.txt` 是从 `docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf` 转换而来的纯文本文件，用于参考论文方法论。

**文件特征**：
- 大小：138,436 字节
- 创建时间：2026/9/4 9:21:51
- 包含 70 个换页符（每页一个 `\f`）
- 原始编码问题导致中文显示为乱码

## 原始转换方式（推测）

基于换页符模式，最可能使用了 **Poppler `pdftotext`** 工具：

```bash
# 如果有 pdftotext（WSL/Git Bash/单独安装）
pdftotext -layout "docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf" "scratch/c228.txt"
```

**注意**：转换时未正确处理中文编码，导致文本中中文显示为乱码。

---

## 推荐复现方法

### 方法 1：使用 Python + PyMuPDF（推荐）

**优点**：纯 Python，跨平台，编码可控

```python
# 安装依赖
pip install PyMuPDF

# 转换脚本（创建为 scratch/convert_pdf.py）
import fitz  # PyMuPDF
from pathlib import Path

pdf_path = Path("docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf")
txt_path = Path("scratch/c228_clean.txt")

doc = fitz.open(pdf_path)
text_parts = []

for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text", sort=True)
    text_parts.append(text)
    # 页间分隔符（可选）
    if page_num < len(doc):
        text_parts.append("\f")  # 换页符

full_text = "".join(text_parts)
txt_path.write_text(full_text, encoding="utf-8")
doc.close()

print(f"转换完成：{len(doc)} 页 -> {txt_path}")
```

**运行**：
```powershell
python scratch/convert_pdf.py
```

---

### 方法 2：使用 pdftotext（需单独安装）

**Windows 安装 Poppler**：

1. 下载 Poppler for Windows：https://github.com/oschwartz10612/poppler-windows/releases
2. 解压到 `C:\Program Files\poppler\`
3. 添加 `C:\Program Files\poppler\Library\bin` 到系统 PATH

**转换命令**：
```powershell
pdftotext -layout -enc UTF-8 `
  "docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf" `
  "scratch/c228_clean.txt"
```

**参数说明**：
- `-layout`：保留原始排版
- `-enc UTF-8`：指定输出编码为 UTF-8（避免中文乱码）

---

### 方法 3：使用 pdfplumber（Python）

**优点**：对表格支持好，纯 Python

```python
# 安装依赖
pip install pdfplumber

# 转换脚本
import pdfplumber
from pathlib import Path

pdf_path = Path("docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf")
txt_path = Path("scratch/c228_clean.txt")

with pdfplumber.open(pdf_path) as pdf:
    text_parts = []
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
            text_parts.append("\f")  # 页间分隔
    
    full_text = "".join(text_parts)
    txt_path.write_text(full_text, encoding="utf-8")

print(f"转换完成：{len(pdf.pages)} 页")
```

---

## 编码问题修复

如果需要修复现有 `c228.txt` 的编码问题：

```python
from pathlib import Path

# 尝试多种编码读取
for encoding in ['utf-8', 'gbk', 'gb18030', 'latin1']:
    try:
        text = Path("scratch/c228.txt").read_text(encoding=encoding)
        # 检查中文是否正常
        if '蔬菜' in text or '定价' in text:
            print(f"✓ 正确编码：{encoding}")
            Path("scratch/c228_fixed.txt").write_text(text, encoding='utf-8')
            break
    except:
        continue
```

---

## 当前状态

- ✅ 原始 PDF 存在：`docs/(C228)基于价格弹性的蔬菜类商品自动定价与补货决策.pdf`
- ✅ 转换结果存在：`scratch/c228.txt`（有编码问题）
- ❌ 转换脚本不存在：未找到项目中的转换脚本
- 📝 推测：使用临时命令行工具转换，未保存为脚本

---

## 建议

如需在其他环境复现或重新生成干净的文本文件：

1. **推荐方法 1（PyMuPDF）**：最简单，跨平台
2. 创建 `scratch/convert_pdf.py` 脚本并加入 Git（如果需要复现）
3. 在 README 中注明转换方法

---

**创建时间**：2026/9/4 15:20  
**用途**：记录 `c228.txt` 来源与复现方法
