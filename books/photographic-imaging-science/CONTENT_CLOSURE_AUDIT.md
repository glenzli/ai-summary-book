# 内容闭合审查

## 当前结论

本目录已形成从场景辐亮度到显示图像、从近轴光线到系统 MTF 的连续教材稿。
正文不是术语大纲：18 个编号章节均有概念定义、适用条件、推导或证明、量纲检查、
算例和练习；48 幅技术图由同一脚本确定性生成；65 道练习均在答案手册中逐题对应。
书名承诺的传感器、ISO/HDR/Log/RAW，以及镜头材料、像差和 MTF 两条主线均已闭合。

当前状态称为“内容收口候选稿”，不称为出版定稿。真实 RAW 数据集、实测光谱响应、
镜头样品统计和更多历史案例仍可作为经验材料加入，但它们不再承担主线概念闭合责任。

## 内部闭合

- 辐亮度、辐照度、曝光量和光子预算：第 1 章。
- Poisson 稀疏化、散粒噪声和 4T 像素：第 2 章。
- 增益位置、量化噪声、配对平场 PTC、噪声预算与动态范围：第 3--5 章。
- 快门时序、BSI/DTI/堆栈/曲面结构：第 6 章。
- CFA 采样、混叠、去马赛克和色彩矩阵：第 7 章。
- 多帧、包围曝光与片上 HDR：第 8 章。
- 连续分段 Log、EI、量化误差与 RAW/视频 RAW 数据契约：第 9--10 章。
- 薄/厚系统矩阵、主平面、瞳、PSF/OTF/MTF 与采样：第 11--12 章。
- Zernike 像差、Strehl、Sellmeier 色散、消色差与二级光谱：第 13--14 章。
- 非球面、衍射元件效率与多层膜矩阵：第 15 章。
- 镜头运动结构、景深、防抖、斜边 MTF 与不确定度：第 16--17 章。

## 图版与复现

- 48 个 SVG 均由 `figures/generate_figures.py` 生成，不依赖外部图片或隐藏二进制源。
- 每幅图恰在正文引用一次，图号、替代文本、相邻图注与 SVG `aria-label` 一致。
- 图注区分解析曲线、机制示意和经验测量；示意参数不被表述为具体器材实测值。
- 2026-07-18 已完成全部 SVG 的栅格化联系表巡检，未发现空白、裁切或文字重叠。

## 外部输入边界

完整硅器件工艺、现代 HDR/去马赛克算法实现、ISO 合格性文本、复杂镜头处方优化、
玻璃目录与制造容差不在书内重建。它们在 [SOURCES.md](SOURCES.md) 和
[CHAPTER_SOURCE_NOTES.md](CHAPTER_SOURCE_NOTES.md) 中按用途登记。正文不以厂商
缩写替代一般物理推导。

## 审查命令

```bash
python3 books/photographic-imaging-science/figures/generate_figures.py
python3 books/photographic-imaging-science/validate.py
python3 books/audit_oet_rigor.py photographic-imaging-science --strict
python3 books/audit_textbook_narrative.py photographic-imaging-science --strict
git diff --check -- books/photographic-imaging-science
```

脚本检查证明清单和结构闭合，不把机械通过等同于真理。数学、物理和工程口径仍按
`SKILL.md` 逐章审读；本次收口已完成对应的第二轮人工审读。
