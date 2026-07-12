# 附录 S：来源定位索引

## 目标

本附录按主题列出正文和附录中的主要来源落点。其功能是帮助读者追溯定义、定理形态、模型论输入和研究边界。

## S.1 基础类型论与 HoTT

| 范围 | 主要来源 | 用途 |
|---|---|---|
| 第 1-2 章 | HoTT Book §§1.3、1.12、2.3、Appendix A；Rijke 2025 Part I | 判断与 conversion、非累积宇宙、恒等类型、J、transport、$\mathsf{apd}$ |
| 第 3-4 章 | HoTT Book Chapters 1、3；Rijke 2025 Part I-II | 归纳类型、可收缩性、命题、集合、h-level |
| 附录 A-D | HoTT Book §§2.1-2.3；Rijke 教材 | 路径代数参考、固定端点归纳、$\Sigma$ 路径、fiber 收缩 |
| 第 5 章、附录 E/G | HoTT Book；Rijke 教材 | 等价定义、准逆、半伴随等价、fiber 可收缩 |

## S.2 单值性、外延性与结构等同性

| 范围 | 主要来源 | 用途 |
|---|---|---|
| 第 6-7 章 | HoTT Book §§2.9-2.10、4.9；Rijke 2025 Part II | 分层函数外延性、universe univalence、命题外延性、沿 $\mathsf{ua}$ 的 transport |
| 附录 F/H/I/J/AG | HoTT Book；结构等同性原则文献 | 子类型外延性、universe 非集合性、SIP、代数结构 transport |
| 附录 T | HoTT Book Definition 4.9.1、Theorems 4.9.4-4.9.5 | 精确外部输入：fiber universe 的 univalence 推出强依赖函数外延性 |

## S.3 HIT、圆与基础合成同伦论

| 范围 | 主要来源 | 用途 |
|---|---|---|
| 第 8-11 章 | HoTT Book Chapter 6；Rijke 教材 | 分层截断与商、圆、悬挂、pushout、基本群 |
| 第 9 章、附录 L | HoTT Book Chapter 6；CHM 2018 §3.3 | 公理化 HIT 输入的计算强度；列举型 cubical HIT 语义及其一般 schema 边界 |
| 附录 M/N/V/W | HoTT Book | 整数、encode-decode、基本群同构 |
| 附录 AD/AI/AY | HoTT Book；合成同伦论文献 | 悬挂、pushout 等价不变性、pushout 路径空间 |

## S.4 单值范畴论

| 范围 | 主要来源 | 用途 |
|---|---|---|
| 第 13-14 章 | HoTT Book；单值范畴论文献 | 预范畴、单值范畴、Yoneda、Rezk 完备化 |
| 附录 P/Q/U/X/AA/AF/AH | 单值范畴论和 Rezk completion 文献 | 范畴路径、Yoneda、函子范畴、终对象、伴随 |
| 附录 BE/BB | displayed categories、univalent bicategories、Rezk/Segal object 文献 | 高阶范畴接口和研究边界 |

## S.5 高级合成同伦论

| 范围 | 主要来源 | 用途 |
|---|---|---|
| 第 12 章 | HoTT Book；合成同伦论和代数拓扑文献 | 高阶同伦群、EM 型、上同调、Blakers-Massey、谱接口 |
| 附录 Y/AP/AU/AL/BF/BJ/BK/BM | 合成同伦论、Postnikov、cofiber、局部系数文献 | 证明核、外部输入和高级接口 |
| 附录 AM/AQ/AV/AZ/BN | 代数拓扑、谱序列、稳定同伦论文献 | smash product、exact couple、Serre/AHSS/Adams、Steenrod/Ext |

## S.6 构造性数学、模型与对象语言

| 范围 | 主要来源 | 用途 |
|---|---|---|
| 附录 AK/AR/AW/BA/BO | HoTT Book；构造性分析文献 | Cauchy/Dedekind 实数、连续性、紧致性、积分接口 |
| 第 15-16 章 | Kapulkin--Lumsdaine arXiv:1211.2851；CCHM DOI 10.4230/LIPIcs.TYPES.2015.5；Huber DOI 10.1007/s10817-018-9469-1；Sterling--Angiuli arXiv:2101.11479 | 相对一致性、对象/元语言边界、CCHM univalence、canonicity、Cartesian cubical normalization |
| 附录 Z/AO/BC/BG | CHM 2018；cubical model、2LTT、QIIT 文献 | HIT/QIIT 元理论、cubical 变体边界、strict equality；不得回流为基础规则 |
| 附录 AN/AS/AX/AT/BD | directed/simplicial type theory、cohesive HoTT、SDG/SAG 文献 | 扩展对象语言和几何模型边界 |

## S.7 使用纪律

1.  引用来源时必须说明其用途：定义、证明、模型论背景、经典计算或研究边界。
2.  一个来源给出模型或语义，不等于对象语言中已有构造。
3.  经典数学来源可支撑计算接口，但若要内部化到 HoTT，需要另补对象语言证明。
4.  近期研究若版本仍不稳定，只能标为外部输入或研究边界。
5.  外部语义条目必须能从 `SOURCES.md` 定位到精确论文、章节或定理，并列出未覆盖的语法与计算性质。
