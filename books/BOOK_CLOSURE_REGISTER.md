# 书籍收口登记表

登记日期：2026-07-17。

本表按每本书自身的体裁判断收口状态；不把随笔集、研究方法书、技术综述和严格数学教材强行套用同一个标准。这里的“收口”表示当前正文、读者入口、状态边界和配套材料已经能支持稳定阅读与后续维护；它不等于所有书都达到出版社 camera-ready 终稿。

## 总判定

当前 `books/` 下带 `README.md` 的书籍均已有明确状态入口。数学与物理教材已经区分书内证明、外部输入和研究边界；技术综述、研究方法书、概念书和随笔集按各自体裁完成读者入口与维护边界说明。后续工作应优先归入出版维护、事实更新、locator 精细化、索引和局部行文润色，而不是继续横向扩张主线。

## 逐书登记

| 书籍 | 体裁 | 当前收口状态 | 后续维护边界 |
| --- | --- | --- | --- |
| [随机鹦鹉解剖学](stochastic-parrot-anatomy/) | 六卷多文类 AI 总本 | 总导论、六卷 78 章、技术附录与 417 道练习内容闭合 | 错误修正、独立审阅、动态来源维护、出版排版 |
| 七部原 AI 专题稿 | 历史编辑输入 | 重要内容已进入总本，源目录已于 2026-07-16 物理退役 | 原貌由 Git 基线提交 `5fb99072860015305a83b1fe0e2644e6e125c4af` 保存 |
| [范畴论](category-theory/) | 严格数学教材 | 出版级数学内容收口稿 | 出版排版、索引、页码级 locator |
| [Chromatic Homotopy Theory](chromatic-homotopy-theory/) | 高级数学教材 | 教材内容基本收口稿 | theorem/page locator、ANSS/低 stem 表、前沿版本维护 |
| [凝聚数学讲义](condensed-mathematics/) | 四卷输入定理型数学教材 | 主线输入定理型收口稿 | 外部深定理 locator、分卷排版和教师手册化 |
| [EFT/SMEFT](effective-field-theory-smeft/) | 物理/数学结构教材 | 第一版内部收口稿 | 出版校对、外部工具接口、非阻塞参考手册扩展 |
| [Geometric Representation Theory](geometric-representation-theory/) | 高级数学教材 | 教材内容收口稿 | 出版级 locator、交叉引用、索引和模型假设边界 |
| [Homological Mirror Symmetry](homological-mirror-symmetry/) | 高级数学教材 | 完整在线教材内容本体已收口 | 页码级 locator、稳定 label、出版 copy-editing |
| [同伦类型论与单值基础](homotopy-type-theory/) | 严格数学教材 | 教材内容收口稿 | 对象语言/元语言边界维护、出版校对 |
| [Langlands 纲领](langlands-program/) | 严格数学教材 | 审定前闭合版，尚非最终出版审定版 | 来源页码、排版审稿、术语和索引维护 |
| [Motivic Homotopy and Six Functors](motivic-homotopy-six-functors/) | 高级数学教材 | 完整教材可读版和学术教学闭合草稿 | 自动化交叉引用、长篇解答、最终版式 |
| [Operad Theory](operad-theory/) | 严格数学教材 | 数学主线已按完整在线教材严格草稿收口 | HPT/signs、模型比较和出版级 locator 维护 |
| [Prismatic / p-adic Hodge Theory](prismatic-p-adic-hodge-theory/) | 高级数学教材 | 逐章教材收口草稿，可在线阅读 | classical comparison 源选择、Bhatt--Lurie preliminary 接口、Nygaard/Tate twist normalization、copy-editing |
| [量子力学](quantum-mechanics/) | 数学化物理教材 | 教材内容与阅读排版收口稿 | 出版排版、题解教师手册化、索引维护 |
| [相对论讲义](relativity/) | 数学化物理教材 | 正式教材范围的一版闭合草稿 | 全书收束强化、出版校对 |
| [String Theory](string-theory/) | 数学化物理教材 | 严格教材第二版，内容层面收口 | 例题、习题、局部证明、附录公式表和出版排版 |

## 维护规则

1. 新增内容必须先判断属于正文主线、配套材料、外部输入更新、出版校对还是另卷目标。
2. 不得把外部深定理、动态产品能力或研究边界改写成无条件书内定理。
3. README 面向读者；审计、locator、closure 和验证脚本说明保留在配套文件中。
4. 后续提交前至少运行 `git diff --check`，涉及 OET 严格数学书时再运行相应本地审计脚本。
