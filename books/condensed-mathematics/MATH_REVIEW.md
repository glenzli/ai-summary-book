# 数学审查记录

本文件记录《凝聚数学讲义》当前版本的数学审查结果。它不是正文，而是后续扩写时的质量控制清单。

## 已修正问题

1.  资料源编号修正  
    旧版本误把 Asgeirsson 基础资料标为 arXiv:2105.07888，并误把 *Condensed Mathematics and Complex Geometry* 标为 arXiv:2205.06130。当前已改为可核查来源：
    - Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658.
    - Asgeirsson 等, *Categorical Foundations of Formalized Condensed Mathematics*, arXiv:2407.12840.
    - Clausen-Scholze, *Condensed Mathematics and Complex Geometry*, arXiv:2605.11731.

2.  第八章正合性检测补充说明  
    在证明 ED 测试对象检测正合性时，已明确使用取值函子 $(-)(E)$ 的正合性，因此它保持 image、kernel 和 quotient。否则从 $I(E)=K(E)$ 推到 $Q(E)=0$ 会显得跳步。

3.  第九章删除未构造的内部 Hom 命题  
    旧稿中曾提到尚未构造的内部 Hom。当前已删除该命题，只保留后续自由模所需的 $R[\underline S]=R\otimes\mathbb Z[\underline S]$。

4.  第十一章右正合性降格为标准证明说明  
    张量积右正合性依赖 sheaf of modules 的一般理论。当前版本不再假装已经完整构造内部 Hom，而是明确标注为标准性质，后续可在附录补证。

5.  文风规则检查  
    已移除正文中违反 `SKILL.md` 的“显然”式跳步表达，并改写为具体理由。

6.  已补内部附录  
    当前已新增附录 A-C，分别处理 universe、小性约定、站点比较定理、阿贝尔群值 sheaf 的范畴性质和局部满射判据。第五章、第七章、第八章对这些结果的依赖现在已有书内参照。

7.  已补 solid 与 analytic 章节  
    当前已新增第十二至十五章，覆盖 solid abelian groups、solid tensor product、analytic rings、全局化与相干对偶纲要。长证明均以“证明说明”标出，不把外部定理伪装为书内已证结果。

8.  已补拓扑和模论附录  
    当前已新增附录 D-F：Stone 对偶与 Gleason cover、sheaf 模和内部 Hom、Nöbeling 定理与 solid 计算。第十二章中 $\mathbb Z^\square[S]\cong\prod_I\underline{\mathbb Z}$ 的证明已修正为对所有 profinite 测试对象 $T$ 检查，而不是只比较全局截面。

## 当前依赖的外部结果

以下结果当前作为引用使用，尚未在本书内证明：

1.  Gleason 定理  
    极不连通紧 Hausdorff 空间等价于 compact Hausdorff 范畴中关于满射的投射对象。

2.  Gleason cover  
    任意紧 Hausdorff 空间存在极不连通紧 Hausdorff 空间满射覆盖。

3.  Profinite/Stone 覆盖  
    任意紧 Hausdorff 空间可由 profinite 空间满射覆盖。

4.  站点比较定理  
    基子站点诱导的 sheaf 范畴等价。附录 B 已给出本书所需版本的证明框架；若要达到正式出版级别，还需补充更完整的共同细化和自然性细节。

5.  Sheaf of modules 的一般代数  
    包括张量积右正合性、相对张量积、内部 Hom、导出张量的标准构造。附录 E 已给出本书所需版本；出版级细节仍需引用 ringed topos 或 Grothendieck 范畴的标准 K-flat 理论。

6.  Nöbeling 定理  
    任意 profinite $S$ 上 $C(S,\mathbb Z)$ 自由。附录 F 给出 metrizable 情形证明和一般情形说明，但一般定理仍作为外部引用。

7.  Solid 范畴结构定理  
    solid abelian groups 构成阿贝尔全子范畴、存在 solidification、乘积型对象给出紧投射生成元等。正文第十二章引用 Scholze 讲义。

8.  Solid 张量积定理  
    $\otimes^\square$ 的对称幺半性、派生版本以及乘积公式仍引用 Scholze 讲义。

9.  Analytic ring 结构定理  
    解析模范畴的阿贝尔/导出性质、解析化左伴随和对称幺半结构仍引用 Scholze 讲义。

10. 全局化与相干对偶  
    rational localization、$f_!$、$f^!$ 和投影公式只给纲要；尚未建立完整六函子形式。

## 后续应补证明的章节

- 第二卷可继续补：liquid vector spaces、完整 analytic rings 技术条件、相干对偶中的 $f_!$ 细节、复几何应用。
- 若要把第一卷提升到出版级，需要把附录 B、D、E、F 中标为“证明框架”的部分补成完整逐步证明。
- 基本 Ext 计算当前只在第八章作为第一层示例出现；更系统的 Ext 表应另设一章。

## 当前数学口径

当前版本可以作为第一卷完整草稿使用，但不能声称已经完整证明所有引用定理。正文中凡使用未证明定理，均应保留“证明说明”或“引用结果”标记。
