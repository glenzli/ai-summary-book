# 资料源与引用规则

本书不是泛泛综述。每个核心定义、定理和证明路线都应能追溯到正式数学资料。

## 核心来源

1.  Peter Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658, 2026.  
    链接：<https://arxiv.org/abs/2605.03658>  
    用途：本书主线来源。该文整理自 2019 年 Bonn 课程，主题包括凝聚集合、凝聚阿贝尔群、紧生成性、$\operatorname{Ext}$ 计算、固体阿贝尔群和 condensed/solid 向量空间等。

2.  Dagur Asgeirsson, Riccardo Brasca, Nikolas Kuhn, Filippo Alberto Edoardo Nuccio Mortarino Majno di Capriglio, Adam Topaz, *Categorical Foundations of Formalized Condensed Mathematics*, arXiv:2407.12840, 2024.  
    链接：<https://arxiv.org/abs/2407.12840>  
    用途：coherent topology、regular/extensive topology 与 compact Hausdorff 站点上 sheaf 条件的形式化基础；该文结果已 Lean 形式化。

3.  Dustin Clausen and Peter Scholze, *Condensed Mathematics and Complex Geometry*, arXiv:2605.11731, 2026.  
    链接：<https://arxiv.org/abs/2605.11731>  
    用途：后续解析与复几何应用背景；第一卷只用作方向标，不提前引入全部理论。

4.  Bhargav Bhatt and Peter Scholze, *The pro-étale topology for schemes*, arXiv:1309.1198.  
    链接：<https://arxiv.org/abs/1309.1198>  
    用途：pro-étale 思想背景、极不连通对象和后续几何动机；不是第一卷的主要定义来源。

5.  The Stacks Project, chapters on sites and sheaves.  
    链接：<https://stacks.math.columbia.edu/>  
    用途：站点、Grothendieck 拓扑、sheaf 条件的一般参考。正文若使用其具体命题，应补充章节或 tag。

6.  Andrew M. Gleason, *Projective topological spaces*, Illinois Journal of Mathematics, 1958.  
    链接：<https://doi.org/10.1215/ijm/1255454110>  
    用途：极不连通紧 Hausdorff 空间与 compact Hausdorff 范畴中投射对象的关系；Gleason cover 的经典来源。

7.  Peter T. Johnstone, *Stone Spaces*, Cambridge Studies in Advanced Mathematics.  
    链接：<https://doi.org/10.1017/CBO9780511629576>  
    用途：Stone 对偶、profinite/Stone 空间、极不连通空间背景。

8.  Franziska Böhnlein, Benjamin Bruske, Sven-Ake Wegner, *Condensed mathematics through compactological spaces*, arXiv:2512.14612, 2025.  
    链接：<https://arxiv.org/abs/2512.14612>  
    用途：compactological spaces 与 quasiseparated condensed sets 的比较；作为后续补充视角，不作为第一卷核心定义来源。

9.  Dagur Asgeirsson, *Towards solid abelian groups: A formal proof of Nöbeling's theorem*, arXiv:2309.07252, 2023/2024.  
    链接：<https://arxiv.org/abs/2309.07252>  
    用途：solid abelian groups 背景、Nöbeling 定理和形式化证明；用于后续 solid 章节。

## 章节依赖映射

- 第 1-5 章依赖 S26 Remarks 1.4-1.5、Proposition 1.7、Definition 2.1、
  Proposition 2.9 与 Definition 2.11 的 universe、拓扑空间凝聚化、截断与大凝聚范畴
  口径，以及 ABKMT24 的 coherent topology、regular/extensive topology 和
  ProFin/CHaus/Stonean 比较；出版 locator 见总台账 1.1。
- 第 6-8 章依赖输入 A.1-A.3：Boolean prime ideal theorem、Sikorski/Gleason 背景和极不连通紧 Hausdorff 空间的投射性；书内只证明 Stone/Gleason cover 的形式模块。
- 第 9-11 章使用 sheaf of modules 的标准闭对称幺半结构和导出张量理论。
- 附录 G 使用标准同调代数，计算投射对象上的 Ext 消失、平坦对象上的 Tor 消失和长正合列。
- 附录 H 使用 sheafification 的 plus 构造、Grothendieck 阿贝尔范畴和无界替换理论；
  plus 构造给出书内证明，K-injective 精确引用 Stacks Tag `079P` Theorem 19.12.6，
  ringed-site 上 termwise-surjective flat K-flat 替换引用 Tag `06YL` Lemma 21.17.11。
  本书不把后一结论推广到任意闭幺半 Grothendieck 阿贝尔范畴。
- 附录 I 使用阿贝尔范畴中的标准同调代数：投射分解比较定理、horseshoe lemma、短正合复形的长正合列和维数平移；这些证明在书内给出。
- 附录 J 使用 regular open algebra、Stone 空间和紧 Hausdorff 正规性，补齐 Gleason cover 的连续满射构造；Gleason 投射性定理仍作为外部输入。
- 附录 K 使用 Gleason cover 存在性、ED 投射性和 sheaf separated/local lifting 条件，证明 ED 测试对象检测单射、满射、同构和正合性；不新增外部来源。
- 附录 L 使用基础 sheaf 理论、拓扑空间上的连续函数 sheaf、普通张量积和拓扑阿贝尔群的标准边界例子；不新增 condensed 数学输入。
- 附录 M 使用第一卷附录 G-I 的同调代数规则，给出有限离散对象、两项投射分解、$\mathbb Z/n$ 型对象和 Tor 的工作例题；不新增外部来源。
- 附录 N 使用 Boolean 代数、超滤子、Stone 对偶和 profinite 逆极限表示；Boolean prime ideal theorem 作为集合论输入，Stone 空间背景参考 Johnstone。
- 附录 O 使用 regular open algebra、完备 Boolean algebra、Sikorski extension theorem、Stone 对偶和 Gleason lifting theorem，补 Gleason 投射性的证明模块。
- 附录 P 使用 Asgeirsson 的 Nöbeling 定理形式化证明、有限商连续函数和超限过滤代数引理，补 Nöbeling 定理的证明模块。
- 第 12-13 章依赖输入 B.1-B.4：Nöbeling theorem、solidification、solid tensor product 和 profinite measure tensor formula；出版 locator 见总台账第 2 节。
- 第 14-15 章依赖输入 C.1 和 C.4，以及第二卷 D.4、D.7 的 analytic localization/rational descent 口径；第一卷只写所需定义、结构定理和纲要。

## 引用纪律

- 正文不大量转述来源原文；只重写数学内容。
- 若某个定理来自来源但本书暂不证明，必须标注“来源”和“后续依赖程度”。
- 外部输入的精确编号统一引用总目录 [INPUT_THEOREM_REGISTER.md](../INPUT_THEOREM_REGISTER.md)，文献定位统一引用 [REFERENCE_LOCATOR_LEDGER.md](../REFERENCE_LOCATOR_LEDGER.md)。
- 若本书给出自己的证明，需要检查证明是否只使用本书已建立的引理。
- 若不同来源采用不同站点口径，例如 compact Hausdorff、profinite、extremally disconnected 或 $\kappa$-small 版本，必须在正文说明口径差异。
