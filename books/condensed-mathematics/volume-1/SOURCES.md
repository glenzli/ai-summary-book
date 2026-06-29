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

- 第 1-5 章主要依赖 Scholze 讲义第一、二讲以及形式化基础论文中的站点比较结果。
- 第 6-8 章依赖 Gleason 的极不连通空间定理、Stone 对偶和 Scholze 讲义中的投射生成元思想。
- 第 9-11 章使用 sheaf of modules 的标准闭对称幺半结构和导出张量理论。
- 附录 G 使用标准同调代数，计算投射对象上的 Ext 消失、平坦对象上的 Tor 消失和长正合列。
- 附录 H 使用 sheafification 的 plus 构造、Grothendieck 阿贝尔范畴和 Spaltenstein 型 K-flat/K-injective 替换理论；plus 构造给出书内证明，K-flat/K-injective 存在性作为一般同调代数输入。
- 附录 I 使用阿贝尔范畴中的标准同调代数：投射分解比较定理、horseshoe lemma、短正合复形的长正合列和维数平移；这些证明在书内给出。
- 附录 J 使用 regular open algebra、Stone 空间和紧 Hausdorff 正规性，补齐 Gleason cover 的连续满射构造；Gleason 投射性定理仍作为外部输入。
- 附录 K 使用 Gleason cover 存在性、ED 投射性和 sheaf separated/local lifting 条件，证明 ED 测试对象检测单射、满射、同构和正合性；不新增外部来源。
- 附录 L 使用基础 sheaf 理论、拓扑空间上的连续函数 sheaf、普通张量积和拓扑阿贝尔群的标准边界例子；不新增 condensed 数学输入。
- 附录 M 使用第一卷附录 G-I 的同调代数规则，给出有限离散对象、两项投射分解、$\mathbb Z/n$ 型对象和 Tor 的工作例题；不新增外部来源。
- 附录 N 使用 Boolean 代数、超滤子、Stone 对偶和 profinite 逆极限表示；Boolean prime ideal theorem 作为集合论输入，Stone 空间背景参考 Johnstone。
- 第 12-13 章主要依赖 Scholze 讲义第五、六讲和 Nöbeling 定理。
- 第 14-15 章主要依赖 Scholze 讲义第七、八讲；当前只写第一卷所需定义、结构定理和纲要。

## 引用纪律

- 正文不大量转述来源原文；只重写数学内容。
- 若某个定理来自来源但本书暂不证明，必须标注“来源”和“后续依赖程度”。
- 若本书给出自己的证明，需要检查证明是否只使用本书已建立的引理。
- 若不同来源采用不同站点口径，例如 compact Hausdorff、profinite、extremally disconnected 或 $\kappa$-small 版本，必须在正文说明口径差异。
