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

## 引用纪律

- 正文不大量转述来源原文；只重写数学内容。
- 若某个定理来自来源但本书暂不证明，必须标注“来源”和“后续依赖程度”。
- 若本书给出自己的证明，需要检查证明是否只使用本书已建立的引理。
- 若不同来源采用不同站点口径，例如 compact Hausdorff、profinite、extremally disconnected 或 $\kappa$-small 版本，必须在正文说明口径差异。
