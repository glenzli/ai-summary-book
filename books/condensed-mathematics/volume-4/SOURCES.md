# 第四卷资料源

## 核心来源

1. Peter Scholze, *Lectures on Condensed Mathematics*, arXiv:2605.03658.  
   链接：<https://arxiv.org/abs/2605.03658>

2. Dagur Asgeirsson, Riccardo Brasca, Nikolas Kuhn, Filippo Alberto Edoardo Nuccio Mortarino Majno di Capriglio, Adam Topaz, *Categorical Foundations of Formalized Condensed Mathematics*, arXiv:2407.12840.  
   链接：<https://arxiv.org/abs/2407.12840>

3. Dagur Asgeirsson, *Towards solid abelian groups: A formal proof of Nöbeling's theorem*, arXiv:2309.07252.  
   链接：<https://arxiv.org/abs/2309.07252>

4. Bhargav Bhatt and Peter Scholze, *The pro-etale topology for schemes*, arXiv:1309.1198.  
   链接：<https://arxiv.org/abs/1309.1198>

5. Clark Barwick and Peter Haine, *Pyknotic objects, I. Basic notions*.
   链接：<https://arxiv.org/abs/1904.09966>
   用途：pyknotic 对象和凝聚同伦方向的背景。

6. Sebastian Wolf, *The Pro-Étale Topos as a Category of Pyknotic Presheaves*.
   链接：<https://arxiv.org/abs/2012.10502>
   用途：coherent scheme 的 hypercomplete pro-étale $\infty$-topos 与
   $\operatorname{Gal}(X)$ 的 pyknotic 连续表示范畴之等价。

7. Jacob Lurie, *Higher Algebra*.
   链接：<https://www.math.ias.edu/~lurie/papers/HA.pdf>
   用途：稳定 $\infty$-范畴、$t$-结构、导出范畴、mapping spectra 与稳定化。

8. Jacob Lurie, *Higher Topos Theory*.
   链接：<https://www.math.ias.edu/~lurie/papers/highertopoi.pdf>
   用途：hypersheaf、hypercompletion、几何态射与高阶下降。

## 章节依赖映射

- 第 1 章依赖 ABKMT24 的 coherent topology 与 condensed set equivalence locator，并引用第一卷固定 universe 下的站点比较口径。
- 第 2-4 章依赖第一、二卷中的站点、同调和输入 B.1-B.4 的 solid 结构；solid 计算只在 Scholze/A23 输入允许范围内使用。
- 第 5-6 章依赖第二卷输入 C.1-C.5 的 analytic/liquid 内容。Banach/Fréchet 的
  liquid membership 精确引用 CS26 Theorem 0.2.14、Lemma 0.2.16 与 Theorem 0.3.11；
  连续满射和 cohomology 的 exactness 则使用第二卷命题 5.9 的局部提升判据，不归入
  membership 输入。
- 第 7 章依赖 Bhatt--Scholze Definition 1.2 的 weakly étale/fpqc 站点口径与 Theorem 1.5 的 w-contractible 覆盖输入；有限 étale 分支的 sheaf 等化子、覆盖检测和分裂 Čech 收缩在正文证明，不把 pro-etale 对象与紧 Hausdorff 测试对象直接等同。Wolf 的 Theorem 3.11 作为 hypercomplete pro-étale--pyknotic 比较输入。
- 第 8 章依赖 *Higher Algebra* 的稳定范畴、$t$-结构和导出范畴语言，依赖 *Higher Topos Theory* 与 Barwick--Haine 的 hypersheaf/hypercompletion 口径，并采用谱值 solid/analytic Bousfield localization 作为外部输入；循环凝聚谱的 Ext/Tor homotopy groups 和 Dirac cone 的 mapping-spectrum 后果在正文计算，monoidal localization 与六函子相容性保留为明确开放条件。
- 附录 A 依赖形式化基础论文、第一卷站点比较和一般同调代数。
- 附录 B 依赖第二、三章的证明与第四至七章的例子。
- 附录 C 依赖第二卷 solid/analytic/liquid 资料。
- 附录 D 依赖 Bhatt-Scholze pro-etale 论文和第一卷站点语言。
- 附录 E 依赖 pyknotic objects 的基本定义、第一卷 sheaf 理论和第四卷形式化语言。
- 附录 F 依赖第一卷站点、sheafification、ED 投射和 Ext/Tor 证明链，以及形式化基础论文中的 coherent topology 口径。
- 附录 G 依赖谱值 sheaf、pyknotic objects、第一卷 sheaf 理论和第二卷 Bousfield localization 语言。
