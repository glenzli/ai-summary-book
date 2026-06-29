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

9.  已补第一卷 Ext/Tor 计算  
    当前已新增附录 G，系统整理投射对象上的 Ext 消失、极不连通自由对象的 Ext 消失、Cech 型投射分解、平坦对象的 Tor 消失、长正合列和维数平移。

10. 已补正合 sheafification 与派生工具
    当前已新增附录 H，给出 plus 构造、sheafification 正合性的书内证明、sheaf 模范畴的 Grothendieck 阿贝尔结构，以及 K-flat 派生张量的定义独立性。K-flat/K-injective 替换存在性仍作为一般同调代数输入定理。

11. 已补基础同调代数形式证明
    当前已新增附录 I，给出投射分解比较定理、horseshoe lemma、短正合复形长正合列、Ext 长正合列、Tor 长正合列和维数平移的书内证明。

12. 已补 Gleason cover 的 regular open 构造细节
    当前已新增附录 J，证明 $\operatorname{RO}(X)$、$E_X=\operatorname{Stone}(\operatorname{RO}(X))$、以及 $E_X\to X$ 的连续满射构造。Gleason 投射性定理本身仍作为外部输入。

13. 已补 ED 覆盖检测正合性的形式证明
    当前已新增附录 K，证明 ED 空间检测 sheaf 截面相等、零 sheaf、满射、单射、同构和阿贝尔 sheaf 正合性。第七、八章使用 ED 自由投射对象时，检测正合性的逻辑边界现在有完整书内参照。

14. 已补边界例子与反例
    当前已新增附录 L，说明 sheaf 满射不等于逐对象满射、separated presheaf 不一定是 sheaf、基子站点缺少交叠会破坏 sheaf 条件、普通张量积不保持无限乘积，以及拓扑阿贝尔群不能替代凝聚阿贝尔群。

15. 已补 Ext/Tor 工作例题
    当前已新增附录 M，计算有限离散自由对象、两项投射分解、$\mathbb Z_{\operatorname{cond}}/n$ 的 Ext，以及乘以 $n$ 的 Tor 模板，补足“规则到手算”的中间层。

16. 已补 Stone 对偶完整证明链
    当前已新增附录 N，证明超滤子判别、Stone 空间紧 Hausdorff 性、$B\simeq\operatorname{Clop}(\operatorname{Stone}(B))$、profinite 空间的 Stone 恢复和有限离散商逆极限表示。Boolean prime ideal theorem 仍作为集合论输入。

17. 已补 Gleason 投射性证明模块
    当前已新增附录 O，证明 regular open algebra 完备性、ED 情形下 regular open 等于 clopen、Stone 空间端的 Sikorski extension 推出投射性，并标出从 Stone 端下降到一般 compact Hausdorff 端的 Gleason lifting 输入。

18. 已补 Nöbeling 定理证明模块
    当前已新增附录 P，证明有限和可数 profinite 情形，给出一般 profinite 情形的超限过滤输入和从过滤到自由性的代数引理，并说明该定理进入 solid 计算的方式。

## 第一卷外部输入定理

以下结果作为第一卷外部输入定理使用。第一卷不再把它们列为正文缺口；第二卷会继续依赖其中的 solid/analytic 结构定理。

1.  Gleason lifting theorem
    极不连通紧 Hausdorff 空间关于 compact Hausdorff 满射的提升性质。附录 O 已证明 Boolean algebra 端的模块和反向方向，一般 compact Hausdorff 下降仍引用 Gleason。

2.  Gleason cover  
    任意紧 Hausdorff 空间存在极不连通紧 Hausdorff 空间满射覆盖。附录 J 构造 regular open Stone cover 的连续满射，投射性仍引用 Gleason。

3.  Profinite/Stone 基础
    Stone 对偶、profinite 逆极限表示已由附录 N 证明；任意紧 Hausdorff 空间的特定覆盖构造仍按正文引用拓扑输入。

4.  站点比较定理  
    基子站点诱导的 sheaf 范畴等价。附录 B 已给出本书所需版本的证明细节。

5.  Sheaf of modules 的一般代数  
    包括张量积右正合性、相对张量积、内部 Hom、导出张量的标准构造。附录 E 给出本书所需版本，附录 H 补充正合 sheafification 和派生张量的比较证明；K-flat/K-injective 替换存在性仍引用 ringed topos 或 Grothendieck 范畴的标准理论。

6.  Nöbeling 定理  
    任意 profinite $S$ 上 $C(S,\mathbb Z)$ 自由。附录 F 和 P 给出有限、可数、超限过滤模块和一般情形说明，但一般 profinite 定理仍作为外部引用。

7.  Solid 范畴结构定理  
    solid abelian groups 构成阿贝尔全子范畴、存在 solidification、乘积型对象给出紧投射生成元等。正文第十二章引用 Scholze 讲义。

8.  Solid 张量积定理  
    $\otimes^\square$ 的对称幺半性、派生版本以及乘积公式仍引用 Scholze 讲义。

9.  Analytic ring 结构定理  
    解析模范畴的阿贝尔/导出性质、解析化左伴随和对称幺半结构仍引用 Scholze 讲义。

10. 全局化与相干对偶  
    rational localization、$f_!$、$f^!$ 和投影公式在第一卷只给入口；完整六函子形式放入第二卷和后续几何部分。

## 第二卷承接内容

- solid abelian groups 的完整结构理论与派生版本。
- solid tensor product、solid rings 和 solid modules 的系统计算。
- analytic rings 的完整技术条件、解析化和 Bousfield localization。
- liquid vector spaces 与实分析方向。
- 相干对偶中的 $f_!$、$f^!$、投影公式和复几何应用。

## 当前数学口径

当前版本作为第一卷使用：基础构造在书内证明，深层外部输入定理明确标注来源和使用位置。正文中凡使用外部输入定理，均应保留“证明说明”或“输入定理”标记。
