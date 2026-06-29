# 第十七章：研究边界、开放问题与版本化阅读

## 本章目标

本章总结 HoTT 与单值基础的研究边界，给出后续阅读和扩写规则。它不是新闻综述，而是教材维护清单：哪些方向已经适合进入核心教材，哪些方向只能作为高级接口，哪些方向必须保持为研究边界。

## 依赖前置知识

本章依赖全书概念和 `SOURCES.md` 的引用纪律。

## 17.1 已相对稳定的核心

**列表 17.1.** 以下内容可作为 HoTT 教材核心层：

1.  intensional dependent type theory；
2.  identity types and path induction；
3.  equivalences and univalence；
4.  homotopy levels；
5.  truncations and quotients, when HITs are accepted；
6.  circle and encode-decode proof of $\pi_1(\mathbb S^1)$；
7.  univalent categories and structure identity principle；
8.  Yoneda lemma and one-categorical Rezk completion, with the proof boundaries recorded in appendices Q, U, X, AA and AF.

这些内容仍需检查具体定义口径，但作为教材主线已经成熟。

## 17.2 活跃研究方向

**列表 17.2.** 以下方向属于活跃研究或高级接口：

1.  cubical type theory 的不同模型、规范化和计算解释；
2.  合成同伦论中的 Blakers-Massey、Freudenthal、Hopf fibration 和球面低维计算；
3.  合成上同调、谱和稳定同伦范畴；
4.  Rezk types、complete Segal objects 和 synthetic $\infty$-categories；
5.  displayed categories、univalent bicategories 和高阶单值性；
6.  HIIT、QIIT、computational HIT semantics 和 canonicity；
7.  Cauchy/Dedekind 实数、构造性分析、度量空间、级数和积分；
8.  cohesive HoTT、synthetic differential geometry 和 synthetic algebraic geometry；
9.  directed/simplicial type theory 与 directed univalence；
10. two-level type theory、strict equality 和 semisimplicial types；
11. Postnikov towers、Whitehead principles、obstruction theory 和 local coefficients；
12. cofiber sequences、Puppe sequences、Mayer-Vietoris 和谱序列接口；
13. Steenrod operations、Adams spectral sequence、Ext 计算和 stable stems；
14. 集合层代数层级、商结构、局部化、基数、序数和选择原则；
15. classical principles、resizing、choice 和 constructivity/canonicity 边界。

**规则 17.3.** 活跃方向写入正文时必须标注“研究边界”或“外部输入”，并记录来源、日期和基础口径。

## 17.3 版本化阅读

**定义 17.4.** 一个资料条目是版本化的，若它至少记录：

1.  标题、作者和链接；
2.  访问日期、版本号或出版年份；
3.  其被本书使用的具体位置；
4.  它提供的是书内证明背景、外部输入、模型论结果还是研究边界。

**原则 17.5.** 任何“最新”断言在六个月后视为需要重新核查。若涉及正在演化的预印本、讲义或模型论声明，重新核查周期应更短。

## 17.4 当前已落地的高级块

**列表 17.6.** 当前版本不再只把以下内容写成愿景，而是给出精确定理形态、证明核或使用边界：

1.  模态、局部化和连通/截断正交分解：附录 AJ；
2.  Cauchy 实数 HIIT、极限唯一性和完备性接口：附录 AK；
3.  Blakers-Massey、Freudenthal 和 Hopf fibration：附录 AL；
4.  Smash product、pointed symmetric monoidal structure 和 cup product 几何来源：附录 AM；
5.  Directed/simplicial type theory、directed univalence 和高阶范畴接口：附录 AN；
6.  Categorical univalence 分离、strict Rezk completion、interval reversal 和非标准模型边界：附录 AO；
7.  Fiber sequence、connecting map 和同伦群长正合列：附录 AP；
8.  Exact couple、derived couple、谱序列页和条件收敛接口：附录 AQ；
9.  Cauchy 实数的乘法、序、倒数和构造性完备有序域接口：附录 AR；
10. Directed/simplicial 对象语言规则核：附录 AS；
11. Left exact 模态、cohesive HoTT 和 modal induction 边界：附录 AT；
12. Join connectivity、flattening lemma 和 Blakers-Massey 证明架构：附录 AU；
13. Serre、Atiyah-Hirzebruch 和 Adams 谱序列接口：附录 AV；
14. Dedekind 实数、locatedness 与 Cauchy-Dedekind 比较：附录 AW；
15. Directed/simplicial 语义接口：附录 AX；
16. Pushout 路径空间 encode-decode 与 gap map fiber code：附录 AY；
17. 谱、稳定同伦范畴、filtered spectra 和强收敛判据：附录 AZ；
18. 连续性、一致连续性、紧致性、located 中值定理和近似极值定理：附录 BA；
19. Rezk 类型、complete Segal object 和 synthetic $\infty$-category 接口：附录 BB；
20. HIIT、QIT、QIIT、初始代数语义和 cubical HIT 计算边界：附录 BC；
21. Cohesive HoTT、合成微分几何、de Rham 接口和 Zariski 覆盖：附录 BD；
22. Displayed categories、univalent bicategories 和 displayed univalence：附录 BE；
23. Higher groups、deloopings、torsors 和 classifying types：附录 BF；
24. Two-level type theory、strict equality、semisimplicial truncation 和 Reedy fibrant diagrams：附录 BG；
25. 集合层代数层级、商群、商环、ideal 和局部化：附录 BH；
26. 有限集、基数、序数、良基关系和选择原则：附录 BI；
27. Postnikov tower、Whitehead theorem、obstruction class 和 $k$-invariant：附录 BJ；
28. Cofiber、Puppe sequence、cofiber exact sequence 和 Mayer-Vietoris：附录 BK；
29. LEM、DNE、resizing、choice 和 canonicity 边界：附录 BL；
30. 局部系数、twisted EM fibration、twisted Mayer-Vietoris 和 Postnikov 系数系统：附录 BM；
31. Steenrod algebra、Ext、Adams convergence 和低维 Adams 类：附录 BN；
32. 构造性度量空间、级数、Banach 不动点和 Riemann 积分：附录 BO。

**原则 17.7.** 这些块的状态不同：AJ、AP、AQ、AY、AZ、BB、BD、BF、BH、BI、BK、BL、BM、BO 的一部分是书内证明核；BJ 混合证明核、外部输入和 tower 收敛边界；BN 主要是经典外部计算接口；AK、AR、AW、BA 的核心构造依赖 HIIT 输入、locatedness 假设或误差预算；BC、BG 主要是签名与元理论边界；BE 的高阶 bicategory 相干大量依赖外部文献；AL 的 Blakers-Massey、AM 的对称幺半相干依赖外部输入。引用时必须保留这些状态标签。

## 17.5 仍未完成的真实缺口

**路线 17.8.** 若继续把本书推进到更完整的研究生教材，应按以下顺序扩写：

1.  补全第 1-5 章路径代数和等价比较的所有压缩证明；
2.  为第 6-8 章加入单值性、截断和商类型的精细证明；
3.  把第 11 章 $\pi_1(\mathbb S^1)$ 拆成整数、覆盖、encode、decode、群同构五个完整章节；
4.  为第 13-14 章加入更多单值范畴论例子与 Rezk 完备化泛性质细节；
5.  把 Blakers-Massey 的完整路径代数、pushout path code 的 transport 计算和编号 convention 独立成章；
6.  把 Serre、Adams、Atiyah-Hirzebruch 谱序列的局部系数、Steenrod algebra、Ext 计算、典型例子和强收敛验证独立成章；
7.  把实数部分扩展为完整构造性分析章节，包括函数空间、微积分、完备度量空间、级数、积分和可选的测度论接口；
8.  为 Rezk/Segal 高阶范畴展开 semisimplicial 相干、mapping Rezk object、高阶 Yoneda 和与 quasicategory/complete Segal space 的比较；
9.  为 HIIT/QIIT 建立逐签名语法、严格正性、初始性、normalization/canonicity 和语义模型；
10. 为 cohesive/SDG/SAG 建立具体模型、crisp 变量、microlinearity、de Rham 比较、Zariski sheaf/gluing 和代数几何例子；
11. 为 displayed categories 和 univalent bicategories 展开具体实例、coherence、total bicategory univalence 和 monoidal Rezk completion；
12. 为 higher groups 展开一般 $BG$ HIT、torsor 分类、非阿贝尔上同调和 principal bundle 的 functoriality；
13. 为 two-level type theory 建立具体语法、模型、conservativity、exo-nat 假设和 semisimplicial/Rezk object 定义；
14. 为集合层代数展开商结构和局部化的全部 well-defined 证明和泛性质；
15. 为基数和序数展开基数唯一性、基数算术、序数等同性、良基递归和选择原则独立性；
16. 为 Postnikov/obstruction theory 展开 local coefficients、twisted EM types、tower convergence 和 Whitehead principle；
17. 为 cofiber/Mayer-Vietoris 展开 Puppe 自然性、局部系数、具体计算和谱序列衔接；
18. 为 Steenrod/Adams 方向构造 HoTT 内部 cohomology operations、Adams resolution、Ext 计算和 Adams chart convention；
19. 为构造性分析展开级数、积分、微积分基本定理和有理误差预算；
20. 为逻辑原则建立模型独立性、最小假设表和 classical theorem 的构造性替代版本；
21. 为 directed/simplicial type theory 建立完整语法替换定理、soundness/completeness 语义证明。

## 本章小结

完整教材不是一次性罗列所有结果，而是维护一条可审查的证明链。本书当前版本已经把若干研究方向降为定理接口、证明核或精确外部输入；尚未完成的是若干高阶相干的逐行证明、Postnikov/obstruction 的收敛证明、cofiber/MV 的具体计算、Steenrod/Adams 的内部构造、具体谱序列计算、完整分析学体系、集合层代数和大小理论的全部细节、QIIT/cubical HIT 元理论、displayed/bicategory coherence、higher group 分类定理、2LTT 模型、cohesive 几何模型和 directed HoTT 的完整元理论。

## 练习

**练习 17.1.** 选择一个近期 HoTT 论文，按定义 17.4 建立版本化条目。

**练习 17.2.** 解释为什么活跃研究不应作为第一章的基础规则。

**练习 17.3.** 为第十一章圆的基本群证明设计一个五章扩写计划。

**练习 17.4.** 检查本书一个“证明说明”定理，列出把它改为书内证明所需的前置引理。
