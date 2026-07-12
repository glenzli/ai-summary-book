# 第五章：Liquid 向量空间入口

## 本章目标

本章精确说明三件事：\(<p\)-测度如何出现，什么叫 \(p\)-liquid，以及经典
Banach/Fréchet 空间如何通过凝聚化进入 liquid 范畴。特别要纠正一个常见误界定：
Banach 或 Fréchet 空间并不需要一个额外且未定义的“realization object”；它的候选对象
就是 \(S\mapsto\operatorname{Cont}(S,E)\)。深定理在于该凝聚对象确实
\(p\)-liquid，而 exactness 仍需单独检查。

## 依赖

需要第三、四章的 analytic ring、解析化与内部 Hom 约定。

## 5.1 \(<p\)-测度对象

固定 \(0<p\le1\)。若 \(F\) 是有限集，对
\(x=(x_s)_{s\in F}\in\mathbb R^{\oplus F}\) 记

$$
\|x\|_q=\sum_{s\in F}|x_s|^q,
\qquad 0<q\le1.
$$

这里没有取 \(q\) 次方根；因此三角不等式来自
\(|a+b|^q\le |a|^q+|b|^q\)。对
\(S=\varprojlim_iS_i\in\mathbf{ProFin}_\kappa\)，定义公式

$$
\mathcal M_q(S)_{\le C}
=\varprojlim_i
\{x\in\mathbb R^{\oplus S_i}\mid \|x\|_q\le C\},
$$

并在凝聚集合中取滤过并

$$
\mathcal M_q[S]=\bigcup_{C>0}\mathcal M_q(S)_{\le C},
\qquad
\mathcal M_{<p}[S]=\bigcup_{0<q<p}\mathcal M_q[S].
$$

有限层上的单位坐标向量给出自然 Dirac 映射
\(\underline S\to\mathcal M_{<p}[S]\)。上述逆极限、滤过并及其凝聚
\(\mathbb R\)-模结构的完整构造是下面输入定理的一部分；公式同时固定本书的
\(<p\) 约定。

**反例边界 5.1.** 不能把 \(\mathcal M_{<p}\) 换成普通 bounded Radon measures
\(\mathcal M_1\)，也不能只取单个 \(\mathcal M_p\)。S26 Example 7.10 说明
\(\mathcal M_1\) 不 analytic；CS26 Lecture III 展示固定 \(p\) 时仍存在只在所有
\(q<p\) 上消失的非零映射。\(<p\) 的并是定义的一部分，不是可忽略的近似记号。

## 5.2 \(p\)-liquid 的定义与结构定理

**定义 5.2（CS26 Definition 2.13）.** 凝聚阿贝尔群 \(V\) 称为
\(p\)-liquid，如果对每个 \(S\in\mathbf{ProFin}_\kappa\) 及每个凝聚集合态射
\(f:\underline S\to U(V)\)，存在唯一凝聚阿贝尔群态射

$$
\widetilde f:\mathcal M_{<p}[S]\longrightarrow V
$$

延拓 \(f\)。记满子范畴为 \(\mathbf{Liquid}_p\)。

**外部输入定理 5.3（liquid 主定理）.** 对 \(0<p\le1\)：

1. \((\underline{\mathbb R},\mathcal M_{<p})\) 是 analytic ring；
2. 定义 5.2 等价于解析模 Hom 判别
   $$
   \operatorname{Hom}_{\mathbb R}(\mathcal M_{<p}[S],V)
   \xrightarrow{\sim}V(S);
   $$
3. 每个 \(p\)-liquid 凝聚阿贝尔群有唯一且函子性的凝聚
   \(\mathbb R\)-模结构；
4. \(\mathbf{Liquid}_p\) 是对所有极限、余极限、扩张和内部 Hom 封闭的阿贝尔
   子范畴，\(\mathcal M_{<p}[S]\) 是自由紧投射生成元；
5. \(D(\mathbf{Liquid}_p)\to D(\mathbf{CondAb})\) 全忠实，并由 cohomology
   objects 检测；liquidification 左伴随及 liquid tensor 存在。

**来源与外部边界.** Analytic ring 断言见 S26 Theorem 7.11；等价刻画和全部范畴
结构见 CS26 Theorem 3.11。第二卷只证明接受该输入后的形式推论，不重做
\(<p\)-测度的核心分析。

派生 \(p\)-liquid 复形因此可精确写成满足

$$
R\underline{\operatorname{Hom}}_{\mathbb R}
(\mathcal M_{<p}[S],C)
\xrightarrow{\sim}
R\underline{\operatorname{Hom}}_{\mathbb R}
(\mathbb R[\underline S],C)
$$

的 \(C\in D(\mathbf{Cond}(\mathbb R))\)。

## 5.3 Banach 与 Fréchet 对象确实 liquid

**定义 5.4.** 一个 \(p\)-Banach 空间是完备实向量空间 \(E\)，配备函数
\(\|\cdot\|:E\to\mathbb R_{\ge0}\)，满足正定性、三角不等式及

$$
\|\lambda x\|=|\lambda|^p\|x\|.
$$

普通 Banach 范数的 \(p\) 次方给出任意 \(0<p\le1\) 的 \(p\)-Banach 结构，
且不改变拓扑。

**外部输入定理 5.5（经典空间的 liquid 判别）.** 每个 \(p\)-Banach 空间 \(E\)
的关联凝聚模

$$
\underline E(S)=\operatorname{Cont}(S,E)
$$

是 \(p\)-liquid。任意 \(p\)-liquid 对象的逆极限仍 \(p\)-liquid；因此任意
complete locally \(p\)-convex 实拓扑向量空间的关联凝聚模都是 \(p\)-liquid。

**来源与边界.** CS26 Theorem 2.14 给出 quasiseparated 对象的 \(q\)-convex
判别，Lemma 2.16 控制 \(p\)-Banach 空间的紧子集；其后的推论给出逆极限结论。
该定理判断的是对象是否 liquid，不声称拓扑短正合列都会变成 liquid 短正合列。

**推论 5.6（Fréchet 空间）.** 每个实 Fréchet 空间 \(E\) 的关联凝聚模对所有
\(0<p\le1\) 都是 \(p\)-liquid。

**证明.** 取定义 Fréchet 拓扑的递增可数半范数族 \((r_n)\)，令 \(E_n\) 为
\(E/\ker r_n\) 对 \(r_n\) 的 Banach 完备化。完备性与 Hausdorff 性给出拓扑向量空间
同构

$$
E\cong\varprojlim_n E_n.
$$

每个普通 Banach 空间 \(E_n\) 对固定 \(p\) 都是 \(p\)-Banach；由输入定理 5.5，
\(\underline E_n\) \(p\)-liquid。凝聚化按定义逐测试对象保持该逆极限，故
\(\underline E\cong\varprojlim_n\underline E_n\) 也 \(p\)-liquid。证毕。

## 5.4 本书的 realization 记号

**定义 5.7.** 令 \(\mathbf{TVS}^{\mathrm{liq}}_p\) 为满足以下条件的 Hausdorff 实
拓扑向量空间及连续线性映射的范畴：对象 \(E\) 是
\(\kappa\)-紧生成的，且 \(\underline E\) 是 \(p\)-liquid。定义

$$
\mathcal L_p:\mathbf{TVS}^{\mathrm{liq}}_p\longrightarrow\mathbf{Liquid}_p,
\qquad
\mathcal L_p(E):=\underline E.
$$

这只是“凝聚化后识别为 liquid”的记号，不是另一个未说明构造的 functor。由第一卷
输入定理 3.16，\(\mathcal L_p\) 在 \(\kappa\)-紧生成对象上全忠实；Banach 和
Fréchet 空间因可度量而属于其定义域。

## 5.5 Exactness 的局部提升边界

本节通过第一卷第五章的站点比较，把凝聚对象计算在
\(\mathbf{ProFin}_\kappa\) 的有限联合满射站点上。第一卷附录 A.3 保证每个
\(\kappa\)-小 compact Hausdorff 对象有同层级的 extremally disconnected（因而
profinite）满射覆盖。因此下面只量化 profinite 测试对象，等价于在原
\(\mathbf{CHaus}_\kappa\) 站点检查 sheaf epimorphism；这一步不能脱离站点比较直接
假设。

**定义 5.8.** 连续满射线性映射 \(q:E\twoheadrightarrow F\) 称为
\(\kappa\)-凝聚有效的，如果对每个 \(S\in\mathbf{ProFin}_\kappa\) 和连续映射
\(f:S\to F\)，存在有限联合满射覆盖 \(\{S_i\to S\}\)，使每个
\(f|_{S_i}\) 都有连续提升 \(S_i\to E\)。

**命题 5.9（cokernel 的精确判别）.** 设

$$
0\longrightarrow E'\xrightarrow{i}E\xrightarrow{q}E''\longrightarrow0
$$

是底层实向量空间正合的连续线性映射列，\(i\) 把 \(E'\) 同胚到
\(\ker q\)。则凝聚列

$$
0\longrightarrow\underline E'\longrightarrow\underline E
\longrightarrow\underline E''\longrightarrow0
$$

正合，当且仅当 \(q\) 是 \(\kappa\)-凝聚有效的。若 \(q\) 有连续（不必线性）截面，
则该条件成立。

**证明.** 凝聚化逐对象保持 kernel，所以左端和中间正合由
\(E'\cong\ker q\) 给出。Sheaf 范畴中 \(\underline q\) 为 epimorphism，当且仅当
每个测试对象上的截面在某个覆盖后可提升；由本节开头的站点比较，可只在
\(\mathbf{ProFin}_\kappa\) 上检查，这正是定义 5.8。若有连续截面
\(s:E''\to E\)，则 \(s\circ f\) 给出不需覆盖的全局提升。证毕。

**边界 5.10.** “\(\operatorname{im}d\) 闭”只保证拓扑 quotient 为 Hausdorff
Fréchet 空间；它本身不等于定义 5.8 的局部提升条件。后文比较
\(H^q(\mathcal L_p(E^\bullet))\) 与拓扑 cohomology 时，必须验证相关 quotient
凝聚有效，或给出连续 splitting。这个义务不能隐藏在“realization exact”一句中。

## 5.6 Solid、analytic、liquid 的接口

| 层次 | 底环与测度 | 局部对象 | 不能混同的操作 |
| --- | --- | --- | --- |
| solid | \((\mathbb Z,\mathbb Z^\square)\) | solid 阿贝尔群/复形 | ordinary completion |
| analytic | 一般 \((A,\mathcal M)\) | \(D(A,\mathcal M)\) | 任意 cone localization |
| \(p\)-liquid | \((\underline{\mathbb R},\mathcal M_{<p})\) | \(\mathbf{Liquid}_p\) | Banach completion 或 \(\mathcal M_p\) |

从一个层次变换到底环不同的另一层次，需要给出 analytic rings 的态射及相对 analytic
tensor；仅有抽象环映射 \(\mathbb Z\to\mathbb R\) 不足以自动识别测度对象和完成化。

## 5.7 本章小结

\(p\)-liquid 的定义是 \(\mathcal M_{<p}[S]\) 上的唯一延拓。Banach 与 Fréchet
空间的凝聚化确实给出 liquid 对象；真正需要额外假设的是 cokernel 与 cohomology 的
exactness，命题 5.9 把它化为可检查的局部提升条件。

## 练习

**练习 5.1.** 对有限 \(S\) 写出 \(\mathcal M_q(S)_{\le C}\) 并验证 Dirac 向量
属于其中某个球。

**练习 5.2.** 写出定义 5.2 与 analytic 模 Hom 判别之间的伴随变换。

**练习 5.3.** 证明有连续截面的满射满足定义 5.8。

**练习 5.4.** 指出闭像假设与凝聚有效性在命题 5.9 中分别控制什么。

**练习 5.5.** 解释为什么从 solid 到 liquid 的 scalar extension 必须包含测度相容数据。
