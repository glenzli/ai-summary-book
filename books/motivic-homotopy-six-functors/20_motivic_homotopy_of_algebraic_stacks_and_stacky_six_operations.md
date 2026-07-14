# 第二十章：Algebraic stacks 上的 motivic homotopy 与六操作

模空间通常不是概形：对象可以有自同构，局部坐标来自 smooth atlas，稳定子群还会
作用在切向与法向方向上。因此，把 `S\mapsto\mathbf{SH}(S)` 延拓到 algebraic
stacks，不能只把“概形”替换成“栈”。定义域必须对局部商表示和 Nisnevich 粘合足够
稳定；六操作中每一个伴随的适用态射也必须单独说明。

本章采用 Khan--Ravi 的 scalloped derived stacks 口径。第十九章的 quotient stacks
构成它的局部模型，而 scallop decomposition 提供从这些模型到一般栈的有限归纳。
这一 genuine 理论还要与任意栈上都能形式定义的 lisse extension 区分；二者之间虽有
自然函子，却通常不是等价。

## 20.1 Scallop 分解

下面的“nice embeddable group”沿用 Khan--Ravi 的专门定义；它不是“任意线性约化
群”的同义词。该限制保证局部商具有所需的 resolution property，并使 Thom twist
能够由向量丛的 `K`-理论类控制。

**定义 20.1.** 设 `\mathcal X` 是 qcqs derived algebraic stack。若存在仿射概形
`S`、`S` 上的 nice embeddable group scheme `G`，以及 quasi-affine 态射

$$
\mathcal X\longrightarrow BG,
$$

则称 `\mathcal X` 为 quasi-fundamental。等价地，`\mathcal X` 可写成
`[X/G]`，其中 `X` 是 `S` 上带 `G`-作用的 quasi-affine derived scheme。

**定义 20.2.** 一个可表 Nisnevich 方块是 Cartesian 方块

$$
\begin{array}{ccc}
\mathcal W&\longrightarrow&\mathcal V\\
\downarrow&&\downarrow p\\
\mathcal U&\xrightarrow{j}&\mathcal X
\end{array}
$$

其中 `j` 是开浸入，`p` 是有限表现的可表 etale 态射，并且
`\mathcal V\setminus\mathcal W\to\mathcal X\setminus\mathcal U` 是同构。这里的
补集取相应的闭子栈；这个条件正是“etale 邻域在开集之外不增加新点”。

**定义 20.3.** `\mathcal X` 的一个 scallop decomposition 是有限的拟紧开子栈过滤

$$
\varnothing=\mathcal U_0\subset\mathcal U_1\subset\cdots
\subset\mathcal U_n=\mathcal X,
$$

以及对每个 `i` 的可表 Nisnevich 方块

$$
\begin{array}{ccc}
\mathcal W_i&\longrightarrow&\mathcal V_i\\
\downarrow&&\downarrow u_i\\
\mathcal U_{i-1}&\longrightarrow&\mathcal U_i
\end{array}
$$

其中 `\mathcal V_i` 是 quasi-fundamental。若 `\mathcal X` 还有分离对角，则称它为
scalloped stack。

**命题 20.4（有限粘合归纳）.** 设 `F` 与 `G` 是 scalloped stacks 上满足可表
Nisnevich 下降的范畴值理论，且 `\eta:F\to G` 是自然变换。若对所有
quasi-fundamental stack `\mathcal V`，`\eta_{\mathcal V}` 都是等价，则
`\eta_{\mathcal X}` 对每个 scalloped stack `\mathcal X` 都是等价。

**证明.** 对 scallop decomposition 的长度归纳。长度为零时对象为空，结论由下降的
空覆盖条件得到。设结论已知于 `\mathcal U_{i-1}`。定义 20.2 的方块给出两个
homotopy pullback 方块

$$
F(\mathcal U_i)\simeq
F(\mathcal U_{i-1})\times_{F(\mathcal W_i)}F(\mathcal V_i),
$$

以及对 `G` 的同一公式。`\mathcal V_i` 是 quasi-fundamental，而
`\mathcal W_i=\mathcal U_{i-1}\times_{\mathcal U_i}\mathcal V_i` 是
`\mathcal V_i` 的开子栈，故仍为 quasi-fundamental。因此右端三项上的 `\eta`
均为等价；极限保持等价，故 `\eta_{\mathcal U_i}` 为等价。取 `i=n` 即得结论。
`\square`

这个命题说明 scallop 不是装饰性的分层：它把一般栈上的比较定理化成有限次
Nisnevich 粘合问题。

## 20.2 稳定 motivic 范畴及六操作的定义域

**外部输入定理 20.5（Khan--Ravi）.** 对每个 scalloped derived stack
`\mathcal X`，存在 presentable stable symmetric monoidal infinity-category
`\mathbf{SH}(\mathcal X)`。赋值 `\mathcal X\mapsto\mathbf{SH}(\mathcal X)` 满足
可表 Nisnevich 下降、向量丛同伦不变性和闭开局部化。其精确构造与这些性质见
Khan--Ravi, Definitions 2.7、2.9，Theorems 4.5、4.10 与 Example 5.12。

**外部输入定理 20.6（六操作）.** 在定理 20.5 的定义域内：

1. 对每个态射 `f:\mathcal X\to\mathcal Y`，有伴随
   `f^*:\mathbf{SH}(\mathcal Y)\rightleftarrows
   \mathbf{SH}(\mathcal X):f_*`，且每个纤维有
   `(\otimes,\underline{\operatorname{Hom}})`；
2. 只有当 `f` **可表且有限型** 时，原定理才无条件给出 exceptional 伴随
   `f_!\dashv f^!`；
3. 这些函子满足相应的 exceptional base change、projection formula、开浸入
   `j_!\simeq j_\sharp`、可表 proper 态射 `f_!\simeq f_*` 与闭开局部化；
4. 若可表 smooth 态射 `f` 可紧化，或源与靶有仿射对角，则有纯性等价

$$
f^!\simeq f^*\langle L_{\mathcal X/\mathcal Y}\rangle,
\qquad
f_!\simeq f_\sharp\langle-L_{\mathcal X/\mathcal Y}\rangle.
$$

这里 `\langle\alpha\rangle` 是 `K`-理论类 `\alpha` 所确定的 Thom twist。前三项见
Theorem 7.1，smooth purity 见 Theorem 7.10。对非可表态射写出 `f_!` 或 `f^!`
需要另一项定理，不能由本定理补出。

**推论 20.7.** 将 qcqs algebraic spaces，特别是满足同样有限性条件的 schemes，
视为 scalloped stacks，定理 20.6 恢复概形上的相应六操作。

**证明.** 这些对象嵌入 scalloped stacks 的定义域；Khan--Ravi 的构造在 algebraic
spaces 上限制为原 stable motivic homotopy theory。操作的比较属于定理 20.5 的
扩张陈述，不是仅由全嵌入形式推出的。`\square`

## 20.3 Genuine 理论与 lisse extension

**定义 20.8.** 对任意 qcqs algebraic stack `\mathcal X`，令
`\operatorname{Lis}_{\mathcal X}` 的对象为 smooth 态射
`u:U\to\mathcal X`，其中 `U` 是 qcqs algebraic space。lisse-extended 范畴定义为

$$
\mathbf{SH}^{\triangleleft}(\mathcal X)
=\varprojlim_{(U,u)\in\operatorname{Lis}_{\mathcal X}}\mathbf{SH}(U),
$$

过渡函子取 inverse image。这个定义给出 lisse site 上的相容族；它并不等于先任选
一个 atlas 再把有限层 Cech 神经截断。

**外部输入定理 20.9.** 若 `\mathcal X` scalloped，则有保持余极限与 inverse image 的
自然函子

$$
\mathbf{SH}(\mathcal X)\longrightarrow
\mathbf{SH}^{\triangleleft}(\mathcal X).
$$

该函子一般远非等价。对 quotient stacks，lisse-extended cohomology 与代数 Borel
近似相连；例如在相应特征和光滑拟射影假设下，lisse-extended motivic cohomology
计算 Edidin--Graham equivariant higher Chow groups。见 Khan--Ravi §12，尤其
Theorems 12.9、12.15 与 Example 12.21。

**例子 20.10（`BG`）.** 取非平凡 nice group `G`。Genuine
`\mathbf{SH}(BG)` 记录 `G`-表示所给的 Thom spheres、稳定子与 genuine transfers；
`\mathbf{SH}^{\triangleleft}(BG)` 则是 Borel 型扩展。两者的 `K`-理论比较在
`\pi_0` 上可表现为从未完备对象到完备化对象的映射，因而不能把 lisse extension
当作 genuine 理论的另一个写法。粗模空间若只是基 `S`，还会进一步丢掉 `G` 这一
自同构群，所以 `\mathbf{SH}(BG)` 也不由 `\mathbf{SH}(S)` 决定。

## 20.4 局部化、纯性与稳定子表示

设 `i:\mathcal Z\hookrightarrow\mathcal X` 为闭浸入，`j:\mathcal U\hookrightarrow
\mathcal X` 为拟紧开补。定理 20.6 给出 cofiber sequence

$$
j_!j^*E\longrightarrow E\longrightarrow i_*i^*E.
$$

当 `\mathcal X=[X/G]` 且 `\mathcal Z=[Z/G]` 时，法丛是带 `G`-线性化的
`N_{Z/X}`。相应 Thom twist 必须保留每个稳定子在法纤维上的表示；若只取粗空间的
普通法丛，固定点公式中的权重与 Euler 类都会消失。

**命题 20.11.** 若一个比较函子保持 inverse image、闭开局部化和 equivariant
Thom twists，并且它在某个 scallop decomposition 的所有
`\mathcal V_i` 上为等价，则它在 `\mathcal X` 上为等价。

**证明.** Thom twist 条件保证局部比较没有忘记稳定子在向量丛方向上的作用；等价性
本身再由命题 20.4 的有限 Nisnevich 归纳得到。`\square`

## 20.5 Torus concentration

固定连通 Noetherian 仿射基 `S` 和 split torus
`T=\mathbb G_{m,S}^{\times l}`。设 `L` 是 `B\mathbb G_m` 上的 tautological line
bundle。对 `F\in\mathbf{SH}(BT)`，令 `S_F` 为各投影拉回的 Euler 类

$$
\operatorname{pr}_i^*e(L^{\otimes n}),
\qquad 1\leq i\leq l,\ n\geq1
$$

所生成的乘法系，并用下标 `\mathrm{loc}` 表示沿 `S_F` 的导出局部化。

**外部输入定理 20.12（concentration）.** 设 `i:Z\hookrightarrow X` 是有限型
`T`-equivariant derived algebraic spaces 的闭浸入，并且 `T` 在 `X\setminus Z`
上无固定点。则对每个 `F\in\mathbf{SH}(BT)`，Borel--Moore homology 的 proper
pushforward 在上述 Euler 类局部化后成为等价：

$$
i_*:C^{\mathrm{BM}}_\bullet([Z/T]/BT,F)_{\mathrm{loc}}
\xrightarrow{\ \simeq\ }
C^{\mathrm{BM}}_\bullet([X/T]/BT,F)_{\mathrm{loc}}.
$$

若 `X` 还分离且有限型，可取 `Z=X^T`。这是 Khan--Ravi Theorem 11.2 与
Corollary 11.3。未局部化的 `i_*` 不在定理结论中；“固定点决定全空间”必须连同
被反演的 Euler 类一起陈述。

**例子 20.13.** 令 `T=\mathbb G_m` 以正权 `r>0` 作用在
`X=\mathbb A^1` 上，固定点为原点。开补 `\mathbb G_m` 没有固定点，定理 20.12
说明包含映射 `\{0\}\hookrightarrow\mathbb A^1` 在反演
`e(L^{\otimes r})` 后诱导 Borel--Moore homology 等价。这个例子也说明 Euler
局部化不是技术冗余：被删除的开轨道正由相应权重的 Euler 类杀掉。

## 20.6 从局部商到一般栈

Scallop decomposition 把 quotient-stack 计算、Nisnevich 下降和有限归纳组合在一起；
定理 20.6 则清楚地区分任意态射上的 `f^*\dashv f_*` 与可表有限型态射上的
`f_!\dashv f^!`。Lisse extension 提供另一种适用于任意栈的 Borel 型理论，却不能
取代 genuine 范畴。由此，栈上的 motivic homotopy 不是“选 atlas 后逐项计算”的
简写，而是一套同时记住下降、稳定子表示和 exceptional functor 定义域的理论。

## 练习

**练习 20.1.** 写出 scallop decomposition 中第 `i` 个 Nisnevich 方块，并说明
`\mathcal W_i` 的几何意义。

**练习 20.2.** 证明命题 20.4 中长度为一的情形。

**练习 20.3.** 对照定理 20.6，说明为什么非可表态射上的 `f_!` 不能在本章中默认存在。

**练习 20.4.** 写出 `\mathbf{SH}^{\triangleleft}(\mathcal X)` 的指标范畴与过渡函子。

**练习 20.5.** 用 `BG` 解释 genuine theory 与 lisse extension 保存的信息有何不同。

**练习 20.6.** 在 `[X/G]` 中说明法丛的稳定子表示为何进入 purity twist。

**练习 20.7.** 在定理 20.12 中找出基、群、态射和系数的全部假设。

**练习 20.8.** 对权 `r` 的 `\mathbb G_m`-作用计算 `\mathbb A^1` 的固定点，并指出
concentration 所反演的 Euler 类。

**练习 20.9.** 解释“genuine 与 lisse-extended 的自然比较存在”为什么不蕴含二者等价。
