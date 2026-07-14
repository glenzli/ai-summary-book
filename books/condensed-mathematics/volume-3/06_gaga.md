# 第六章：Properness、解析化与 derived GAGA

代数仿射直线的全局函数只有多项式，解析直线的全局函数却包含 $e^z$；所以“把复点赋予
解析拓扑”一般会产生更多全纯函数。GAGA 的力量正来自 properness：无穷远被几何地
纳入空间后，代数相干层与解析相干层不但对象相同，上同调也相同。这个结论依赖 Serre
消没、扭转与代数化等深层几何，不能由凝聚范畴的形式性质单独推出。

接受经典或 Clausen--Scholze 的 GAGA 输入后，仍须证明它如何提升到有界导出范畴、
如何与 $R\Gamma$ 和 Euler characteristic 相容。本章完成这些形式步骤，并以
$\mathbb P^1$ 上 $\mathcal O(d)$ 的全局截面比较为 worked example；非 proper
$\mathbb A^1$ 则给出失败条件，而不是抽象警告。

## 6.1 解析化与一个最小反例

若 $X$ 是有限型 $\mathbb C$-scheme，其解析化 $X^{an}$ 的点为 $X(\mathbb C)$，局部
解析结构由代数方程在复解析拓扑中的零点定义。对 coherent sheaf $\mathcal F$，局部
presentation 的矩阵同样可解析化，得到 $\mathcal F^{an}$。

取

$$
X=\mathbb A^1_\mathbb C,
\qquad
X^{an}=\mathbb C.
$$

则

$$
\Gamma(X,\mathcal O_X)=\mathbb C[z],
\qquad
\Gamma(X^{an},\mathcal O_{X^{an}})=\mathcal O(\mathbb C).
$$

**命题 6.1.** 自然映射
$\mathbb C[z]\to\mathcal O(\mathbb C)$ 不是满射。

**证明.** $e^z$ 是整函数。若它是次数 $m$ 的多项式，则沿正实轴满足
$e^x=O(x^m)$；但 $e^x/x^m\to\infty$，矛盾。故 $e^z$ 不在像中。证毕。

失败已经发生在结构层的零次上同调，因此非 proper 情形不可能无条件有相干范畴连同
$R\Gamma$ 的 GAGA 比较。

## 6.2 深层比较输入

**外部输入定理 6.2（Serre GAGA）.** 设 $X$ 是 proper 有限型
$\mathbb C$-scheme。解析化函子给出 exact equivalence

$$
\operatorname{an}:
\operatorname{Coh}(X)
\xrightarrow{\sim}
\operatorname{Coh}(X^{an}),
$$

并且对每个 $\mathcal F\in\operatorname{Coh}(X)$、每个 $i\ge0$，自然映射

$$
H^i(X,\mathcal F)
\xrightarrow{\sim}
H^i(X^{an},\mathcal F^{an})
$$

为同构。

本书不重证 full faithfulness 与 essential surjectivity。射影情形中，它们由 Serre
twisting、上同调比较和有限表示逐步推出；证明结构见
[附录 Y](Y_projective_gaga_proof_architecture.md)。Clausen--Scholze 版本还需把两侧
送入共同的 analytic 派生范畴，精确范围见附录 AR.4。

## 6.3 Exact 等价提升到有界导出等价

**命题 6.3.** 若 $F:\mathcal A\to\mathcal B$ 是阿贝尔范畴间的 exact equivalence，
则逐项作用诱导三角等价

$$
D^b(\mathcal A)\xrightarrow{\sim}D^b(\mathcal B).
$$

**证明.** $F$ 逐项作用于有界复形并与微分相容。exactness 给

$$
H^q(F(C^\bullet))\cong F(H^q(C^\bullet)),
$$

所以 $C^\bullet$ acyclic 当且仅当 $F(C^\bullet)$ acyclic，且 $F$ 保持并反映
quasi-isomorphism。因此它下降到把 quasi-isomorphism 局部化所得的导出范畴。取
$F$ 的 exact quasi-inverse $G$；逐项作用的 $G$ 同样下降，原来的自然同构
$GF\cong\operatorname{id}$、$FG\cong\operatorname{id}$ 在导出范畴中仍成立。证毕。

对输入定理 6.2 应用命题 6.3，得到

$$
D^b(\operatorname{Coh}(X))
\simeq
D^b(\operatorname{Coh}(X^{an})).
$$

在标准 Noetherian 有限维假设下，这与通常记号
$D^b_{\operatorname{coh}}(X)$、$D^b_{\operatorname{coh}}(X^{an})$ 相容。

## 6.4 从逐层上同调到 $R\Gamma$ 比较

**命题 6.4.** 输入定理 6.2 的上同调同构等价于对每个 coherent sheaf 的
quasi-isomorphism

$$
R\Gamma(X,\mathcal F)
\xrightarrow{\sim}
R\Gamma(X^{an},\mathcal F^{an}).
$$

并且逐项解析化给出的比较自然扩张到
$E\in D^b(\operatorname{Coh}(X))$。

**证明.** 对置于次数零的 $\mathcal F$，比较态射在第 $i$ 个 cohomology 上正是输入
定理 6.2 的映射；复形态射是 quasi-isomorphism 当且仅当所有 cohomology 映射为同构，
故第一句成立。

令 $\mathcal T$ 为使比较态射为同构的 bounded derived 对象所成全子范畴。
$R\Gamma$ 和解析化都是三角函子，所以 $\mathcal T$ 对 shift 封闭，并由三角的
two-out-of-three 对 cone 封闭。它包含每个 coherent sheaf。任意 bounded complex 可由
stupid truncation 的有限过滤从各项的 shift 经有限次 cone 构造，故全部对象都在
$\mathcal T$ 中。证毕。

**推论 6.5.** 对 $E\in D^b(\operatorname{Coh}(X))$，

$$
\chi(X,E)=\chi(X^{an},E^{an}).
$$

**证明.** proper coherent finiteness 保证两个交错和有限。命题 6.4 给出同构的导出
全局截面复形，所以各 cohomology 维数逐项相等。证毕。

## 6.5 Worked example：$\mathbb P^1$ 上的解析截面必为多项式

取 $d\in\mathbb Z$。代数侧的 $\mathcal O(d)$ 截面由齐次次数 $d$ 多项式给出；在
$z=X_1/X_0$ 坐标中，$d\ge0$ 时是次数不超过 $d$ 的多项式，$d<0$ 时为零。

解析侧一个全局截面由整函数 $f_0(z)$、$f_\infty(w)$ 满足

$$
f_0(z)=z^df_\infty(1/z)
$$

给出。若 $d\ge0$，函数 $w^df_0(1/w)=f_\infty(w)$ 在 $w=0$ 全纯，故 $f_0$ 在
无穷远至多有 $d$ 阶极点。于是 $f_0$ 是次数不超过 $d$ 的多项式：从 Laurent 展开看，
$w^df_0(1/w)$ 无负幂恰好排除了 $f_0$ 中次数大于 $d$ 的项。若 $d=-k<0$，则
$f_\infty(w)=w^{-k}f_0(1/w)$ 在 $w=0$ 全纯，故 $f_0(z)=O(|z|^{-k})$ 当
$|z|\to\infty$。于是 $f_0$ 有界；Liouville 定理说明它为常数，而趋于零又迫使该常数
为零。因此 $f_0=0$。

因此自然映射

$$
H^0(\mathbb P^1,\mathcal O(d))
\longrightarrow
H^0((\mathbb P^1)^{an},\mathcal O(d)^{an})
$$

在两侧都给出

$$
\begin{cases}
\mathbb C[z]_{\le d},&d\ge0,\\
0,&d<0.
\end{cases}
$$

输入是转移函数与整性，步骤是在无穷远检查 Laurent 指数，输出是显式同构。高阶
上同调比较可由第四章的两图 Čech 计算直接核验，也由输入定理 6.2 统一保证。

## 6.6 共同 analytic 目标中的形式后果

**外部输入定理 6.6（Clausen--Scholze GAGA 建模）.** 在附录 AR.4 的适用范围内，
代数与解析相干对象进入共同的 analytic 派生范畴，且解析化等价、$R\Gamma$、proper
pushforward、trace 与 duality 相容。

接受该输入后，命题 6.3--6.5 仍然适用：它们只使用 exactness、有限 cone 与
$R\Gamma$ 比较。特别地，analytic 增强不会改变 Euler characteristic，也不会把
命题 6.1 的非 proper 失败消除；properness 仍是阻止额外整函数从无穷远进入的几何
条件。

## 6.7 比较之后只剩一个整数吗

GAGA 说明代数和解析两侧计算同一个 perfect $R\Gamma$，所以其 Euler characteristic
相同。Riemann--Roch 更进一步：它不用逐个求上同调，而把这个整数表示为
$\operatorname{ch}(E)\operatorname{td}(T_X)$ 的顶次积分。第七章将明确哪些特征类
性质是外部输入，并完整证明 $K$-理论可加性、到点映射的 HRR 推导和
$\mathbb P^1$ 上的数值核验。

## 练习

**练习 6.1.** 用整函数 $\sin z$ 代替 $e^z$，给出命题 6.1 的另一反例，并说明为何
“非多项式”已经足够。

**练习 6.2.** 对三项有界复形写出 stupid truncation 三角，并用它演示命题 6.4 的
有限 cone 归纳。

**练习 6.3.** 对 $d=2$ 在例 6.5 中从 $f_\infty(w)=w^2f_0(1/w)$ 的全纯性逐项推出
$f_0=a_0+a_1z+a_2z^2$。
