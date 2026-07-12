# 附录 L：高阶归纳类型输入规则表

## 目标

本附录把本书当前实际使用的高阶归纳类型（HIT）规则集中列出。它们是本书的对象语言输入规则，不是本书已证明的元理论存在性定理。若采用公理化 HoTT 阅读，这些规则作为公理化类型构造加入；若采用 cubical type theory 阅读，应逐项核查相应计算规则。

本附录解决的是“本书到底用了哪些 HIT 规则”的精确性问题，不解决“一般 HIT 都存在”的元理论问题。

**宇宙口径。** 本附录沿用第一章的非累积宇宙。出现 $\mathcal U_{\max(i,j)}$ 或 $\mathcal U_{\max(i,j,k)}$ 时，这是相应 HIT 的直接形成规则，不是先把参数累积到共同宇宙。无参数 HIT（例如圆）按 universe-polymorphic 规则在每个层级给出一个实例；省略层级下标不表示 resizing 或不同实例之间的 judgmental equality。

## L.1 命题截断

**输入规则 L.1（命题截断形成）.** 对任意 $A:\mathcal U_i$，有
$$
\|A\|:\mathcal U_i.
$$

**输入规则 L.2（命题截断点构造）.** 有映射
$$
|-|:A\to\|A\|.
$$

**输入规则 L.3（命题性构造）.** 有
$$
\mathsf{trunc}_{-1}:\mathsf{isProp}(\|A\|).
$$

**输入规则 L.4（命题截断递归）.** 若 $P$ 是命题，则预合成
$$
(\|A\|\to P)\to(A\to P)
$$
由命题截断消去原则给出反向延拓。具体地，任意 $f:A\to P$ 延拓为
$$
\bar f:\|A\|\to P
$$
并满足 $\bar f(|a|)=f(a)$。点构造上的计算规则按本书公理化口径为 propositional computation；若具体系统给出 judgmental computation，应单独标注。

## L.2 集合截断与一般截断

**输入规则 L.5（$n$-截断形成）.** 对 $n\ge -2$ 和类型 $A:\mathcal U_i$，有
$$
\|A\|_n:\mathcal U_i
$$
以及映射
$$
|-|_n:A\to\|A\|_n.
$$

**输入规则 L.6（$n$-截断层级）.** 有
$$
\mathsf{isOfHLevel}_{n+2}(\|A\|_n).
$$

**输入规则 L.7（$n$-截断递归泛性质）.** 若 $B$ 具有相应 $n$-type 层级，则从 $\|A\|_n$ 到 $B$ 的函数由从 $A$ 到 $B$ 的函数唯一延拓。完整依赖消去和计算规则视具体 HIT 实现而定。

**说明 L.8.** 第八章只使用命题截断和集合截断的泛性质；一般 $n$-截断作为后续合成同伦论和基本群定义的输入规则保留。

## L.3 集合商

设 $A:\mathcal U_i$，且 $R:A\to A\to\mathcal U_j$ 是 fiberwise 命题值等价关系。

**输入规则 L.9（集合商形成与点构造）.** 有集合商
$$
A/R:\mathcal U_{\max(i,j)}
$$
和点构造
$$
[-]:A\to A/R.
$$

**输入规则 L.10（关系路径构造）.** 对任意 $r:R(x,y)$，有路径
$$
\mathsf{glue}_R(r):[x]=[y].
$$

**输入规则 L.11（集合性构造）.** 有
$$
\mathsf{isSet}(A/R).
$$

**输入规则 L.12（集合商递归）.** 若 $B:\mathcal U_k$ 是集合，$f:A\to B$，并且
$$
h:\prod_{x,y:A}R(x,y)\to(f(x)=f(y)),
$$
则存在
$$
\bar f:A/R\to B
$$
使 $\bar f([x])=f(x)$，且 $\mathsf{ap}_{\bar f}(\mathsf{glue}_R(r))=h(r)$。唯一性由 $B$ 是集合和商的消去原则给出。

**输入规则 L.13（集合商依赖消去）.** 若 $P:A/R\to\mathcal U_k$ 是 fiberwise 集合值或命题值族，并给出点上数据及与关系路径兼容的数据，则得到依赖函数 $\prod_{z:A/R}P(z)$。本书当前不使用完整依赖消去；若后文新增依赖使用，必须在使用点补齐构造子上的 dependent computation。

## L.4 圆

**输入规则 L.14（圆形成与构造）.** 对每个 $i$，有类型 $\mathbb S^1_i:\mathcal U_i$，点
$$
\mathsf{base}_i:\mathbb S^1_i
$$
和路径
$$
\mathsf{loop}_i:\mathsf{base}_i=\mathsf{base}_i.
$$

固定层级后省略下标。

**输入规则 L.15（圆递归）.** 若 $A:\mathcal U_j$，给出
$$
a:A,\qquad \ell:a=a,
$$
则有 $f:\mathbb S^1_i\to A$，满足 judgmental 点计算
$$
f(\mathsf{base})\equiv a
$$
和 propositional 路径计算
$$
\beta_{\mathsf{loop}}:\mathsf{ap}_f(\mathsf{loop})=\ell.
$$
这两种计算强度是本书公理化圆输入的一部分；采用其他系统时不得静默更改。

**输入规则 L.16（圆依赖消去）.** 若 $P:\mathbb S^1_i\to\mathcal U_j$，给出
$$
b:P(\mathsf{base})
$$
和
$$
\ell_P:\mathsf{transport}^{P}(\mathsf{loop},b)=b,
$$
则有
$$
s:\prod_{x:\mathbb S^1_i}P(x)
$$
并满足 judgmental 点计算
$$
s(\mathsf{base})\equiv b
$$
及 propositional dependent computation
$$
\beta^P_{\mathsf{loop}}:
\mathsf{apd}_s(\mathsf{loop})=\ell_P.
$$
由于点计算是 judgmental，两边都具有类型
$$
\mathsf{transport}^{P}(\mathsf{loop},b)=b.
$$

## L.5 悬挂

**输入规则 L.17（悬挂形成与构造）.** 对 $A:\mathcal U_i$，有 $\mathsf{susp}(A):\mathcal U_i$，点
$$
\mathsf{north},\mathsf{south}:\mathsf{susp}(A)
$$
以及路径族
$$
\mathsf{merid}:A\to(\mathsf{north}=\mathsf{south}).
$$

**输入规则 L.18（悬挂递归/消去）.** 对 $B:\mathcal U_j$，定义 $\mathsf{susp}(A)\to B$ 需要给出 $n:B$、$s:B$ 和 $\prod_{a:A}(n=s)$；依赖消去版本把最后一项替换为相应 dependent path 条件。点与路径计算的强度必须按具体使用点声明。

## L.6 Pushout

给定 $A:\mathcal U_i$、$B:\mathcal U_j$、$C:\mathcal U_k$ 以及 $f:A\to B$、$g:A\to C$。

**输入规则 L.19（Pushout 形成与构造）.** 有
$$
\mathsf{pushout}(f,g):\mathcal U_{\max(i,j,k)}
$$
以及构造子
$$
\mathsf{inl}:B\to\mathsf{pushout}(f,g),
$$
$$
\mathsf{inr}:C\to\mathsf{pushout}(f,g),
$$
$$
\mathsf{glue}:\prod_{a:A}\mathsf{inl}(f(a))=\mathsf{inr}(g(a)).
$$

**输入规则 L.20（Pushout 递归）.** 对 $X:\mathcal U_\ell$，要定义 $\mathsf{pushout}(f,g)\to X$，给出 $u:B\to X$、$v:C\to X$ 和同伦
$$
\prod_{a:A}u(f(a))=v(g(a)).
$$
依赖消去版本要求在路径构造子上给出 dependent path 相容性。

## L.7 当前依赖边界

本书第八至十一章目前只依赖 L.1-L.20。一般 HIT 存在性、所有 HIT 的统一语法和元理论、以及 cubical canonicity 不属于本附录证明内容，仍在第十六章和附录 K 中作为外部元理论义务登记。
