# 附录 G：基本 Ext 与 Tor 计算

## G.0 目标

第八章给出 $\operatorname{Ext}$ 的定义，第十一章给出 $\operatorname{Tor}$ 的定义。本附录补上第一卷所需的基本计算规则。更深的 Scholze 型 $\operatorname{Ext}$ 计算属于第二卷。

## G.1 投射对象上的 Ext 消失

**命题 G.1.** 若 $P\in\mathbf{CondAb}$ 是投射对象，则对任意 $A\in\mathbf{CondAb}$，

$$
\operatorname{Ext}^i_{\mathbf{CondAb}}(P,A)=0,\qquad i>0.
$$

**证明.** $\operatorname{Ext}^i(P,A)$ 是左正合函子

$$
\operatorname{Hom}(P,-)
$$

的右导出函子。若 $P$ 投射，则 $\operatorname{Hom}(P,-)$ 正合，因此高阶右导出函子为零。证毕。

**推论 G.2.** 若 $E$ 是极不连通紧 Hausdorff 空间，则

$$
\operatorname{Ext}^i_{\mathbf{CondAb}}(\mathbb Z[\underline E],A)=0,\qquad i>0.
$$

**证明.** 第七章证明 $\mathbb Z[\underline E]$ 投射。代入命题 G.1。证毕。

## G.2 自由分解与 Ext 复形

设 $M\in\mathbf{CondAb}$。取投射分解

$$
\cdots\to P_2\to P_1\to P_0\to M\to0.
$$

则

$$
\operatorname{Ext}^i(M,A)
=
H^i\operatorname{Hom}(P_\bullet,A),
$$

其中

$$
\operatorname{Hom}(P_\bullet,A):
0\to\operatorname{Hom}(P_0,A)\to\operatorname{Hom}(P_1,A)\to\operatorname{Hom}(P_2,A)\to\cdots.
$$

**命题 G.3.** 若每个 $P_n$ 是若干 $\mathbb Z[\underline{E}]$ 的直和，其中 $E$ 极不连通，则

$$
\operatorname{Hom}(P_n,A)
$$

可由 $A(E)$ 的乘积计算。

**证明.** 由自由对象的泛性质，

$$
\operatorname{Hom}(\mathbb Z[\underline E],A)\cong A(E).
$$

Hom 把直和变为乘积，故若

$$
P_n=\bigoplus_{\alpha\in I_n}\mathbb Z[\underline{E_\alpha}],
$$

则

$$
\operatorname{Hom}(P_n,A)\cong\prod_{\alpha\in I_n}A(E_\alpha).
$$

证毕。

## G.3 Cech 型分解

设 $p:E\to S$ 是 $\mathbf{CHaus}$ 中的满射，且 $E$ 极不连通。记

$$
E^{[n]}=
\underbrace{E\times_S\cdots\times_S E}_{n+1\text{ 个}}.
$$

有增广单纯对象

$$
\cdots\rightrightarrows E^{[2]}\rightrightarrows E^{[1]}\rightrightarrows E\to S.
$$

**命题 G.4.** 若每个 $E^{[n]}$ 可由极不连通空间覆盖并取相应自由分解，则可得到 $\mathbb Z[\underline S]$ 的投射分解。

**证明.** sheaf 条件说明

$$
\underline S\to
\operatorname{Eq}\left(\underline E\rightrightarrows \underline{E\times_S E}\right)
$$

在凝聚集合中为同构。对自由阿贝尔群函子取增广链复形，得到解析 $S$ 的 Cech 复形。若各层进一步用极不连通覆盖替换，则每一项变成投射对象的直和。增广复形的正合性来自 sheaf 的 descent 条件。证毕。

这给出计算原则：

$$
\operatorname{Ext}^i(\mathbb Z[\underline S],A)
$$

可以通过把 $S$ 的极不连通覆盖代入 $A$ 后形成的 Cech 型上同调来计算。

## G.4 Tor 的基本消失

设 $R$ 是凝聚交换环。

**命题 G.5.** 若 $F$ 是平坦凝聚 $R$-模，则

$$
\operatorname{Tor}_i^R(F,M)=0,\qquad i>0.
$$

**证明.** 平坦性表示 $F\otimes_R-$ 正合。因此 $F\otimes_R^L M$ 可由普通张量 $F\otimes_R M$ 表示，高阶同调为零。证毕。

**推论 G.6.** 对极不连通 $E$，自由模

$$
R[\underline E]=R\otimes\mathbb Z[\underline E]
$$

满足

$$
\operatorname{Tor}_i^R(R[\underline E],M)=0,\qquad i>0.
$$

**证明.** 附录 E 证明 $R[\underline E]$ 平坦。代入命题 G.5。证毕。

## G.5 长正合列与维数平移

**命题 G.7.** 短正合列

$$
0\to A'\to A\to A''\to0
$$

诱导长正合列

$$
0\to\operatorname{Hom}(M,A')\to\operatorname{Hom}(M,A)
\to\operatorname{Hom}(M,A'')
\to\operatorname{Ext}^1(M,A')\to\cdots.
$$

**证明.** 对 $M$ 取投射分解 $P_\bullet\to M$。由于 $P_n$ 投射，逐项 Hom 后得到短正合复形

$$
0\to\operatorname{Hom}(P_\bullet,A')
\to\operatorname{Hom}(P_\bullet,A)
\to\operatorname{Hom}(P_\bullet,A'')\to0.
$$

短正合复形给出同调长正合列。证毕。

若

$$
0\to K\to P\to M\to0
$$

且 $P$ 投射，则由长正合列得

$$
\operatorname{Ext}^{i+1}(M,A)\cong\operatorname{Ext}^{i}(K,A),
\qquad i\ge1.
$$

这称为维数平移。

完整的 horseshoe lemma、长正合列和维数平移证明见附录 I。

## G.6 第一卷计算边界

第一卷到此为止能系统处理：

1. 投射对象的 Ext 消失。
2. 极不连通自由对象的 Ext 消失。
3. 由 Cech 型投射分解得到的 Ext 计算原则。
4. 平坦对象的高阶 Tor 消失。
5. 长正合列和维数平移。

Scholze 讲义中的更深 Ext 计算、solid 生成元的派生 Hom 分析、以及 analytic rings 中的 Bousfield localization 计算，放入第二卷。

## 练习

**练习 G.1.** 证明 $\operatorname{Ext}^0(M,A)=\operatorname{Hom}(M,A)$。

**练习 G.2.** 对极不连通 $E$，直接用提升性质证明 $\mathbb Z[\underline E]$ 投射。

**练习 G.3.** 写出两项投射分解

$$
P_1\to P_0\to M\to0
$$

给出的 $\operatorname{Ext}^1(M,A)$ 的 cokernel 表达式。

**练习 G.4.** 证明若 $M$ 有长度为 $n$ 的投射分解，则 $\operatorname{Ext}^i(M,A)=0$ 对 $i>n$ 成立。
