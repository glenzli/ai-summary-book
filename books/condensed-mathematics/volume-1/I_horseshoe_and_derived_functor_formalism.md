# 附录 I：Horseshoe 引理与导出函子形式

## I.0 目标

第一卷第八章和第十一章使用投射分解、Ext、Tor、长正合列和维数平移。本附录把这些同调代数事实写成书内证明。环境是有足够多投射对象的阿贝尔范畴 $\mathcal A$。应用时可取

$$
\mathcal A=\mathbf{CondAb}
$$

或

$$
\mathcal A=\mathbf{CondMod}_R.
$$

## I.1 投射分解的比较定理

**定义 I.1.** 对象 $P\in\mathcal A$ 称为投射，如果函子

$$
\operatorname{Hom}_{\mathcal A}(P,-)
$$

保持满射。

**命题 I.2（提升链映射）.** 设 $P_\bullet\to A$ 是投射分解，$Q_\bullet\to B$ 是任意分解，并给定态射 $f:A\to B$。则存在链映射

$$
\tilde f:P_\bullet\to Q_\bullet
$$

提升 $f$。任意两个提升链同伦。

**证明.** 在次数 $0$，有满射 $Q_0\to B$。复合 $P_0\to A\xrightarrow f B$ 可由 $P_0$ 的投射性提升为 $P_0\to Q_0$。假设已构造到次数 $n-1$。记 $Z_{n-1}(Q)=\ker(Q_{n-1}\to Q_{n-2})$。由于 $Q_\bullet\to B$ exact，映射 $Q_n\to Z_{n-1}(Q)$ 满射。链映射条件给出 $P_n\to Z_{n-1}(Q)$，由 $P_n$ 投射性提升到 $Q_n$。归纳得到链映射。

若 $\tilde f,\tilde f'$ 是两个提升，差 $d=\tilde f-\tilde f'$ 提升零映射 $A\to B$。构造同伦 $h_n:P_n\to Q_{n+1}$ 使

$$
d_n=\partial h_n+h_{n-1}\partial
$$

同样逐次使用 $Q_{n+1}\to Z_n(Q)$ 的满射和 $P_n$ 的投射性。证毕。

**推论 I.3.** 用投射分解定义的右导出函子与投射分解选择无关。

**证明.** 任意两个投射分解之间由命题 I.2 得到互相提升恒等映射的链映射，复合链同伦于恒等。对左正合函子取复形后得到同伦等价，故同调同构。证毕。

## I.2 Horseshoe 引理

**定理 I.4（Horseshoe lemma）.** 设

$$
0\to A'\xrightarrow u A\xrightarrow v A''\to0
$$

是 $\mathcal A$ 中短正合列。给定 $A'$ 和 $A''$ 的投射分解

$$
P'_\bullet\to A',
\qquad
P''_\bullet\to A'',
$$

则存在 $A$ 的投射分解 $P_\bullet\to A$，并且可取

$$
P_n=P'_n\oplus P''_n.
$$

这些分解组成短正合链复形

$$
0\to P'_\bullet\to P_\bullet\to P''_\bullet\to0.
$$

**证明.** 在次数 $0$，考虑满射

$$
P'_0\to A',
\qquad
P''_0\to A''.
$$

因为 $P''_0$ 投射，复合 $P''_0\to A''$ 可沿 $A\to A''$ 提升为 $s_0:P''_0\to A$。定义

$$
P_0=P'_0\oplus P''_0\to A,\qquad
(x,y)\mapsto u(x)+s_0(y).
$$

这是满射：任取 $a\in A$，其像 $v(a)\in A''$ 来自某个 $y\in P''_0$；于是 $a-s_0(y)$ 落在 $A'$ 的像中，再由 $P'_0\to A'$ 提升。

令 $K',K,K''$ 分别为 $P'_0\to A'$、$P_0\to A$、$P''_0\to A''$ 的 kernel。由 snake lemma 得短正合列

$$
0\to K'\to K\to K''\to0.
$$

对这条短正合列重复同样构造，并把所得 $P'_1\oplus P''_1\to K$ 复合到 $P_0$，得到第一阶微分。继续归纳。每一步的 exactness 由 snake lemma 和 kernel 定义保证。证毕。

## I.3 长正合列

**引理 I.5（短正合复形给长正合同调列）.** 若

$$
0\to C'^\bullet\to C^\bullet\to C''^\bullet\to0
$$

是阿贝尔群复形的短正合列，则存在自然长正合列

$$
\cdots\to H^n(C')\to H^n(C)\to H^n(C'')
\xrightarrow{\delta}
H^{n+1}(C')\to\cdots.
$$

**证明.** 对 $[z'']\in H^n(C'')$，取提升 $z\in C^n$。因为 $d z''=0$，$dz$ 在 $C^{n+1}$ 中映到 $0$，故来自唯一的 $z'\in C'^{n+1}$。定义

$$
\delta[z'']=[z'].
$$

若改变提升或代表，差值给出边界；因此 $\delta$ 良定义。逐项检查 kernel 等于 image：例如 $[z'']$ 映到 $0$ 当且仅当可取提升 $z$ 为 cocycle，当且仅当它来自 $H^n(C)$。其他位置同理。证毕。

**定理 I.6（Ext 长正合列）.** 对短正合列

$$
0\to B'\to B\to B''\to0
$$

和任意 $A$，有长正合列

$$
0\to\operatorname{Hom}(A,B')\to\operatorname{Hom}(A,B)
\to\operatorname{Hom}(A,B'')
\to\operatorname{Ext}^1(A,B')\to\cdots.
$$

**证明.** 取 $A$ 的投射分解 $P_\bullet\to A$。由于每个 $P_n$ 投射，逐项应用 $\operatorname{Hom}(P_n,-)$ 得到短正合复形

$$
0\to\operatorname{Hom}(P_\bullet,B')
\to\operatorname{Hom}(P_\bullet,B)
\to\operatorname{Hom}(P_\bullet,B'')\to0.
$$

由引理 I.5 取同调，得到长正合列。证毕。

**定理 I.7（第一变量短正合列的 Ext 长正合列）.** 对短正合列

$$
0\to A'\to A\to A''\to0
$$

和任意 $B$，有长正合列

$$
0\to\operatorname{Hom}(A'',B)\to\operatorname{Hom}(A,B)
\to\operatorname{Hom}(A',B)
\to\operatorname{Ext}^1(A'',B)\to\cdots.
$$

**证明.** 用定理 I.4 取三者相容的投射分解，得到短正合链复形

$$
0\to P'_\bullet\to P_\bullet\to P''_\bullet\to0.
$$

应用反变函子 $\operatorname{Hom}(-,B)$ 后得到短正合上链复形

$$
0\to\operatorname{Hom}(P''_\bullet,B)
\to\operatorname{Hom}(P_\bullet,B)
\to\operatorname{Hom}(P'_\bullet,B)\to0.
$$

由引理 I.5 得长正合列。证毕。

## I.4 维数平移

**命题 I.8（dimension shifting）.** 若

$$
0\to K\to P\to A\to0
$$

短正合且 $P$ 投射，则对 $n\ge1$，

$$
\operatorname{Ext}^{n+1}(A,B)
\cong
\operatorname{Ext}^n(K,B).
$$

**证明.** 对短正合列应用定理 I.7。由于 $P$ 投射，

$$
\operatorname{Ext}^m(P,B)=0
$$

对 $m>0$。长正合列中相邻项消失，留下自然同构

$$
\operatorname{Ext}^n(K,B)\cong\operatorname{Ext}^{n+1}(A,B).
$$

证毕。

## I.5 Tor 长正合列

设 $\mathcal A$ 还是带右正合双函子 $\otimes$ 的阿贝尔范畴，并且可用投射分解定义左导出函子 $\operatorname{Tor}_i$。

**定理 I.9.** 对短正合列

$$
0\to A'\to A\to A''\to0
$$

和任意 $N$，有自然长正合列

$$
\cdots\to
\operatorname{Tor}_1(A'',N)
\to A'\otimes N
\to A\otimes N
\to A''\otimes N
\to0.
$$

**证明.** 用 horseshoe lemma 取相容投射分解

$$
0\to P'_\bullet\to P_\bullet\to P''_\bullet\to0.
$$

逐项张量 $N$。由于张量积右正合，不保证短正合逐项保持左端正合；因此更稳妥的做法是在导出范畴中使用三角

$$
A'\to A\to A''\to A'[1].
$$

由左导出函子 $-\otimes^LN$ 的三角性得到三角

$$
A'\otimes^LN\to A\otimes^LN\to A''\otimes^LN\to(A'\otimes^LN)[1].
$$

取同调得到长正合列。末端

$$
A'\otimes N\to A\otimes N\to A''\otimes N\to0
$$

是普通张量积右正合性。证毕。

## I.6 对正文的回填

1. 第八章 Ext 定义的分解独立性由推论 I.3 给出。
2. 第八章命题 8.5 是定理 I.7 和投射对象 Ext 消失的特例。
3. 附录 G 的长正合列和维数平移由定理 I.6-I.8 给出。
4. 第十一章 Tor 长正合列由定理 I.9 给出。

## 练习

**练习 I.1.** 在命题 I.2 中写出链同伦构造的次数 $0$ 和次数 $1$ 步。

**练习 I.2.** 证明 horseshoe lemma 中 kernel 的短正合列来自 snake lemma。

**练习 I.3.** 对两项投射分解 $P_1\to P_0\to A$，用命题 I.8 写出 $\operatorname{Ext}^2(A,B)$ 的维数平移形式。

**练习 I.4.** 解释为什么定理 I.9 的证明不能只说“短正合复形张量后仍短正合”。
