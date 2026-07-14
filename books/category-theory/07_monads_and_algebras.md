# 第七章：单子、余单子与代数

每个伴随 $F\dashv U$ 都在源范畴上产生端函子 $T=UF$，而单位与余单位迫使 $T$ 带有乘法和单位；这正是单子。单子代数把“由自由对象生成并满足结构律”的对象抽象出来，Kleisli 范畴突出自由计算，Eilenberg--Moore 范畴则收集全部代数。两者揭示了同一个单子如何保留伴随的不同部分。本章从伴随构造单子，证明比较函子的基本性质，并以 Beck 单子性定理说明何时一个右伴随可以完全由其单子恢复。

读者只需前四章的伴随、自然变换与函子复合。Beck 定理需要的分裂余等化子条件会精确写出，并作为外部输入使用；不会把“有一个单子”无条件等同于“原范畴就是代数范畴”。

## 7.1 单子的定义

**定义 7.1.** 范畴 $\mathcal C$ 上的单子（monad）是三元组 $(T,\eta,\mu)$，其中

$$
T:\mathcal C\to\mathcal C
$$

是函子，

$$
\eta:\operatorname{id}_{\mathcal C}\Rightarrow T,\qquad
\mu:T^2\Rightarrow T
$$

是自然变换，满足单位律和结合律：

$$
\mu\circ T\eta=\operatorname{id}_T,\qquad
\mu\circ \eta T=\operatorname{id}_T,
$$

以及

$$
\mu\circ T\mu=\mu\circ\mu T:T^3\Rightarrow T.
$$

**例子 7.2.** 在 $\mathbf{Set}$ 上，自由幺半群函子 $T(S)$ 取 $S$ 上有限字。单位把元素送到长度一的字，乘法 $\mu:T^2(S)\to T(S)$ 把“字的字”拼接成一个字。单位律和结合律分别来自空层拼接无效和拼接结合律。

## 7.2 伴随产生单子

**命题 7.3.** 若 $F:\mathcal C\rightleftarrows\mathcal D:G$ 且 $F\dashv G$，单位为 $\eta$、余单位为 $\varepsilon$，则

$$
T=GF:\mathcal C\to\mathcal C
$$

带有单位 $\eta:\operatorname{id}_{\mathcal C}\to GF$ 和乘法

$$
\mu=G\varepsilon F:GFGF\to GF
$$

构成单子。

**证明.** 单位律为

$$
G\varepsilon F\circ GF\eta=\operatorname{id}_{GF},
\qquad
G\varepsilon F\circ\eta GF=\operatorname{id}_{GF},
$$

它们分别是伴随三角恒等式在 $F$ 或 $G$ 后的函子像。结合律要求

$$
G\varepsilon F\circ GF(G\varepsilon F)
=
G\varepsilon F\circ G\varepsilon FGF.
$$

这由 $\varepsilon:FG\Rightarrow\operatorname{id}_{\mathcal D}$ 的自然性应用于态射 $\varepsilon_{F X}:FGF X\to F X$ 得到。$\square$

## 7.3 单子代数

**定义 7.4.** 设 $(T,\eta,\mu)$ 是 $\mathcal C$ 上的单子。一个 $T$-代数是对象 $A\in\mathcal C$ 和态射

$$
a:T A\to A
$$

满足

$$
a\circ\eta_A=\operatorname{id}_A,
\qquad
a\circ T(a)=a\circ\mu_A.
$$

若 $(A,a)$ 与 $(B,b)$ 是 $T$-代数，代数同态 $f:(A,a)\to(B,b)$ 是态射 $f:A\to B$，满足

$$
f\circ a=b\circ T(f).
$$

**命题 7.5.** $T$-代数和代数同态构成范畴，记作 $\mathcal C^T$。

**证明.** 恒等态射满足代数同态条件，因为

$$
\operatorname{id}_A\circ a=a=a\circ T(\operatorname{id}_A).
$$

若 $f:(A,a)\to(B,b)$ 和 $g:(B,b)\to(C,c)$ 是代数同态，则

$$
(g f)\circ a
=g\circ(f\circ a)
=g\circ b\circ T(f)
=c\circ T(g)\circ T(f)
=c\circ T(g f).
$$

故复合仍为代数同态。结合律和单位律来自 $\mathcal C$。$\square$

## 7.4 Kleisli 范畴

**定义 7.6.** 单子 $(T,\eta,\mu)$ 的 Kleisli 范畴 $\mathcal C_T$ 定义为：

- 对象与 $\mathcal C$ 相同。
- Hom 集为
  $$
  \mathcal C_T(X,Y)=\mathcal C(X,T Y).
  $$
- 态射 $f:X\to T Y$ 与 $g:Y\to T Z$ 的 Kleisli 复合为
  $$
  X\xrightarrow{f}T Y\xrightarrow{Tg}T^2 Z\xrightarrow{\mu_Z}T Z.
  $$
- $X$ 的恒等态射为 $\eta_X:X\to T X$。

**命题 7.7.** $\mathcal C_T$ 是范畴。

**证明.** 设 $f:X\to TY$ 为 Kleisli 态射。右单位复合为

$$
\mu_Y\circ T(\eta_Y)\circ f=f
$$

由单子单位律 $\mu\circ T\eta=\operatorname{id}_T$。左单位复合为

$$
\mu_Y\circ T(f)\circ\eta_X.
$$

由 $\eta$ 的自然性，$T(f)\eta_X=\eta_{TY}f$，再由另一单位律 $\mu_Y\eta_{TY}=\operatorname{id}_{TY}$，得到左单位复合等于 $f$。

现在设

$$
f:X\to TY,\qquad g:Y\to TZ,\qquad h:Z\to TW.
$$

先复合 $f$ 与 $g$，再与 $h$，得到

$$
\mu_W\circ T(h)\circ \mu_Z\circ T(g)\circ f.
$$

由 $\mu$ 的自然性应用于 $h:Z\to TW$，

$$
T(h)\circ \mu_Z=\mu_{TW}\circ T^2(h).
$$

所以该复合等于

$$
\mu_W\circ\mu_{TW}\circ T^2(h)\circ T(g)\circ f.
$$

另一种括号先复合 $g$ 与 $h$，再与 $f$，得到

$$
\mu_W\circ T(\mu_W\circ T(h)\circ g)\circ f
=\mu_W\circ T\mu_W\circ T^2(h)\circ T(g)\circ f.
$$

单子结合律在对象 $W$ 处给出

$$
\mu_W\circ\mu_{TW}=\mu_W\circ T\mu_W.
$$

故两种复合相等。$\mathcal C_T$ 的结合律和单位律成立。$\square$

## 7.5 Kleisli 与 Eilenberg-Moore 伴随

**命题 7.8.** 每个单子 $(T,\eta,\mu)$ 都产生一个 Kleisli 伴随

$$
J:\mathcal C\rightleftarrows \mathcal C_T:G_T,\qquad J\dashv G_T,
$$

其中 $J$ 在对象上为恒等，且把 $f:X\to Y$ 送为

$$
X\xrightarrow{f}Y\xrightarrow{\eta_Y}T Y
$$

作为 Kleisli 态射；$G_T$ 把对象 $Y$ 送到 $T Y$，并把 Kleisli 态射 $g:X\to T Y$ 送为

$$
T X\xrightarrow{T g}T^2Y\xrightarrow{\mu_Y}TY.
$$

该伴随产生的单子正是原来的 $T$。

**证明.** 先验证 $G_T$ 是函子。Kleisli 恒等态射 $\eta_X:X\to TX$ 被送到

$$
T X\xrightarrow{T\eta_X}T^2X\xrightarrow{\mu_X}TX,
$$

由单位律等于 $\operatorname{id}_{TX}$。若 $f:X\to TY$ 与 $g:Y\to TZ$ 是 Kleisli 态射，则 Kleisli 复合为

$$
\mu_ZTg f:X\to TZ.
$$

$G_T$ 作用于该复合得到

$$
\mu_ZT(\mu_ZTg f)
=\mu_ZT\mu_ZT^2gTf.
$$

另一方面，

$$
G_T(g)G_T(f)=\mu_ZTg\,\mu_YTf.
$$

由 $\mu$ 对 $g:Y\to TZ$ 的自然性，

$$
Tg\,\mu_Y=\mu_{TZ}T^2g,
$$

再由结合律 $\mu_Z\mu_{TZ}=\mu_ZT\mu_Z$，两式相等。因此 $G_T$ 是函子。

现在有自然等式

$$
\mathcal C_T(JX,Y)=\mathcal C(X,TY)=\mathcal C(X,G_TY),
$$

它给出 $J\dashv G_T$。该伴随的单位是 $\eta_X:X\to TX$。其诱导单子在对象上为 $G_TJX=TX$；乘法来自余单位，正是 $\mu:T^2\to T$。故恢复原单子。$\square$

**命题 7.9.** 遗忘函子

$$
U^T:\mathcal C^T\to\mathcal C,\qquad (A,a)\mapsto A
$$

有左伴随

$$
F^T:\mathcal C\to\mathcal C^T,\qquad X\mapsto(TX,\mu_X).
$$

称 $F^T X$ 为自由 $T$-代数。伴随双射为

$$
\mathcal C^T((TX,\mu_X),(A,a))\cong\mathcal C(X,A).
$$

**证明.** 对态射 $f:X\to Y$，令

$$
F^T(f)=T f:TX\to TY.
$$

它是代数同态，因为 $\mu$ 的自然性给出

$$
T f\circ\mu_X=\mu_Y\circ T^2f.
$$

因此 $F^T$ 是函子。

定义映射

$$
\mathcal C^T((TX,\mu_X),(A,a))\to\mathcal C(X,A),
\qquad h\mapsto h\eta_X.
$$

反向地，给定 $k:X\to A$，令

$$
\bar k= a\circ T k:TX\to A.
$$

这是代数同态，因为

$$
\bar k\mu_X
=aT k\mu_X
=a\mu_A T^2k
=aT aT^2k
=aT(aTk)
=aT\bar k,
$$

其中第二步用 $\mu$ 的自然性，第三步用 $T$-代数结合律。两个构造互逆：一方面

$$
\bar k\eta_X=aT k\eta_X=a\eta_Ak=k
$$

由 $\eta$ 的自然性和代数单位律；另一方面，若 $h:(TX,\mu_X)\to(A,a)$ 是代数同态，则

$$
aT(h\eta_X)
=aT h\,T\eta_X
=h\mu_XT\eta_X
=h.
$$

自然性由复合的结合律直接验证。因此 $F^T\dashv U^T$。$\square$

**命题 7.10.** 存在全忠实函子

$$
\mathcal C_T\to\mathcal C^T
$$

把对象 $X$ 送到自由代数 $(TX,\mu_X)$，把 Kleisli 态射 $f:X\to TY$ 送到代数同态

$$
TX\xrightarrow{T f}T^2Y\xrightarrow{\mu_Y}TY.
$$

**证明.** 命题 7.9 的伴随给出自然双射

$$
\mathcal C^T(F^TX,F^TY)\cong\mathcal C(X,U^TF^TY)=\mathcal C(X,TY).
$$

右边正是 $\mathcal C_T(X,Y)$。在该双射下，$f:X\to TY$ 对应的代数同态正是 $\mu_YT f$。因此上述函子在 Hom 集上为双射，故全忠实；复合保持性由该双射的自然性或 Kleisli 复合公式直接得到。$\square$

## 7.6 单子性

**定义 7.11.** 对伴随 $F\dashv G$ 产生的单子 $T=GF$，比较函子

$$
K:\mathcal D\to\mathcal C^T
$$

把 $Y\in\mathcal D$ 送到 $T$-代数

$$
(G Y,\, G\varepsilon_Y:GFGY\to GY).
$$

若 $K$ 是范畴等价，则称右伴随 $G$ 是单子的（monadic）。

**外部输入定理 7.12（Beck 单子性定理）.** 设 $G:\mathcal D\to\mathcal C$ 有左伴随。称平行对 $f,g:X\rightrightarrows Y$ 为 $G$-split，若其像 $Gf,Gg$ 在 $\mathcal C$ 中带有一个指定的分裂余等化子图。若 $\mathcal D$ 中每个 $G$-split 平行对都有余等化子，且 $G$ 保持这些余等化子，则 $G$ 单子当且仅当 $G$ 保守，即反映同构。

等价的常用表述把“存在且被 $G$ 保持”合写为“$G$ 创建 $G$-split 平行对的余等化子”。这里不需要 $G$ 反映任意余等化子；保守性只要求它反映同构。

本书在本章不证明该定理；后续讨论可表现范畴和代数理论时均使用上述版本。来源见 `SOURCES.md` 中 Mac Lane、Borceux 与 Riehl 的单子章节。

## 7.7 例子与边界条件

**例子 7.13（恒等单子）.** 恒等函子 $\operatorname{id}_{\mathcal C}$ 配单位和乘法都为恒等自然变换，构成单子。其 Eilenberg-Moore 代数是对象 $A$ 配态射 $\operatorname{id}_A:A\to A$，所以 $\mathcal C^{\operatorname{id}}\cong\mathcal C$。其 Kleisli 范畴也等于 $\mathcal C$。

**定义 7.14.** 单子 $(T,\eta,\mu)$ 称为幂等单子（idempotent monad），若乘法

$$
\mu:T^2\to T
$$

是自然同构。等价地，在很多常见情形中，$\eta_T:T\to T^2$ 和 $T\eta:T\to T^2$ 也是自然同构。

**命题 7.15.** 若 $L:\mathcal C\rightleftarrows\mathcal A:I$ 是反射子范畴，且 $I$ 全忠实，则诱导单子

$$
T=IL:\mathcal C\to\mathcal C
$$

是幂等单子。

**证明.** 该伴随的乘法为

$$
ILIL\xrightarrow{I\varepsilon L}IL,
$$

其中 $\varepsilon:LI\to\operatorname{id}_{\mathcal A}$ 是余单位。由于 $I$ 全忠实，命题 4.9 的对偶形式给出 $\varepsilon$ 是自然同构。因此 $I\varepsilon L$ 是自然同构，故 $T$ 幂等。$\square$

**例子 7.16（同一单子的两个标准伴随）.** 任意单子 $T$ 至少来自两个标准伴随：Kleisli 伴随

$$
J:\mathcal C\rightleftarrows\mathcal C_T:G_T
$$

和 Eilenberg-Moore 伴随

$$
F^T:\mathcal C\rightleftarrows\mathcal C^T:U^T.
$$

二者诱导同一个单子 $T$，但中间范畴通常不同。Kleisli 范畴只自由加入 $T$-效应态射；Eilenberg-Moore 范畴包含所有满足代数公理的真实代数。命题 7.10 给出 $\mathcal C_T$ 到 $\mathcal C^T$ 的全忠实嵌入，其像为自由代数之间的态射。

## 7.8 从伴随到代数范畴

单子把“自由-遗忘”伴随中的代数结构压缩到一个自函子 $T$ 及其单位、乘法中。每个单子反过来产生 Kleisli 伴随与 Eilenberg-Moore 自由-遗忘伴随；前者记录带效应的态射，后者记录真实代数。Beck 定理说明何时一个范畴可以完全由某个单子的代数恢复。

## 练习

**练习 7.1.** 验证自由幺半群单子的单位律和结合律。

**练习 7.2.** 证明自由阿贝尔群伴随产生的单子，其代数范畴等价于 $\mathbf{Ab}$。

**练习 7.3.** 写出 powerset 单子 $\mathcal P$ 在 $\mathbf{Set}$ 上的单位和乘法。

**练习 7.4.** 完成命题 7.7 的结合律证明。

**练习 7.5.** 对偶定义余单子（comonad）及其余代数。

**练习 7.6.** 验证命题 7.8 中 $J:\mathcal C\to\mathcal C_T$ 是函子。

**练习 7.7.** 写出 Kleisli 伴随 $J\dashv G_T$ 的余单位，并验证诱导乘法为 $\mu$。

**练习 7.8.** 证明命题 7.9 中 $aTk:TX\to A$ 是唯一延拓 $k:X\to A$ 的 $T$-代数同态。

**练习 7.9.** 对列表单子，解释自由 $T$-代数和 Kleisli 态射分别是什么。

**练习 7.10.** 证明命题 7.10 的函子保持 Kleisli 复合。

**练习 7.11.** 证明恒等单子的 Kleisli 范畴和 Eilenberg-Moore 范畴都同构于原范畴。

**练习 7.12.** 设 $L\dashv I$ 为反射子范畴。写出诱导幂等单子的单位和乘法。

**练习 7.13.** 对阿贝尔化反射 $\mathbf{Grp}\to\mathbf{Ab}$，描述诱导幂等单子在群 $G$ 上的值。

**练习 7.14.** 说明为什么同一个单子可由不同伴随诱导，并用 Kleisli 与 Eilenberg-Moore 伴随作例子。
