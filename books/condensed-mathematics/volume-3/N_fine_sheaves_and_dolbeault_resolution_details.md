# 附录 N：Fine sheaf 与 Dolbeault resolution 细节

## N.0 目标

附录 F 已经把 Dolbeault lemma 作为输入，并说明 Dolbeault resolution 计算上同调。本附录补齐其中的形式层：

1. fine sheaf 的定义。
2. 光滑形式层为什么是 fine。
3. fine sheaf 在 paracompact 空间上的高阶上同调为什么消失。
4. acyclic resolution 为什么计算 derived global sections。
5. Dolbeault lemma 作为输入时，如何推出 Dolbeault cohomology 计算 sheaf cohomology。

本附录不证明局部 $\bar\partial$-Poincare lemma；那是复分析输入。

## N.1 Fine sheaf

设 $X$ 是 paracompact Hausdorff 空间，$\mathcal F$ 是 $X$ 上阿贝尔群 sheaf。

**定义 N.1.** $\mathcal F$ 称为 fine sheaf，如果对任意 locally finite 开覆盖 $\{U_i\}_{i\in I}$，存在 sheaf endomorphism

$$
\theta_i:\mathcal F\to\mathcal F
$$

满足：

1. $\operatorname{supp}(\theta_i)\subset U_i$，即若开集 $V$ 与 $U_i$ 不交，则 $\theta_i|_V=0$。
2. 对每个开集 $V\subset X$ 和 $s\in\mathcal F(V)$，和式

   $$
   \sum_{i\in I}\theta_i(s)
   $$

   在 $V$ 上局部有限，因此定义良好。
3. 有

   $$
   \sum_{i\in I}\theta_i=\operatorname{id}_{\mathcal F}.
   $$

**输入定理 N.2（partition of unity）.** 设 $X$ 是 paracompact smooth manifold。对任意 locally finite smooth open cover $\{U_i\}$，存在光滑 partition of unity $\{\rho_i\}$，满足

$$
\operatorname{supp}(\rho_i)\subset U_i,\qquad
\sum_i\rho_i=1
$$

且和式局部有限。

**命题 N.3（光滑形式层是 fine）.** 若 $E$ 是 smooth complex vector bundle，则光滑 $E$-值 $k$-形式层

$$
\mathcal A_X^k(E)
$$

是 fine sheaf。特别地，复流形上的

$$
\mathcal A_X^{p,q}(E)
$$

是 fine sheaf。

**证明.** 给定 locally finite open cover $\{U_i\}$，取 subordinate partition of unity $\{\rho_i\}$。定义

$$
\theta_i(\alpha)=\rho_i\alpha
$$

对任意局部光滑形式 $\alpha$。乘以 $\rho_i$ 是 sheaf endomorphism；其支撑包含在 $\operatorname{supp}(\rho_i)\subset U_i$ 中。局部有限性保证 $\sum_i\rho_i\alpha$ 在每个开集上定义良好，并且

$$
\sum_i\theta_i(\alpha)=\left(\sum_i\rho_i\right)\alpha=\alpha.
$$

故 $\mathcal A_X^k(E)$ fine。复型 $(p,q)$-形式是复化后按双次数分解的直和因子，乘以 $\rho_i$ 保持双次数，因此同理 fine。证毕。

## N.2 Fine sheaf 的 Cech 消没

设 $\mathfrak U=\{U_i\}_{i\in I}$ 是 locally finite open cover，并固定一个全序于 $I$ 上。对 sheaf $\mathcal F$，Cech $p$-cochain 记为

$$
C^p(\mathfrak U,\mathcal F)
=
\prod_{i_0<\cdots<i_p}\mathcal F(U_{i_0\cdots i_p}).
$$

**命题 N.4（fine sheaf 的 Cech 同伦）.** 若 $\mathcal F$ 是 fine sheaf，则

$$
\check H^p(\mathfrak U,\mathcal F)=0,\qquad p>0.
$$

**证明.** 取定义 N.1 中对应覆盖的 endomorphisms $\theta_i$。对 $p>0$，定义同伦算子

$$
K:C^p(\mathfrak U,\mathcal F)\to C^{p-1}(\mathfrak U,\mathcal F)
$$

如下。若 $c\in C^p$，则在 $U_{i_0\cdots i_{p-1}}$ 上令

$$
(Kc)_{i_0\cdots i_{p-1}}
=
\sum_{j\in I}
\theta_j\left(
c_{j\,i_0\cdots i_{p-1}}
|_{U_{j\,i_0\cdots i_{p-1}}}
\right)
$$

并把每一项由支撑条件延拓为 $U_{i_0\cdots i_{p-1}}$ 上的截面。和式局部有限，故定义良好。若指标顺序不是递增，则按 Cech 交替符号重排；若出现重复指标，则该交替项取为 $0$。

直接计算 Cech 微分 $\delta$ 得

$$
\delta K+K\delta=\operatorname{id}_{C^p}
$$

在 $p>0$ 上成立。计算理由是：$\delta$ 的交替删除指标项与 $K$ 中插入指标 $j$ 的项两两相消，剩下的项为

$$
\sum_j\theta_j(c_{i_0\cdots i_p})
=c_{i_0\cdots i_p}.
$$

因此每个 $p>0$ 的 cocycle 都是 coboundary，Cech cohomology 消失。证毕。

**输入定理 N.5（paracompact Cech-sheaf 比较）.** 在 paracompact Hausdorff 空间上，若 sheaf $\mathcal F$ 对每个 locally finite open cover 的 Cech 高阶上同调消失，则

$$
H^p(X,\mathcal F)=0,\qquad p>0.
$$

等价地，fine sheaf 是全局截面函子 $\Gamma(X,-)$ 的 acyclic 对象。

**推论 N.6（fine acyclicity）.** 若 $X$ 是 paracompact Hausdorff 空间，$\mathcal F$ 是 fine sheaf，则

$$
H^p(X,\mathcal F)=0,\qquad p>0.
$$

**证明.** 由命题 N.4 和输入定理 N.5。证毕。

## N.3 Acyclic resolution 计算导出全局截面

**定义 N.7.** 设 $\Gamma:\mathbf{Sh}(X)\to\mathbf{Ab}$ 为全局截面函子。sheaf $\mathcal G$ 称为 $\Gamma$-acyclic，如果

$$
R^p\Gamma(\mathcal G)=H^p(X,\mathcal G)=0,\qquad p>0.
$$

**定理 N.8（acyclic resolution 定理）.** 设

$$
0\to\mathcal F\to\mathcal G^0\to\mathcal G^1\to\cdots
$$

是 $\mathcal F$ 的 resolution，并且每个 $\mathcal G^q$ 都是 $\Gamma$-acyclic。则自然映射

$$
R\Gamma(X,\mathcal F)\to\Gamma(X,\mathcal G^\bullet)
$$

是导出范畴中的 quasi-isomorphism。特别地，

$$
H^n(X,\mathcal F)\cong H^n(\Gamma(X,\mathcal G^\bullet)).
$$

**证明.** 对复形 $\mathcal G^\bullet$ 使用 hypercohomology 谱序列

$$
E_1^{p,q}=H^q(X,\mathcal G^p)
\Rightarrow
\mathbb H^{p+q}(X,\mathcal G^\bullet).
$$

由于每个 $\mathcal G^p$ 是 $\Gamma$-acyclic，$q>0$ 行全为零。于是谱序列退化，得到

$$
\mathbb H^n(X,\mathcal G^\bullet)
\cong
H^n(\Gamma(X,\mathcal G^\bullet)).
$$

另一方面，resolution 假设说明 $\mathcal F\to\mathcal G^\bullet$ 是 quasi-isomorphism，因此

$$
R\Gamma(X,\mathcal F)
\simeq
R\Gamma(X,\mathcal G^\bullet)
$$

并且右侧的同调就是上述 hypercohomology。证毕。

## N.4 Dolbeault resolution 的形式后果

设 $X$ 是复维数 $n$ 的 paracompact 复流形，$E$ 是全纯向量丛。

**输入定理 N.9（Dolbeault lemma with coefficients）.** 复形

$$
0\to
\mathcal O(E)
\to
\mathcal A_X^{0,0}(E)
\xrightarrow{\bar\partial}
\mathcal A_X^{0,1}(E)
\xrightarrow{\bar\partial}
\cdots
\xrightarrow{\bar\partial}
\mathcal A_X^{0,n}(E)
\to0
$$

是 $\mathcal O(E)$ 的 resolution。

**定理 N.10（Dolbeault cohomology 计算 sheaf cohomology）.** 在 N.9 的假设下，

$$
H^q(X,\mathcal O(E))
\cong
H^q\left(\Gamma(X,\mathcal A_X^{0,\bullet}(E)),\bar\partial\right).
$$

**证明.** 由命题 N.3，每个 $\mathcal A_X^{0,q}(E)$ 是 fine sheaf。由推论 N.6，它们是 $\Gamma$-acyclic。输入定理 N.9 给出 $\mathcal O(E)$ 的 resolution。应用定理 N.8，得到结论。证毕。

## N.5 与 liquid/analytic 版本的边界

定理 N.10 是经典 sheaf cohomology 结论。第三卷在 condensed/analytic 语境中使用更强陈述：

$$
R\Gamma(X,\mathcal O(E))
\simeq
\Gamma(X,\mathcal A_X^{0,\bullet}(E))
$$

不仅作为复向量空间复形成立，还要作为 liquid 或 analytic 派生范畴中的等价成立。

**输入定理 N.11（condensed/liquid Dolbeault comparison）.** 对第三卷使用的 compact complex manifold 和全纯向量丛，Dolbeault 复形中的 Fréchet 空间及 $\bar\partial$ 算子可提升到 liquid/analytic 范畴，并且 N.10 的 quasi-isomorphism 在该范畴中仍成立。

**边界说明.** 本附录证明 N.10 的 sheaf-theoretic 形式部分；N.11 需要 Clausen-Scholze 的 analytic/liquid 函数空间理论，仍作为外部输入。

## N.6 练习

**练习 N.1.** 对实直线上的覆盖 $U_1=(-\infty,1)$、$U_2=(-1,\infty)$，写出 subordinate smooth partition of unity 的构造思路。

**练习 N.2.** 验证命题 N.4 中 $\delta K+K\delta=\operatorname{id}$ 在 $p=1$ 的情形。

**练习 N.3.** 说明为什么 fine sheaf 的直和与直和因子仍是 fine。

**练习 N.4.** 用定理 N.8 证明：若 $0\to\mathcal F\to\mathcal G^0\to\mathcal G^1\to0$ 是 acyclic resolution，则有长正合列截断给出的计算公式。
