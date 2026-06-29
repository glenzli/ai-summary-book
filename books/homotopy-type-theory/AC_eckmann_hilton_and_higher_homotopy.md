# 附录 AC：Eckmann-Hilton 与高阶同伦群交换性

本附录补全第十二章定理 12.3 的证明核。核心事实是：二重 loop space 上存在两个单位相同且满足 interchange law 的复合；Eckmann-Hilton 论证推出这两个复合相同且交换。

## AC.1 抽象 Eckmann-Hilton 引理

**定义 AC.1（双幺半群结构）.** 设类型 $M$ 上有两个二元运算
$$
\star,\diamond:M\to M\to M
$$
和共同单位 $e:M$，并满足左右单位律：
$$
e\star a=a,\quad a\star e=a,\quad
e\diamond a=a,\quad a\diamond e=a.
$$
若还满足 interchange law
$$
(a\diamond b)\star(c\diamond d)
=
(a\star c)\diamond(b\star d),
$$
则称其为 Eckmann-Hilton 数据。

**定理 AC.2（Eckmann-Hilton）.** 在 AC.1 的数据下，
$$
a\star b=a\diamond b
$$
且
$$
a\star b=b\star a.
$$

**证明.** 先证明两种运算相同：
$$
\begin{aligned}
a\star b
&=(a\diamond e)\star(e\diamond b)\\
&=(a\star e)\diamond(e\star b)\\
&=a\diamond b.
\end{aligned}
$$
第一步用 $\diamond$ 的右/左单位律；第二步用 interchange；第三步用 $\star$ 的右/左单位律。

再证明交换性：
$$
\begin{aligned}
a\star b
&=a\diamond b\\
&=(e\star a)\diamond(b\star e)\\
&=(e\diamond b)\star(a\diamond e)\\
&=b\star a.
\end{aligned}
$$
第二步用 $\star$ 的单位律，第三步用 interchange 的逆路径，最后用 $\diamond$ 的单位律。$\square$

## AC.2 二重 loop space 的两个复合

设 $X:\mathcal U$，$x:X$。记
$$
\Omega X\coloneqq(x=x),
\qquad
\Omega^2X\coloneqq(\mathsf{refl}_x=\mathsf{refl}_x).
$$

**定义 AC.3（纵向复合）.** 若
$$
\alpha,\beta:\Omega^2X,
$$
定义
$$
\alpha\star\beta\coloneqq \alpha\cdot\beta
$$
为路径类型 $\mathsf{refl}_x=\mathsf{refl}_x$ 中的普通路径复合。

**定义 AC.4（横向复合）.** 二重路径也可沿一重 loop 的复合诱导横向复合：
$$
\alpha\diamond\beta
:\mathsf{refl}_x\cdot\mathsf{refl}_x
=
\mathsf{refl}_x\cdot\mathsf{refl}_x,
$$
再用左右单位律把端点识别回
$$
\mathsf{refl}_x=\mathsf{refl}_x.
$$
直观地，$\alpha\diamond\beta$ 是把 $\alpha$ 放在左方、$\beta$ 放在右方后进行 whiskering。

**引理 AC.5（共同单位）.** $\star$ 与 $\diamond$ 的单位同为
$$
\mathsf{refl}_{\mathsf{refl}_x}.
$$

**证明.** 对 $\star$ 是路径复合单位律。对 $\diamond$，横向复合的定义中 $\mathsf{refl}_{\mathsf{refl}}$ 左右 whiskering 后仍由一重路径复合单位律化为自身。$\square$

**引理 AC.6（interchange law）.** 对
$$
\alpha,\beta,\gamma,\delta:\Omega^2X,
$$
有
$$
(\alpha\diamond\beta)\star(\gamma\diamond\delta)
=
(\alpha\star\gamma)\diamond(\beta\star\delta).
$$

**证明.** 这是恒等类型的二维路径代数。对四个二重路径依次作路径归纳；反身情形中两边都化为
$$
\mathsf{refl}_{\mathsf{refl}_x}.
$$
若采用显式 whiskering 定义，中间只需要路径复合的结合律、左右单位律和 $\mathsf{ap}$ 对复合的计算。$\square$

**定理 AC.7（二重 loop space 的复合交换）.** 对任意
$$
\alpha,\beta:\Omega^2X,
$$
有
$$
\alpha\cdot\beta=\beta\cdot\alpha.
$$

**证明.** 由 AC.5 和 AC.6，$\Omega^2X$ 上的纵向复合与横向复合满足 Eckmann-Hilton 数据。应用 AC.2，纵向复合交换。$\square$

## AC.3 高阶同伦群的交换性

**定理 AC.8（高阶同伦群交换性）.** 对 $n\ge2$，
$$
\pi_n(X,x)
$$
的群运算交换。

**证明.** $\pi_n(X,x)$ 是 $\Omega^n(X,x)$ 的集合截断。当 $n=2$ 时，运算由 $\Omega^2X$ 的路径复合下降到集合截断；AC.7 给出代表元层面的交换性，因此截断归纳给出群交换律。

当 $n>2$ 时，$\Omega^nX=\Omega^2(\Omega^{n-2}X)$，同样应用 AC.7 于类型 $\Omega^{n-2}X$ 的基点。集合截断递归把代表元层面的交换性下降为 $\pi_n$ 中的交换律。$\square$

**依赖说明。** 本附录只使用恒等类型的路径归纳、路径复合、whiskering、集合截断递归，以及第十二章对 $\pi_n$ 的定义。它不使用单值性。
