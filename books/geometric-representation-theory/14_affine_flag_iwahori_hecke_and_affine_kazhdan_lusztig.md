# 第十四章：Affine flag varieties、Iwahori-Hecke categories 与 affine Kazhdan-Lusztig theory

## 本章目标

本章从 affine Grassmannian 过渡到 affine flag variety，引入 Iwahori subgroup、affine Weyl group、Iwahori-Hecke category 和 affine Kazhdan-Lusztig theory。

## 依赖前置知识

需要第十二章的 loop group 和 affine Grassmannian，第四章的 Hecke category。

## 14.1 Iwahori subgroup 和 affine flag variety

**定义 14.1.** 取 $G$ split reductive，令
$$
L^+G=G(k[[z]]).
$$
评价映射
$$
\operatorname{ev}_0:L^+G\to G,\qquad z\mapsto0
$$
的 $B$-逆像
$$
I=\operatorname{ev}_0^{-1}(B)
$$
称为 Iwahori subgroup。

**定义 14.2.** affine flag variety 定义为 fpqc sheaf
$$
\operatorname{Fl}_G=LG/I.
$$
其 $k$-点为 $G(k((z)))/I(k)$。

**外部输入定理 14.3.** $\operatorname{Fl}_G$ 可表示为 ind-projective ind-scheme，并有由 extended affine Weyl group $\widetilde W$ 参数化的 Iwahori orbit decomposition
$$
\operatorname{Fl}_G=\coprod_{w\in\widetilde W} IwI/I.
$$
每个 orbit 是 affine space，维数为 affine length $\ell(w)$。

**定义 14.3.1.** 令 $W_0$ 为有限 Weyl group，$X_\ast(T)$ 为 coweight lattice，$Q^\vee$ 为 coroot lattice。affine Weyl group 和 extended affine Weyl group 分别为
$$
W_{\mathrm{aff}}=Q^\vee\rtimes W_0,\qquad
\widetilde W=X_\ast(T)\rtimes W_0.
$$
对 $\lambda\in X_\ast(T)$ 记 $t^\lambda$ 为对应 translation。affine simple reflections 是 alcove walls 的反射；它们生成 Coxeter group $W_{\mathrm{aff}}$。$\widetilde W$ 一般不是 Coxeter group，但含有长度为零的有限群 $\Omega$，并有分解
$$
\widetilde W=W_{\mathrm{aff}}\rtimes\Omega.
$$

**例 14.3.2.** 对 $G=GL_n$，点 $\operatorname{Gr}_G$ 可用 $\mathcal O$-lattices $L\subset \mathcal K^n$ 描述。点 $\operatorname{Fl}_G$ 则可用 periodic lattice chains
$$
L_0\supset L_1\supset\cdots\supset L_{n-1}\supset zL_0
$$
描述，其中每个商 $L_i/L_{i+1}$ 为一维。Iwahori subgroup 是保持标准 lattice chain 的 $G(\mathcal K)$ 子群。

**命题 14.3.3.** 自然投影
$$
\pi:\operatorname{Fl}_G=LG/I\longrightarrow LG/L^+G=\operatorname{Gr}_G
$$
在中性点所在 fiber 上给出
$$
\pi^{-1}(L^+G/L^+G)\simeq L^+G/I\simeq G/B.
$$

**证明.** 投影由包含 $I\subset L^+G$ 诱导。中性点的 fiber 是所有 $gI$ 使得 $gL^+G=L^+G$，即 $g\in L^+G$，故 fiber 为 $L^+G/I$。评价映射 $\operatorname{ev}_0:L^+G\to G$ 的核是 pro-unipotent congruence subgroup，$I=\operatorname{ev}_0^{-1}(B)$，所以商 $L^+G/I$ 与 $G/B$ 同构。$\square$

**命题 14.3.4.** $\pi$ 把 $I$-orbit $IwI/I$ 映到某个 $L^+G$-orbit，后者由 $w$ 的 translation part 所决定的 dominant coweight 参数化。

**证明.** $\operatorname{Gr}_G$ 的 $L^+G$-orbits 由 Cartan decomposition 参数化：
$$
LG=\coprod_{\lambda\in X_\ast(T)^+}L^+G\,t^\lambda\,L^+G.
$$
把 $IwI$ 放入 $L^+G w L^+G$ 后，double coset 只记录 $w$ 在
$$
L^+G\backslash LG/L^+G
$$
中的像。该 double coset 由唯一 dominant coweight 给出。严格的唯一性是 Cartan decomposition 的一部分。$\square$

## 14.2 Iwahori-Hecke category

**定义 14.4.** Iwahori-Hecke category 定义为
$$
\mathsf H_I=D^b_I(\operatorname{Fl}_G,E)
$$
或相应 mixed/monodromic 版本。其 perverse heart 中的 simple objects 为 Schubert IC sheaves
$$
\operatorname{IC}_w=\operatorname{IC}(\overline{IwI/I}).
$$

**定义 14.5.** 卷积 correspondence 为
$$
\operatorname{Fl}_G\times\operatorname{Fl}_G
\xleftarrow{\ p\ }
LG\times^I\operatorname{Fl}_G
\xrightarrow{\ m\ }
\operatorname{Fl}_G.
$$
定义
$$
\mathcal F\star\mathcal G=m_!p^\ast(\mathcal F\boxtimes\mathcal G).
$$

**定义 14.5.1.** 对 $w\in\widetilde W$ 记
$$
\Delta_w=j_{w!}E_{IwI/I}[\ell(w)],\qquad
\nabla_w=j_{w\ast}E_{IwI/I}[\ell(w)]
$$
为 standard 和 costandard objects，其中 $j_w:IwI/I\hookrightarrow\operatorname{Fl}_G$。若工作在 mixed setting，则还要加入 Tate twist 以锁定 weight convention。

**命题 14.5.2.** 若 $s$ 是 affine simple reflection，则
$$
\overline{IsI/I}\simeq\mathbb P^1
$$
并且 $IsI/I\simeq\mathbb A^1$ 是其开 cell。

**证明.** $s$ 对应 affine Dynkin 图上的一个 parahoric subgroup $P_s$，满足
$$
I\subset P_s,\qquad P_s/I\simeq\mathbb P^1.
$$
Bruhat 分解给出
$$
P_s/I=I/I\sqcup IsI/I.
$$
第二个 cell 维数为 $\ell(s)=1$，因此为 $\mathbb A^1$；闭包即 $P_s/I\simeq\mathbb P^1$。$\square$

**命题 14.5.3.** 若 $\ell(ww')=\ell(w)+\ell(w')$，则在标准对象层面有规范同构
$$
\Delta_w\star\Delta_{w'}\simeq\Delta_{ww'}.
$$

**证明.** 长度可加时，乘法映射
$$
IwI\times^I Iw'I/I\longrightarrow Iww'I/I
$$
是相应 open Schubert cell 的同构。对常值层取 extension by zero，并使用 $m_!$ 的定义，得到 $\Delta_{ww'}$。shift 的相容性来自
$$
\ell(ww')=\ell(w)+\ell(w').
$$
若长度不可加，乘法像落入较小 Bruhat stratum 的闭包并出现 lower terms，这正是 Hecke 关系的几何来源。$\square$

**命题 14.6.** 在 ind-proper 和 constructibility 假设满足时，Iwahori convolution 结合。

**证明.** 三重 convolution space
$$
LG\times^I LG\times^I\operatorname{Fl}_G
$$
控制两种加括号方式。contracted product 的 associativity 和 loop group 乘法结合律给出 correspondence 的同构；六函子 base change 给出 functor associator。$\square$

## 14.3 Affine KL theory

**外部输入定理 14.7.** 在合适 mixed setting 中，$K_0(\mathsf H_I)$ 与 affine Hecke algebra 同构，Schubert IC sheaves 的类对应 affine Kazhdan-Lusztig basis。

**例 14.7.1.** 在 rank one affine 情形中，$W_{\mathrm{aff}}$ 是 infinite dihedral group，由两个 affine simple reflections $s_0,s_1$ 生成。长度为 $0,1,2,\ldots$ 的元素交替出现。对应 Schubert varieties 形成一串闭包包含关系；标准对象的卷积满足
$$
\Delta_{s_i}\star\Delta_{s_j}\simeq \Delta_{s_is_j}\quad(i\ne j),
$$
而 $\Delta_{s_i}\star\Delta_{s_i}$ 产生 Hecke 二次关系的几何版本。

**边界说明 14.8.** affine KL theory 在 modular representation theory、quantum groups at roots of unity 和 affine Lie algebra category $\mathcal O$ 中有多个版本。每个版本的 Coxeter group、alcove convention、参数和 grading 必须单独登记。

**检查表 14.9.** 使用 affine Hecke category 结果时必须声明：

1. 使用 $W_{\mathrm{aff}}$ 还是 $\widetilde W$；
2. 是否允许 monodromic、mixed 或 parity 版本；
3. 标准对象 shift 是否为 $[\ell(w)]$；
4. Hecke algebra 参数是 $v$、$q^{1/2}$ 还是 cohomological grading；
5. IC sheaf 类对应标准 KL basis 还是反标准归一化；
6. 是否处在正特征或 modular 系数下。

## 本章小结

本章定义 affine flag variety、Iwahori-Hecke category、标准对象、affine simple reflection 的局部模型，并给出卷积结合性和长度可加乘法的证明。ind-projectivity、orbit decomposition、Cartan decomposition 和 affine KL basis theorem 是外部输入。

## 练习

**练习 14.1.** 对 $G=GL_n$，描述 Iwahori subgroup 为模 $z$ 后落入上三角矩阵的 loop matrices。

**练习 14.2.** 比较 $\operatorname{Gr}_G=LG/L^+G$ 与 $\operatorname{Fl}_G=LG/I$ 的 orbit 参数。

**练习 14.3.** 写出 affine simple reflection 对应的最小 Schubert variety 的预期形状。

**练习 14.4.** 对 $G=SL_2$，把 $\widetilde W$ 识别为 infinite dihedral group 的一个扩张，并写出前四个 Schubert closures。

**练习 14.5.** 证明长度可加条件失效时，$\Delta_s\star\Delta_s$ 不可能等于 $\Delta_e$ 或 $\Delta_s$ 中的单一对象。
