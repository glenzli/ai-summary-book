# 第十二章：K3 曲面、四次曲面与 Calabi-Yau hypersurfaces

椭圆曲线上可以直接画出交点与三角形，高维 Calabi--Yau hypersurface 却通常没有一张覆盖全范畴的显式图。可行的证明会先退化几何，选出 vanishing cycles、cocores 或 immersed generators，再把庞大的范畴问题压缩为带幂等元的 $A_\infty$ 代数比较。这个压缩只有在两侧 split-generation 已证明时才有效。本章先用 adjunction 确认 Calabi--Yau 条件，再区分四次 K3、projective hypersurfaces 与 Batyrev mirror pairs 的外部定理范围，最后抽取它们共同使用的生成元逻辑与 Serre 函子检验。

## 12.1 Calabi-Yau hypersurface 数据

**定义 12.1.** 光滑 projective variety $X$ 称为 Calabi-Yau hypersurface 型，若 $X\subset \mathbb P^{n}$ 是光滑 hypersurface，且
$$
K_X\simeq\mathcal O_X.
$$
对 degree $n+1$ hypersurface，此条件由 adjunction formula 给出。

**命题 12.2.** 若 $X_d\subset\mathbb P^n$ 是光滑 degree $d$ hypersurface，则
$$
K_{X_d}\simeq\mathcal O_{X_d}(d-n-1).
$$

**证明.** Adjunction formula 给出
$$
K_{X_d}\simeq (K_{\mathbb P^n}\otimes\mathcal O_{\mathbb P^n}(X_d))|_{X_d}.
$$
由于 $K_{\mathbb P^n}\simeq\mathcal O_{\mathbb P^n}(-n-1)$ 且 $\mathcal O_{\mathbb P^n}(X_d)\simeq\mathcal O_{\mathbb P^n}(d)$，得到结论。证毕。

**例 12.3.** Quartic surface $X_4\subset\mathbb P^3$ 是 K3 surface。Quintic threefold $X_5\subset\mathbb P^4$ 是 Calabi-Yau threefold。

## 12.2 HMS 的典型形态

**定义 12.4.** 对 Calabi-Yau mirror pair $(X,Y)$，HMS 的增强版本写作
$$
\mathcal F(Y,\omega_Y)\simeq\operatorname{Perf}(X)
$$
或在 split-closed derived Fukaya category 上写作
$$
D^\pi\mathcal F(Y)\simeq H^0\operatorname{Perf}(X).
$$
若 $Y$ 以 Landau-Ginzburg 或 affine hypersurface model 出现，则 A-side category 可能替换为 Fukaya-Seidel、wrapped 或 relative Fukaya category。

**外部输入定理 12.5（Seidel quartic surface HMS）.** 对 $\mathbb P^3$
中的光滑 quartic surface 及 Seidel 所构造的镜像族和 Novikov 系数，存在其
论文所述的 derived/enhanced HMS 等价。该定理使用 Picard--Lefschetz、
graded Lagrangian spheres 与 directed Fukaya 技术；本章不重建其几何生成性
和 $A_\infty$ 计算。
来源：Seidel, *Homological mirror symmetry for the quartic surface*,
arXiv:math/0310414v4。

**外部输入定理 12.6（Sheridan Calabi-Yau hypersurfaces）.** 对 projective space 中维数 $d>2$ 的光滑 Calabi-Yau hypersurfaces，Sheridan 证明了 HMS 的重要版本。证明使用 pair-of-pants、relative Fukaya category、branched covers、Morse-Bott 模型和 matrix factorizations。
来源：Sheridan, *Homological Mirror Symmetry for Calabi-Yau hypersurfaces in projective space*。

**外部输入定理 12.7（Batyrev mirror pairs）.** 对
Ganatra--Hanlon--Hicks--Pomerleano--Sheridan 所规定的一大类、由对偶 reflexive
polytopes 构造的 Batyrev Calabi--Yau hypersurface mirror pairs，HMS 成立；
系数特征可取 $0$，并可取除有限多个素特征以外的正特征。精确 polytope
admissibility 与两侧 category model 保留为来源定理的假设，本章后续不把
该条目作为内部命题的前提。
来源：上述作者，*Homological mirror symmetry for Batyrev mirror pairs*,
arXiv:2406.05272。

## 12.3 从退化几何到生成元比较

高维 Calabi--Yau hypersurface HMS 的证明会产生下列彼此依赖的数据。

**定义 12.8（hypersurface comparison datum）.** 这样的比较数据包括：

1. 把镜像退化到 pair-of-pants 或 tropical pieces。
2. 在 A-side 构造相对或 wrapped Fukaya category。
3. 找到有限生成对象，例如 immersed spheres、vanishing cycles 或 cocores。
4. 计算这些对象的 endomorphism $A_\infty$ algebra。
5. 在 B-side 找到 matrix factorizations、graded modules 或 coherent sheaves 的生成对象。
6. 构造两边生成 full subcategories 的 strictly unital
   dg/$A_\infty$ quasi-equivalence；有限直和口径下才压缩为保持对象
   idempotents 的 endomorphism-algebra quasi-isomorphism。

**命题 12.9.** 若定义 12.8 的第 3、5、6 项成立，且第 3、5 项所列对象
分别 split-generate 两边，则得到相应 HMS 的 Morita 版本。

**证明.** 第 3、5 步给出 A/B 两边 split-generating full subcategories。第 6 步给出这些 full subcategories 的 quasi-equivalence。由生成元比较原则 8.9 得到 Morita equivalence。证毕。

## 12.4 数值检查

**定义 12.10.** 对 Calabi-Yau $n$-fold $X$，Serre functor 在 $\operatorname{Perf}(X)$ 上同构于 shift $[n]$。若 A-side Fukaya category 的 Calabi-Yau 结构与此匹配，则称维数检查通过。

**命题 12.11.** 若 HMS 等价
$$
\mathcal F(Y)\simeq\operatorname{Perf}(X)
$$
为增强等价，且 $X$ 是光滑 proper Calabi-Yau $n$-fold，则 A-side 的 Serre functor 在 Morita 意义下也对应 shift $[n]$。

**证明.** 增强等价保持 perfect module category、dualizability 和 Serre functor 的自然同构类。B-side 的 Serre functor 由 Grothendieck-Serre duality 给出 $(-)\otimes K_X[n]\simeq[n]$。经等价传回 A-side 即得结论。证毕。

高维例子的共同核心不是一份步骤清单，而是命题 12.9 的逻辑：几何退化用于找到可算生成元，真正推出全局 Morita 等价的则是两侧 split-generation 与生成 full subcategories 的 quasi-equivalence。Adjunction 和 Serre 函子提供独立的一致性检查，却不承担生成性。下一章进一步解释，pair-of-pants 与 tropical decomposition 如何把全局生成元计算拆成局部问题。

## 练习

**练习 12.1.** 用 adjunction formula 判断 degree $d$ hypersurface $X_d\subset\mathbb P^n$ 何时 Calabi-Yau。

**练习 12.2.** 解释为什么 Serre functor 是 HMS 的必要不变量。

**练习 12.3.** 把 Sheridan 的 hypersurface 结果写成定义 8.15 的比较数据，
并指出 relative Fukaya category、branched cover 与 matrix factorization
分别出现在哪一项。

**练习 12.4.** 说明调用定理 12.7 时为何必须保留 coefficient
characteristic、polytope admissibility 和两侧 category model。
