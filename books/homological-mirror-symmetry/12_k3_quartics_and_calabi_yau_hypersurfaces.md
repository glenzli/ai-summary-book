# 第十二章：K3 曲面、四次曲面与 Calabi-Yau hypersurfaces

## 本章目标

本章记录 HMS 在高维 Calabi-Yau hypersurfaces 中的标准形态和证明策略。重点是把 quartic K3、quintic threefold、projective-space hypersurfaces 与 Batyrev mirror pairs 放进同一范畴模板，而不是复述完整证明。

## 依赖前置知识

需要第八章 HMS 模板、第十一章 Fukaya-Seidel 技术和第二章 B-side enhancement。

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

**外部输入定理 12.5（Seidel quartic surface HMS）.** 四次 K3 曲面的 HMS 有 Seidel 证明的版本，使用 Picard-Lefschetz、graded Lagrangian spheres 和 directed Fukaya category 技术。  
来源：Seidel 的 quartic surface HMS 相关论文与专著接口；后续 theorem locator 需精确定位。

**外部输入定理 12.6（Sheridan Calabi-Yau hypersurfaces）.** 对 projective space 中维数 $d>2$ 的光滑 Calabi-Yau hypersurfaces，Sheridan 证明了 HMS 的重要版本。证明使用 pair-of-pants、relative Fukaya category、branched covers、Morse-Bott 模型和 matrix factorizations。  
来源：Sheridan, *Homological Mirror Symmetry for Calabi-Yau hypersurfaces in projective space*。

**外部输入定理 12.7（Batyrev mirror pairs 方向）.** 对由 dual reflexive polytopes 构造的大类 Batyrev Calabi-Yau mirror pairs，已有 2024 年证明型结果建立 HMS。  
来源：Ganatra-Hanlon-Hicks-Pomerleano-Sheridan, *Homological mirror symmetry for Batyrev mirror pairs*。本书把它列为近期研究边界，进入基础定理链前需 theorem locator 和假设审查。

## 12.3 证明策略的共同结构

高维 Calabi-Yau hypersurface HMS 的证明通常包含以下步骤。

**步骤 12.8.**

1. 把镜像退化到 pair-of-pants 或 tropical pieces。
2. 在 A-side 构造相对或 wrapped Fukaya category。
3. 找到有限生成对象，例如 immersed spheres、vanishing cycles 或 cocores。
4. 计算这些对象的 endomorphism $A_\infty$ algebra。
5. 在 B-side 找到 matrix factorizations、graded modules 或 coherent sheaves 的生成对象。
6. 构造两边生成 full subcategories 的 strictly unital
   dg/$A_\infty$ quasi-equivalence；有限直和口径下才压缩为保持对象
   idempotents 的 endomorphism-algebra quasi-isomorphism。

**命题 12.9.** 若步骤 12.8 的第 3、5、6 步成立，且第 3、5 步所列对象
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

## 本章小结

K3、quintic 和更一般 Calabi-Yau hypersurfaces 的 HMS 已有深刻证明型结果，但证明依赖大量几何和 $A_\infty$ 技术。本书在此阶段只把它们纳入严格模板：明确两边类别、生成对象、endomorphism algebra 和外部输入，而不把大型证明压缩成未经验证的正文推导。

## 练习

**练习 12.1.** 用 adjunction formula 判断 degree $d$ hypersurface $X_d\subset\mathbb P^n$ 何时 Calabi-Yau。

**练习 12.2.** 解释为什么 Serre functor 是 HMS 的必要不变量。

**练习 12.3.** 把 Sheridan 证明策略按模板 8.15 重写成九项清单。

**练习 12.4.** 说明 Batyrev mirror pairs 结果进入基础定理链前需要核查哪些假设。
