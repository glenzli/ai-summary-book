# 第四卷练习答案与教师手册补充

作者：Dr. Stochastic Parrot

## 使用说明

全书统一答案见 [../SOLUTIONS.md](../SOLUTIONS.md)。本文件补第四卷形式化、计算和谱值接口题目。

## 1. Sheaf 等化子的形式化数据

形式化有限覆盖 sheaf 条件需要：

1. 对象 \(U\)；
2. 有限指标集 \(I\)；
3. 态射 \(U_i\to U\)；
4. 纤维积 \(U_i\times_UU_j\)；
5. 两个限制映射；
6. 等化子证明。

核心公式为

$$
F(U)\to\prod_iF(U_i)
\rightrightarrows
\prod_{i,j}F(U_i\times_UU_j).
$$

## 2. 普通张量积反例

映射

$$
(\prod_{n\ge1}\mathbb Z)\otimes\mathbb Q
\to
\prod_{n\ge1}\mathbb Q
$$

不是满射。若 \((1,1/2,1/3,\ldots)\) 来自左侧，则它可写为有限和，每项有有限分母。于是所有坐标分母被一个整数统一控制。但序列第 \(n\) 项需要分母 \(n\)，不存在统一分母，矛盾。

## 3. pro-étale 与 condensed site 的差异

按 Bhatt--Scholze 的原始口径，pro-étale site 的对象是 weakly étale 几何对象
\(U\to X\)，覆盖取 fpqc covering；pro-étale 逆极限描述给同一个 Grothendieck
topology。condensed site 的对象是紧 Hausdorff 空间，覆盖来自有限联合满射。二者都
使用投射型局部对象简化 sheaf cohomology，但不是同一站点；不能把紧 Hausdorff 空间
直接当作 \(X_{\mathrm{proet}}\) 的对象，除非额外给出到 \(X\) 的几何结构并验证覆盖。

## 4. 谱值 sheaf 条件

阿贝尔群值 sheaf 对有限覆盖可用一阶等化子表达。谱值 sheaf 必须使用 totalization：

$$
E(U)\simeq\varprojlim_{\Delta}E(U_\bullet).
$$

原因是谱范畴含高阶同伦相容；一阶 equalizer 只看 \(\pi_0\) 层面的匹配，不控制 higher coherence。

## 5. 凝聚谱的零对象检测

在 hypercomplete 谱值 sheaf 范畴中，若 \(E\) 的所有 homotopy sheaves
\(\pi_n(E)\) 为零，则每个 Postnikov 截断都为零。hypercompleteness 把 \(E\) 识别为
这些截断的极限，故 \(E\simeq0\)。不能把 homotopy sheaf 的值直接等同于每个
\(E(S)\) 的 homotopy group；在非 hypercomplete 范畴中还须另加 Postnikov
left-completeness 假设。
