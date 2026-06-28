# 第七章：pro-etale topology 与凝聚数学

## 本章目标

本章说明 pro-etale topology 与凝聚数学的关系。二者都强调 pro-对象、极不连通对象和更好的同调行为，但站点不同。

## 7.1 pro-etale topology

Bhatt-Scholze 的 pro-etale topology 为 scheme 引入更细的站点，以改善 etale cohomology 的局部结构。

pro-etale site 记为

$$
X_{\operatorname{proet}}.
$$

对象可理解为 $X$ 上的 pro-etale 映射 $U\to X$，覆盖由 jointly surjective 的 pro-etale 族给出。这里的 $U$ 仍是几何对象，而不是任意紧 Hausdorff 测试空间。

## 7.2 极不连通对象

pro-etale topology 中也出现类似“足够投射”的对象。它们使许多 sheaf cohomology 问题可用更接近代数的方式处理。

凝聚数学中的极不连通紧 Hausdorff 空间扮演类似角色，但所在范畴是 compact Hausdorff 测试空间。

**比较命题 7.2.1（角色相似，不是对象相同）。** pro-etale 理论中的 w-contractible 或极不连通型对象，与凝聚数学中的 extremally disconnected compact Hausdorff 空间，承担相同的技术角色：它们使覆盖提升问题更容易，并降低 sheaf cohomology 的复杂度。但二者属于不同站点，不能直接互换。

**证明。** 技术角色来自投射性：给定覆盖 $V\to U$ 和到 $U$ 的映射，投射型对象允许在细化后提升到 $V$。这个性质在 sheaf 计算中意味着匹配族和高阶 Čech 障碍更容易控制。pro-etale 站点中的对象带有到 scheme $X$ 的结构映射，凝聚站点中的对象是紧 Hausdorff 测试空间；二者的态射、覆盖和纤维积定义不同。因此只能比较方法，不能把一个站点中的对象当作另一个站点中的对象。证毕。

## 7.3 不应混同

1. condensed site 测试的是紧 Hausdorff 空间。
2. pro-etale site 测试的是 scheme 上的 pro-etale 对象。
3. 二者都使用 sheaf 和极不连通/投射思想。
4. 二者服务的几何问题不同，但技术哲学相近。

## 7.4 一个可检查的共同模式

许多计算可抽象成如下模式。设 $\mathcal C$ 是一个站点，$\mathcal P\subset\mathcal C$ 是一族“投射测试对象”，满足每个对象可由 $\mathcal P$ 中对象覆盖，并且 $\mathcal P$ 上的覆盖提升问题足够好。则 sheaf $F$ 的信息常可通过 $F(P)$、$P\in\mathcal P$ 和共同细化来控制。

凝聚数学中，$\mathcal P$ 可取极不连通紧 Hausdorff 空间；pro-etale 理论中，$\mathcal P$ 是相应的 w-contractible/pro-etale 局部对象。第二章的基子站点比较定理正是这一模式的抽象版本。

## 7.5 本章小结

pro-etale topology 是凝聚数学的重要背景。理解它有助于理解为什么极不连通对象和 pro-对象会在现代几何中反复出现。

## 练习

**练习 7.1.** 查阅 pro-etale topology 的定义，写出 pro-etale morphism 的基本特征。

**练习 7.2.** 比较 pro-etale site 与 compact Hausdorff site。

**练习 7.3.** 解释“投射测试对象”思想在两个理论中的共同点。

**练习 7.4.** 举出一个陈述，说明为什么“pro-etale 对象”和“紧 Hausdorff 测试对象”不能直接等同。
