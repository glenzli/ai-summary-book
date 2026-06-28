# 第一章：形式化凝聚数学的对象

## 本章目标

本章说明凝聚数学中哪些对象适合形式化，以及形式化证明通常如何拆分。

## 1.1 已形式化的基础方向

形式化基础论文处理了 coherent topology、regular/extensive topology 和 compact Hausdorff 站点上的 sheaf 条件。这类内容适合 Lean 形式化，因为它们主要由范畴论定义和有限极限条件组成。

## 1.2 形式化对象清单

适合形式化的对象包括：

1. Grothendieck topology。
2. presheaf 与 sheaf。
3. coherent site。
4. compact Hausdorff 站点的覆盖条件。
5. 站点比较定理的抽象版本。
6. 凝聚集合和凝聚阿贝尔群。
7. 极不连通测试对象的投射性，若接受 Gleason 定理作为输入。

## 1.3 形式化路线

一个典型路线：

1. 定义站点 $(\mathcal C,J)$。
2. 定义 sheaf 条件为 matching families 的唯一粘合。
3. 证明 sheaf 条件与等化子条件等价。
4. 定义基子站点。
5. 证明限制函子给出 sheaf 范畴等价。
6. 专门化到 $\mathbf{CHaus}$、$\mathbf{ProFin}$ 或 $\mathbf{ED}$。

## 1.4 难点

形式化凝聚数学的主要难点不是写公式，而是：

1. universe 层级。
2. topological spaces 的小性控制。
3. choice 和 ultrafilter。
4. compact Hausdorff 定理库。
5. derived category 和 stable category 的库支持。

## 1.5 本章小结

形式化工作应从站点和 sheaf 的抽象层开始，再逐步进入 compact Hausdorff 几何。solid、analytic、liquid 的形式化需要更多底层库。

## 练习

**练习 1.1.** 把 sheaf 条件写成 Lean 风格的数据：匹配族、粘合、唯一性。

**练习 1.2.** 说明站点比较定理需要哪些假设。
