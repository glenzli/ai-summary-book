# 第四卷数学审查记录

## 当前状态

第四卷已从“计算模板草稿”补强为“计算、证明和形式化补充卷”。本次审查发现原稿偏薄，证明不足；已补入以下内容：

1. 有限覆盖 sheaf 条件的等化子证明。
2. Čech 微分 $d^2=0$ 的逐项计算。
3. 紧 Hausdorff 站点中可表 sheaf 的粘合证明。
4. 基子站点比较定理的证明骨架。
5. Ext 定义与投射分解无关的证明。
6. 投射对象 Ext 消失、两项投射分解公式和维数平移。
7. Tor 的两项分解公式和平坦消失判别。
8. solid 张量积的有限层证明和普通张量积反例。
9. analytic ring 例子的有限/无限 profinite 类型检查。
10. liquid 函数分析例子的凝聚化证明。
11. pro-etale 与 condensed site 的边界说明。
12. 五个附录：形式化蓝图、练习解答、类型检查、pro-etale 对照和 pyknotic/凝聚同伦方向。
13. 凝聚基础的形式化证明义务：站点、sheaf 等化子、可表 sheaf、sheafification exact、ED 投射和 Ext/Tor 接口。
14. 凝聚谱、pyknotic 接口和谱值 sheaf 稳定性的证明模块。

## 数学口径

本卷是计算和形式化补充，不声称替代前三卷主线。凡涉及深层结构定理，仍引用前三卷或原始资料。第四卷现在的角色是：帮助读者把前三卷定理用于具体计算，并知道哪些步骤是初等证明、哪些步骤是输入定理。

## 仍作为输入定理的内容

- $\mathbf{CondAb}$ 是有足够投射对象的 Grothendieck 阿贝尔范畴。
- 极不连通紧 Hausdorff 空间给出投射自由凝聚阿贝尔群。
- Nöbeling/solid 理论中的 solid 张量积核心计算。
- analytic rings 的局部化、完备性和相干性定理。
- liquid 向量空间的深层判别定理。
- pro-etale 理论中的 w-contractible 局部对象和同调消失结论。
- pyknotic objects 与凝聚同伦类型的完整 $\infty$-范畴理论。
- 谱值 solid/analytic localization 的 monoidal compatibility。

这些内容不适合在第四卷重证；若继续扩展，应写成专题小册，而不是第五卷主线。

## 风险点

- 无限乘积和张量积的交换性必须在 solid 语境中检查。
- pro-etale site 和 condensed site 相互启发，但不是同一个站点。
- 形式化路线需要处理 universe、小性和 choice。
- ordinary completion、solidification、analytic localization 和 liquid localization 不能混用。
- 凝聚化拓扑向量空间时，必须保留连续线性结构，而不只是底层向量空间。
- 谱值 sheaf 条件必须用 totalization/hyperdescent，不能退化为一阶等化子。
