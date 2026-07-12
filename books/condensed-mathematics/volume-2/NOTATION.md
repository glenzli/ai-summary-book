# 第二卷符号约定

本卷沿用第一卷 [NOTATION.md](../volume-1/NOTATION.md)。

## 固定符号

- $\mathbf{CondAb}$：凝聚阿贝尔群范畴。
- $\mathbf{Solid}$：固体阿贝尔群范畴。
- $D(\mathbf{CondAb})$：固定 \(\kappa\)-层级凝聚阿贝尔群的无界导出稳定
  \(\infty\)-范畴；三角范畴陈述指其同伦范畴。
- $D_{\square}(\mathbb Z)$：solid 复形构成的派生范畴。
- $\mathbb Z^\square[S]$：profinite 集合 $S$ 上的自由 solid 阿贝尔群。
- $M^\square$：solidification。
- $L^\square$：solidification 的导出函子。
- $\otimes^{L,\square}$：派生 solid 张量积。
- $(A,\mathcal M)$：解析环。
- $(A,\mathcal M)\text{-}\mathbf{Mod}$：解析模范畴。
- $K_S^{\mathcal M}$：$\operatorname{Cone}(A[\underline S]\to\mathcal M[S])$。
- $L_{(A,\mathcal M)}$：解析化函子。
- $\mathcal M_p[S]$、$\mathcal M_{<p}[S]$：liquid 理论中的 $p$-型测度对象。
- $\mathbf{Liquid}_p$：满足 \(\mathcal M_{<p}[S]\) 唯一延拓条件的凝聚阿贝尔群
  满子范畴，固定 \(0<p\le1\)。
- $\mathcal L_p(E)$：仅表示经典拓扑向量空间的关联凝聚模
  \(\underline E(S)=\operatorname{Cont}(S,E)\) 已被证明属于
  \(\mathbf{Liquid}_p\)；不是额外 realization 对象。
- $(A,A^+)$：离散 Huber pair。
- $\operatorname{Spa}(A,A^+)$：Huber pair 的 valuation 空间。
- $f_!$：紧支撑推前。
- $f^!$：$f_!$ 的右伴随。

## 约定

- “输入定理”表示本卷使用但不在当前位置完整证明的正式定理。
- “第一卷”总是指 [凝聚数学讲义：第一卷](../volume-1/)。
- 若某个公式需要集合论大小条件，默认沿用第一卷附录 A 的 universe 约定。
- 无下标测试对象 \(S\) 属于 \(\mathbf{ProFin}_\kappa\) 或
  \(\mathbf{ED}_\kappa\)；“所有集合 \(I\)”在固定层级正文中表示 \(|I|<\kappa\)。
- $R\operatorname{Hom}$ 表示取值于普通导出阿贝尔群的导出 Hom；
  $R\underline{\operatorname{Hom}}$ 表示内部凝聚导出 Hom。
- $\operatorname{Cone}$ 在稳定增强中按 cofiber 解释，避免把三角范畴中的非函子性
  cone 当作已选定对象。
- “逐项 liquid”只判断对象类型，不推出连续满射凝聚化后为 epimorphism。后者称为
  凝聚严格性，按第五章的 profinite 参数族局部提升判据检查。
