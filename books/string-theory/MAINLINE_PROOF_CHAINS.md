# 主线证明链

本文档把《String Theory》的正文组织成若干条可审查的证明链。每条证明链只列入 string theory 主线必需的结构；若一步依赖大型外部理论，则标明外部输入和使用边界。

## 1. 玻色弦主线

目标是从二维作用量推出物理态空间和 tree-level 振幅。

1. Polyakov 作用量给出二维重参数化不变性和 Weyl 不变性。
2. 变分 $X$ 得到世界面波动方程，变分 $h$ 得到 stress tensor 约束。
3. 在 conformal gauge 中，约束变成 Virasoro constraints。
4. 自由 boson CFT 给出 OPE、Virasoro algebra 和 central charge $c=D$。
5. 正则量子化给出 oscillator algebra、normal ordering constant 和质量公式。
6. BRST 量子化把 gauge redundancy 写成 cohomology。
7. 顶点算子把 BRST cohomology 类对应到 worldsheet local operators。
8. Tree-level 散射振幅由 punctured sphere 或 disk 上的 CFT correlator 和 ghost gauge fixing 给出。

外部输入：no-ghost theorem、Riemann surface moduli 分解。正文只使用其接口结论，不把证明纳入主线。

## 2. T-duality 与 D-brane 主线

目标是从圆紧化和开弦边界条件推出 D-brane 的必要性。

1. 紧化坐标 $X\sim X+2\pi R$ 允许 momentum number $n$ 和 winding number $w$。
2. 闭弦零模分解给出左右动量
   $$
   p_L=\frac nR+\frac{wR}{\alpha'},\qquad
   p_R=\frac nR-\frac{wR}{\alpha'}.
   $$
3. 谱公式在 $R\leftrightarrow \alpha'/R$、$n\leftrightarrow w$ 下不变。
4. 对开弦，T-duality 沿紧化方向把 Neumann 条件变为 Dirichlet 条件。
5. Dirichlet 条件的端点支撑子流形即 D-brane。
6. Chan-Paton factors 给出 brane stack 上的 gauge degrees of freedom。
7. 低能极限中 D-brane 动力学由 DBI 与 Wess-Zumino 耦合描述。

外部输入：D-brane tension 和 RR charge 的精确归一化由 disk one-point function 与超引力规范共同固定，正文在第十二章作为接口使用。

## 3. 超弦一致性主线

目标是从 worldsheet supersymmetry 得到十维 tachyon-free string theories。

1. RNS 形式加入 worldsheet Majorana fermions $\psi^\mu$。
2. $X^\mu,\psi^\mu$ 生成 $N=1$ superconformal matter theory。
3. Gauge fixing 引入 $bc$ 与 $\beta\gamma$ ghosts。
4. 总 superconformal anomaly 消失要求
   $$
   c_{\mathrm{matter}}+c_{\mathrm{ghost}}=\frac32D-15=0,
   $$
   因而 $D=10$。
5. NS/R sectors 给出 spacetime bosons 和 fermions。
6. GSO projection 移除 tachyon，并在 type II 情形给出 spacetime supersymmetry。
7. IIA/IIB 由左右 R sector 手征性相反或相同区分。
8. Heterotic string 由左、右不同 worldsheet CFT 组合，并由 modular invariance 限制 gauge lattice。

外部输入：spin structure 求和、modular invariance 与 anomaly cancellation 的完整证明分别属于 Riemann surface theory、CFT 和 index theory；正文只展开 string theory 中必须使用的接口和典型计算。

## 4. 几何紧化主线

目标是说明从十维超弦到低维有效理论的数学接口。

1. 低能有效作用给出 target-space supergravity fields。
2. 紧化把十维场分解为非紧空间场与紧空间 harmonic modes。
3. 保持部分 supersymmetry 要求紧空间具有相应 special holonomy；Calabi-Yau 三维情形给出四维 $\mathcal N=2$ 或经额外投影给出 $\mathcal N=1$。
4. 模空间参数对应低维标量场，flux 与量子修正可产生势能。
5. A/B model 与 mirror symmetry 抽取紧化几何中受保护的 topological sector。

外部输入：Yau theorem、Hodge theory、variation of Hodge structure。正文只在附录和第十三、十六章中给出接口化使用。

## 5. 非微扰与全息主线

目标是把 perturbative string 扩展到 branes、duality 和 holography 的受控陈述。

1. D-branes 是开弦边界条件，也是 RR charged objects。
2. BPS 条件使部分非微扰量可被弱耦合计算控制。
3. S-duality、U-duality 和 M-theory 统一不同微扰展开，但多数陈述在当前数学意义上属于物理猜想或受检验对偶。
4. Brane near-horizon limit 给出 AdS backgrounds。
5. AdS/CFT 把 bulk string theory 与 boundary conformal field theory 对应起来。
6. GKPW dictionary 把 bulk fields 的边界值对应到 CFT sources。

外部输入：AdS/CFT 的完整非微扰定义不作为已证数学定理使用；正文必须区分 conjectural dictionary、可计算检验和低能 supergravity 极限。

