# 主线证明链

本文档把《String Theory》的正文组织成若干条可审查的证明链。每条证明链只列入 string theory 主线必需的结构；若一步依赖大型外部理论，则标明外部输入和使用边界。

## 1. 玻色弦主线

目标是从二维作用量推出物理态空间和 tree-level 振幅。

1. Polyakov 作用量给出二维重参数化不变性和 Weyl 不变性。
2. 变分 $X$ 得到世界面波动方程，变分 $h$ 得到 stress tensor 约束。
3. 在 conformal gauge 中，约束变成 Virasoro constraints；正则形式仍是原 metric
   equation，不是 gauge-fixed action 自动给出的零算符。
4. 自由 boson CFT 在 point-split operator algebra 中给出 OPE、Virasoro algebra 和
   central charge $c=D$。
5. 正则量子化先在有限激发公共定义域上给出 oscillator/Virasoro operators；截距使用
   声明的 regulator，再由 BRST/Lorentz closure 固定。
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
3. 谱公式在 $R\leftrightarrow \alpha'/R$、$n\leftrightarrow w$ 下不变；这一步只证明谱。
4. 完整 compact-boson operator algebra 等价是 `E` 类 CFT 输入；Buscher dilaton
   shift 另依赖 gauged path integral 与 determinant regulator。
5. 对开弦，on-shell dual coordinate 把 Neumann 条件变为 Dirichlet 条件。
6. 光滑子流形加 Chan--Paton bundle 是几何 D-brane 的领先阶模型；exact 对象是
   boundary CFT condition。
7. Chan-Paton factors 给出 brane stack 上的 gauge degrees of freedom。
8. 低能极限中 D-brane 动力学由 DBI 与 Wess-Zumino 耦合描述。

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
6. GSO projection 的有限 oscillator calculation 移除 tachyon 并选择 R chirality。
7. Genus-one spin-structure modular invariance 是独立 `E` 类输入；它不由局部
   central-charge cancellation 或单弦投影自动推出。
8. IIA/IIB 由左右 R sector 手征性相反或相同区分。
9. Heterotic string 由左、右不同 worldsheet CFT 组合，并由 modular invariance 限制 gauge lattice；target-space local anomaly factorization 是另一条必要条件。

外部输入：spin structure 求和、modular invariance 与 anomaly cancellation 的完整证明分别属于 Riemann surface theory、CFT 和 index theory；正文只展开 string theory 中必须使用的接口和典型计算。

## 4. 几何紧化主线

目标是说明从十维超弦到低维有效理论的数学接口。

1. 低能有效作用给出 target-space supergravity fields。
2. 紧化把十维场分解为非紧空间场与紧空间 harmonic modes。
3. 保持部分 supersymmetry 要求紧空间具有相应 special holonomy；holonomy 恰为
   $SU(3)$、无 flux/orientifold 的 Calabi--Yau threefold 给出四维 $\mathcal N=2$，
   真子 holonomy 可增强 supersymmetry，额外投影可降为 $\mathcal N=1$。
4. 模空间参数对应低维标量场，flux 与量子修正可产生势能。
5. A/B model 与 mirror symmetry 抽取紧化几何中受保护的 topological sector。

外部输入：Yau theorem、Hodge theory、variation of Hodge structure。正文只在附录和第十三、十六章中给出接口化使用。

## 5. 非微扰与全息主线

目标是把 perturbative string 扩展到 branes、duality 和 holography 的受控陈述。

1. D-branes 是开弦边界条件，也是 RR charged objects。
2. BPS 条件使部分非微扰量可被弱耦合计算控制。
3. D1--D5--P 的 Cardy 与 area 结果只在声明的 large-charge regime 比较 leading
   asymptotic；quantum entropy 与微观 index 的全阶相等保持研究边界。
4. S-duality、U-duality 和 M-theory 统一不同微扰展开，但多数陈述在当前数学意义上属于物理猜想或受检验对偶。
5. 外部输入的 D3 supergravity solution 经有限 near-horizon 计算给出 AdS background；
   decoupling 与完整对偶不是该坐标计算的定理推论。
6. AdS/CFT 与 GKPW dictionary 都保持 `C` 状态；固定 AdS 上的 indicial equation、
   CFT two-point scaling 和 holographic counterterm calculation 分别保留各自的 `P/S`
   条件状态。

外部输入：AdS/CFT 的完整非微扰定义不作为已证数学定理使用；正文必须区分 conjectural dictionary、可计算检验和低能 supergravity 极限。
