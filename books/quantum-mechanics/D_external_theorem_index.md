# 附录 D：外部输入定理索引

本附录集中列出本书使用但不在正文内部证明的大型定理。

**外部输入定理 D.1（谱定理，QM-EXT-1）.** 设 $A$ 为复 Hilbert 空间 $\mathcal H$ 上自伴算子。存在唯一投影值测度 $E_A$，使得
$$
A=\int_{\mathbb R}\lambda\,dE_A(\lambda)
$$
按无界算子的谱积分意义成立。对 Borel 函数 $f:\mathbb R\to\mathbb C$ 可定义
$$
f(A)=\int_{\mathbb R}f(\lambda)\,dE_A(\lambda).
$$
其定义域为
$$
\mathcal D(f(A))
=\left\{\psi\in\mathcal H:
\int_{\mathbb R}|f(\lambda)|^2\,d\mu_\psi^A(\lambda)<\infty
\right\}.
$$

**外部输入定理 D.2（Stone 定理，QM-EXT-2）.** 强连续一参数酉群 $U(t)$ 唯一写成 $U(t)=e^{-itH}$，其中 $H$ 为自伴算子。

**外部输入定理 D.3（Kato-Rellich 定理，QM-EXT-3）.** 若 $A$ 自伴，$B$ 对 $A$ 相对有界且相对界小于 $1$，则 $A+B$ 在 $\mathcal D(A)$ 上自伴。

**外部输入定理 D.4（Wigner 定理，QM-EXT-5）.** 保持射线转移概率的射线变换由酉或反酉算子实现，且实现算子在相位意义下唯一。

**外部输入定理 D.5（Stinespring 表示，QM-EXT-6）.** 任意正规完全正映射可表示为 Hilbert 空间扩张上的压缩；有限维情形等价于 Kraus 表示。

**外部输入定理 D.6（球谐函数完备性，QM-EXT-10）.** 球面 Laplacian 的本征函数 $Y_\ell^m$ 构成 $L^2(S^2)$ 的正交归一基，且对应本征值为 $\ell(\ell+1)$。

**外部输入定理 D.7（Coulomb Hamiltonian 谱理论，QM-EXT-10）.** 氢型 Hamiltonian
$$
-\frac1{2\mu}\Delta-\frac{Ze^2}{r}
$$
在标准定义域上自伴，其负离散谱由
$$
E_n=-\frac{\mu Z^2e^4}{2n^2}
$$
给出，并带有由球谐函数和径向 Laguerre 函数组成的完备束缚态族。这里沿用正文 $\hbar=1$ 和 $e^2$ 表示库仑耦合常数的约定。

**外部输入定理 D.8（磁 Schrodinger 算子，QM-EXT-11）.** 取
$m>0$、$q\in\mathbb R$，并设
$A\in L^2_{\mathrm{loc}}(\mathbb R^d;\mathbb R^d)$、
$\Phi\in L^1_{\mathrm{loc}}(\mathbb R^d;\mathbb R)$。若定义在
$C_c^\infty(\mathbb R^d)$ 上的最小耦合二次型
$$
q_{A,\Phi}[\psi]
=\frac1{2m}\|(-i\nabla-qA)\psi\|_2^2
+\int_{\mathbb R^d}q\Phi|\psi|^2\,dx
$$
稠定、下有界且可闭，则其闭包唯一表示一个下有界自伴 Hamiltonian。
具体的充分条件可用 $q\Phi$ 的负部相对于磁动能形式的相对形式界
小于 $1$ 来表述；本书只使用上述形式层结论。

**外部输入定理 D.9（有限维 $\mathfrak{su}(2)$ 表示分解，QM-EXT-12）.** 自旋 $j_1$ 与 $j_2$ 的不可约表示张量积分解为
$$
j_1\otimes j_2\cong\bigoplus_{j=|j_1-j_2|}^{j_1+j_2}j.
$$

**外部输入定理 D.10（Wigner-Eckart 定理，QM-EXT-13）.** 球张量算子的角动量矩阵元分解为 Clebsch-Gordan 系数和与磁量子数无关的约化矩阵元。

**外部输入定理 D.11（Sturm-Liouville 与 Fourier-Hermite 完备性，QM-EXT-14）.** 正则 Sturm-Liouville 问题的本征函数在相应 $L^2$ 空间中完备；特别地，区间 Dirichlet Laplacian 的 sine 函数族完备，谐振子的 Hermite 函数族在 $L^2(\mathbb R)$ 中完备。

**外部输入定理 D.12（WKB 转折点连接公式，QM-EXT-15）.** 取
$m>0$。对光滑一维势阱的紧正则能区，若每条束缚轨道恰有两个简单
转折点，则 Airy 连接公式给出 Maslov 相位；当 $\hbar\to0$ 且指标
$n=n(\hbar)$ 使 $E_n(\hbar)$ 留在该能区时，它导出
$$
\int_{a(E_n(\hbar))}^{b(E_n(\hbar))}
\sqrt{2m(E_n(\hbar)-V(x))}\,dx
=\pi\hbar\left(n+\frac12\right)+O(\hbar^2).
$$
余项在该紧正则能区内理解；硬壁、高阶或并合转折点不属于此版本。

**外部输入定理 D.13（光学定理与 partial wave 展开，QM-EXT-16）.** 对短程势的三维弹性散射，在标准谱归一化和适当渐近完备假设下，散射振幅满足光学定理；中心势情形可按角动量通道展开为相移级数。

**外部输入定理 D.14（Friedrichs 扩张，QM-EXT-17）.** 稠定、闭、半有界二次型唯一对应一个半有界自伴算子；半有界对称算子有规范的 Friedrichs 自伴扩张。

**外部输入定理 D.15（Lindblad 生成元定理，QM-EXT-18）.** 有限维量子 Markov 半群的生成元具有 GKSL/Lindblad 形式；无限维情形需要额外定义域和闭性假设。

**外部输入定理 D.16（Uhlmann 定理，QM-EXT-19）.** 混合态保真度可由其纯化之间的最大重叠刻画；任意两个纯化之间的自由度由辅助空间上的等距或酉变换控制。

**外部输入定理 D.17（Kato 解析扰动理论，QM-EXT-20）.** 解析算子族的孤立谱子空间、谱投影和本征值在适当假设下具有解析或 Puiseux 型扰动展开；有限维非简并公式是其初等特例。

**外部输入定理 D.18（Stone-von Neumann 定理，QM-EXT-4）.** 有限自由度正则 Weyl 形式 CCR 的不可约强连续表示在标准正则性假设下酉等价；无限自由度场论中该结论不再成立。

**外部输入定理 D.19（绝热定理，QM-EXT-7）.** 对具有谱隙和足够光滑参数依赖的 Hamiltonian，慢变演化近似保持瞬时谱子空间，并产生动力学相位和 Berry 相位；无隙或交叉情形需额外假设。

**外部输入定理 D.20（散射波算子存在与渐近完备性，QM-EXT-8）.** 对短程势和适当谱假设，Møller 波算子存在并在连续谱子空间上给出散射算子；渐近完备性是更深的散射理论输入。

**外部输入定理 D.21（Lie--Trotter 乘积公式，QM-EXT-9）.** 设 $T,V$ 自伴，$\mathcal D(T)\cap\mathcal D(V)$ 稠密，且 $T+V$ 在该交定义域上本质自伴。则对每个 $t\in\mathbb R$，
$$
e^{-it\overline{(T+V)}}
=\operatorname{s-lim}_{n\to\infty}
\left(e^{-itT/n}e^{-itV/n}\right)^n.
$$
该强算子极限为路径积分离散化提供严格入口，但本身不蕴含逐点核收敛。
