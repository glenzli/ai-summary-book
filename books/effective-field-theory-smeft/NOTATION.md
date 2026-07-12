# 符号约定

本书默认自然单位 $\hbar=c=1$，度规号差为 $(+,-,-,-)$。

## 尺度与展开

- $M_{\rm gap}$：从所选低能运动学域到最近遗漏重 pole/threshold 的物理尺度；它控制局域展开的解析边界。
- $\Lambda_{\rm ref}>0$：把有量纲 Wilson 系数写成 $C_i^{(d)}/\Lambda_{\rm ref}^{d-4}$ 时选取的参考尺度；单独改变它只是 Wilson 坐标重标度。
- $\mu$：重整化尺度；$\mu_{\rm match}$：施加阈值匹配条件时选取的重整化尺度。二者都不是新的物理阈值。
- $Q$：由给定外态、phase-space bin 和全部独立硬不变量定义的运动学尺度；单个质心能量未必足以代替它。
- $\epsilon_{\rm kin}=Q/M_{\rm gap}$：运动学局域展开参数。
- $\epsilon_i^{(d)}=|C_i^{(d)}|(Q/\Lambda_{\rm ref})^{d-4}$：给定算符插入的粗略大小；实际还需耦合、helicity 和群论因子。
- $\epsilon_{\rm loop}$：所选微扰方案的圈参数，典型为 $g^2/(16\pi^2)$，有大对数时还需检查 $\epsilon_{\rm loop}|\log(\mu_1/\mu_2)|$。
- $\Lambda$：仅在明确采用单重尺度简写 $M_{\rm gap}=\Lambda_{\rm ref}=\Lambda$ 时使用；它不代表 $\mu$ 或 $\mu_{\rm match}$。
- $d$：算符的质量维数。
- $C_i^{(d)}$：维数 $d$ 算符 $\mathcal O_i^{(d)}$ 的 Wilson 系数。
- $\Box=\partial_\mu\partial^\mu$：在本书 $(+,-,-,-)$ 度规下的 d'Alembertian；作用到复合算符时，其作用域由紧邻的右侧表达式决定。

## EFT 拉氏量

- $\mathcal L_{\mathrm{UV}}$：高能理论拉氏量。
- $\mathcal L_{\mathrm{EFT}}$：低能有效拉氏量。
- $\mathcal L_{\mathrm{SM}}$：标准模型拉氏量。
- $\mathcal L_{\mathrm{SMEFT}}$：标准模型有效场论拉氏量。
- $\mathcal O_i^{(d)}$：质量维数为 $d$ 的局域规范不变算符。

## Hermiticity 与结构计数

- $\mathfrak H_d$：维数 $d$、在 Hermitian conjugation 下闭合的自伴算符族。若某固定分量满足 $\mathcal O^\dagger=\mathcal O$，其系数为实数；若 dagger 置换 flavor 指标，则 Wilson 张量满足对应 Hermiticity 关系。
- $\mathfrak N_d$：从每个互不相同的非自伴对 $\{\mathcal O,\mathcal O^\dagger\}$ 中选取一个代表的集合。Hermitian 拉氏量必须写成 $C\mathcal O+C^*\mathcal O^\dagger$。
- **结构计数**：非自伴 dagger pair 只计一个代表；**Wilson 实参数计数**：计入复系数的实部、虚部以及 flavor/Hermiticity 约束。Weinberg 的“一个类型”和 Warsaw 的“59 个结构”都采用前一种口径。

## 标准模型群与场

- $G_{\mathrm{SM}}=SU(3)_c\times SU(2)_L\times U(1)_Y$。
- $G_{\mu\nu}^A$、$W_{\mu\nu}^I$、$B_{\mu\nu}$：三类规范场强。
- $q,\ell$：左手 quark/lepton 双重态。
- $u,d,e$：右手 singlet Weyl 或等价的手征场。
- $H$：Higgs 双重态，$\widetilde H=i\sigma^2 H^\ast$。
- $v$：电弱真空期望值。

## 状态标签

- **书内推导**：本书直接推出。
- **推导说明**：路线完整但压缩标准计算。
- **外部输入**：引用文献或教材。
- **研究边界**：仍需版本化或超出当前闭包。
