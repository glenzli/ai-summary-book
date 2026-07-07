# 附录 E：supersymmetry、spinors 和 Clifford algebras

## 目标

本附录固定十维 spinor 和 supersymmetry 基本记号。

## E.1 Clifford algebra

**定义 E.1.** 在签名 $(1,D-1)$ 下，gamma matrices 满足
$$
\{\Gamma^\mu,\Gamma^\nu\}=2\eta^{\mu\nu}.
$$

偶数维中 chirality operator 定义为
$$
\Gamma_* = c_D\,\Gamma^0\Gamma^1\cdots\Gamma^{D-1},
$$
其中常数 $c_D$ 由 $\Gamma_*^2=1$ 固定。Weyl spinors 是 $\Gamma_*$ 的本征空间元素。

十维 Lorentzian signature 中可同时施加 Majorana 与 Weyl 条件，因此 type IIB 有同 chirality 的两个 Majorana-Weyl supersymmetry parameters，type IIA 有相反 chirality 的两个参数。

## E.1A Little group spinors

Massless 十维粒子的 little group 为 $SO(8)$。其三个八维 irreducible representations 通常记为
$$
8_v,\qquad 8_s,\qquad 8_c.
$$
RNS 与 GS spectrum matching 的核心是 bosonic transverse oscillators 给出 $8_v$，fermionic zero modes 给出 $8_s$ 或 $8_c$，并由 triality 解释自由度匹配。

## E.2 Supersymmetry algebra

**定义 E.2.** 平坦时空 supersymmetry algebra 的基本形式为
$$
\{Q_\alpha,Q_\beta\}=(\Gamma^\mu C^{-1})_{\alpha\beta}P_\mu+\text{central charges}.
$$

**注 E.3.** BPS bounds 来自该反对易子的正性和 central charge 项。

## E.3 BPS bounds

若 supersymmetry algebra 在某个 charge sector 中可写成
$$
\{Q,Q^\dagger\}\sim M-|Z|,
$$
正性给出
$$
M\ge |Z|.
$$
饱和 $M=|Z|$ 的态被部分 supercharges 湮灭，形成短 multiplets。由于短 multiplet 的维数受表示论保护，BPS 态常可跨耦合常数比较。

D-branes、black branes 和 wrapped branes 的质量-电荷关系都可按这一原则理解。正文中的 BPS entropy 计算依赖的正是这种 protected counting。

## E.4 Supersymmetry variations

Supergravity background 保持 supersymmetry 的条件是 fermionic variations 消失：
$$
\delta\psi_\mu=0,\qquad \delta\lambda=0
$$
以及可能的 gaugino variation。Calabi-Yau compactification、flux compactification 和 brane solution 的 supersymmetry 条件都可视为这些方程的几何化。

本书不列出所有 type II/heterotic supergravity variations 的完整系数；正文只使用它们推出 holonomy、ISD flux 或 BPS projector 的结构性后果。
