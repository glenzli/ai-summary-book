# 第二十三章：电磁耦合、规范变换与磁场

## 本章目标

本章介绍非相对论粒子在外电磁场中的 Hamiltonian、最小耦合、规范协变性、Landau 能级和 Aharonov-Bohm 效应的数学边界。

## 依赖前置知识

需要动量算子、酉变换、对易关系、谐振子和角动量。

## 23.1 最小耦合

**定义 23.1.** 电荷为 $q$ 的粒子在电磁势 $(\Phi,A)$ 中的 Hamiltonian 形式为
$$
H=\frac1{2m}(P-qA(X))^2+q\Phi(X).
$$
其中 $A(X)$ 和 $\Phi(X)$ 是乘法算子。记
$$
\Pi=P-qA(X)
$$
为动力学动量。

**外部输入定理 23.2（磁 Schrodinger 算子自伴性，QM-EXT-11）.** 在适当局部平方可积和下界假设下，磁 Schrodinger 算子可由闭半有界二次型定义为自伴算子。

## 23.2 规范协变性

**定义 23.3.** 规范变换由光滑实函数 $\chi$ 给出：
$$
A\mapsto A+\nabla\chi,\qquad
\Phi\mapsto \Phi-\partial_t\chi.
$$
波函数同时变换为
$$
\psi\mapsto e^{iq\chi}\psi.
$$

**命题 23.4.** 在时间无关规范变换下，动量最小耦合满足
$$
(P-q(A+\nabla\chi))e^{iq\chi}\psi
=e^{iq\chi}(P-qA)\psi.
$$

**证明.** 使用 $P=-i\nabla$：
$$
P(e^{iq\chi}\psi)
=-i(iq\nabla\chi\,e^{iq\chi}\psi+e^{iq\chi}\nabla\psi)
=e^{iq\chi}(q\nabla\chi\,\psi+P\psi).
$$
再减去 $q(A+\nabla\chi)e^{iq\chi}\psi$，$\nabla\chi$ 项相消，得到结论。$\square$

**推论 23.5.** 时间无关规范变换下 Hamiltonian 由酉共轭联系：
$$
H_{A+\nabla\chi}=e^{iq\chi}H_Ae^{-iq\chi}.
$$
因此谱不变。

## 23.3 匀强磁场与 Landau 能级

**设定 23.6.** 在二维平面上取匀强磁场 $B$ 垂直平面。动力学动量满足
$$
[\Pi_x,\Pi_y]=iqB.
$$

**命题 23.7.** 若 $qB>0$，Hamiltonian
$$
H=\frac1{2m}(\Pi_x^2+\Pi_y^2)
$$
形式上等价于频率 $\omega_c=qB/m$ 的谐振子，能级为
$$
E_n=\omega_c\left(n+\frac12\right).
$$

**证明.** 定义
$$
a=\frac{1}{\sqrt{2qB}}(\Pi_x+i\Pi_y),\qquad
a^*=\frac{1}{\sqrt{2qB}}(\Pi_x-i\Pi_y).
$$
由 $[\Pi_x,\Pi_y]=iqB$ 得 $[a,a^*]=1$。并且
$$
a^*a=\frac1{2qB}(\Pi_x^2+\Pi_y^2-qB).
$$
故
$$
H=\frac{qB}{m}\left(a^*a+\frac12\right).
$$
$\square$

## 23.4 Aharonov-Bohm 边界

**说明 23.8.** 在非单连通区域中，即使磁场 $B=\nabla\times A$ 在粒子可达区域为零，环路积分
$$
\oint A\cdot dx
$$
仍可给出可观测相位。这是 Aharonov-Bohm 效应。严格处理需要带洞区域上的自伴扩张、规范丛或边界条件分析，本书只记录其规范相位机制。

## 本章小结

电磁场通过最小耦合进入 Hamiltonian。规范变换不改变物理谱，而是由波函数相位变换实现。匀强磁场给出 Landau 能级；非单连通空间中的规范势可产生 Aharonov-Bohm 相位。

## 练习

**练习 23.1.** 证明磁场 $B=\nabla\times A$ 在规范变换 $A\mapsto A+\nabla\chi$ 下不变。

**练习 23.2.** 验证命题 23.7 中 $[a,a^*]=1$。

**练习 23.3.** 说明为什么规范相关的 $A$ 本身不是直接可观测量。

