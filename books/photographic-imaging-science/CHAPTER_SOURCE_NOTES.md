# 逐章来源注释

本表说明外部资料在正文中的用途。书内公式能直接推导者仍给出背景专著，但证明责任
由正文承担；工艺细节、标准定义和具体厂商实现作为外部输入。

| 章节 | 主要来源 | 外部输入与边界 |
|---|---|---|
| 0 | Holst--Lomheim；Goodman | 成像链分层由本书组织；完整相机工程不在序章展开 |
| 1 | Born--Wolf；Smith | 辐射度学与近轴像面照度；实际镜头暗角需具体瞳和机械模型 |
| 2 | Fossum；Janesick；Holst--Lomheim | 硅吸收、PPD 工艺为器件物理输入；Beer--Lambert、响应度和 Poisson 稀疏化书内推导 |
| 3 | Fossum；Janesick；EMVA 1288 | 列 ADC 电路细节外置；$kT/C$、线性读出、噪声带宽和量化模型书内推导 |
| 4 | EMVA 1288；Janesick, *Photon Transfer* | 标准设备容差外置；成对平场、DSNU/PRNU、PTC 拟合和动态范围口径书内推导 |
| 5 | ISO 12232:2019/Amd 1:2020；Sony Base ISO；Canon DGO | 标准合格性文本不复制；增益位置模型和 DCG 权衡书内计算 |
| 6 | Sony Pregius S；Hamamatsu qCMOS；Guenter et al. 2017 | BSI/DTI/堆栈/曲面具体制造为外部输入；时序畸变书内推导 |
| 7 | Hunt--Pointer；Poynton；DNG specification | 相机光谱响应与色彩标定实践外置；采样混叠和同色异谱书内证明 |
| 8 | Hasinoff et al. 2016；Debevec--Malik 1997；Sony HF-HDR | 具体配准/融合算法外置；独立堆栈与加权估计书内推导 |
| 9 | ARRI LogC4 specification；ARRI Dynamic Range；Sony creator guide | 厂商常数和 EI 行为按具体系统引用；分段 Log 连续性与量化误差书内证明 |
| 10 | Adobe DNG 1.7.1；Apple ProRes RAW | 标签语义和码流规范外置；位打包、码率与黑白电平计算书内完成 |
| 11 | Born--Wolf；Smith；Kingslake--Johnson | 真实处方数值追迹外置；近轴矩阵、主平面分解和厚透镜算例书内完成 |
| 12 | Goodman；Zeiss MTF notes | 标量 Fourier 模型为外部物理框架；OTF 自相关书内证明，圆瞳经典结果注明边界 |
| 13 | Mahajan；Kingslake--Johnson | Seidel/Zernike 体系为外部经典框架；RMS--Strehl 小误差关系在正文推导 |
| 14 | Nikon Research Report 2022；Canon Fluorite；Zeon COP | 材料目录与制造性质外置；Sellmeier 用法、消色差和二级光谱残差书内推导 |
| 15 | Born--Wolf；Macleod；Nikon PF；Canon lens technologies | 具体制造外置；量化 blaze 效率与薄膜特征矩阵书内推导 |
| 16 | Smith；Kingslake--Johnson | 复杂变焦凸轮和防抖控制外置；两组光焦度、景深和抖动尺度书内推导 |
| 17 | Zeiss MTF notes；ISO 12233:2024（标准边界） | 标准容差与合格性实现外置；斜边有限差分、尺度换算和不确定度模型书内推导 |

## 一手资料使用原则

厂商资料只支持其明确披露的实现。例如 Sony Pregius S 可作为“存在背照堆栈全局
快门结构”的来源，不能据此推出所有堆栈传感器的读噪；Nikon PF 资料可支持衍射元件
与普通折射元件色散相反及可能出现 PF flare，不能替代某支镜头的样品测试。

## 图版来源与证据等级

正文 48 幅 SVG 全部由本项目的 `figures/generate_figures.py` 原创生成，没有复制
厂商剖面图、标准测试靶或论文插图。结构图只表达层次、方向和变量关系；解析曲线由
正文模型取示意参数绘制，不作为任何相机或镜头的实测证据。若未来加入外部照片或
实测图，必须在图注中登记器材、条件、数据处理与许可来源，并与现有机制图分开编号。
