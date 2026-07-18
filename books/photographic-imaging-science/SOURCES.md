# 资料源

正文全部采用本书自己的推导和表述。标准、白皮书和厂商技术页用于固定术语、公式
版本和具体实现；产品宣传不作为一般物理规律的证据。

## 成像物理与传感器

- James R. Janesick, *Photon Transfer*, SPIE Press, 2007.
- James R. Janesick, *Scientific Charge-Coupled Devices*, SPIE Press, 2001.
- Gerald C. Holst and Terrence S. Lomheim, *CMOS/CCD Sensors and Camera
  Systems*, 2nd ed., JCD Publishing/SPIE Press, 2011.
- Eric R. Fossum, “CMOS Image Sensors: Electronic Camera-on-a-Chip,”
  *IEEE Transactions on Electron Devices* 44(10), 1997, pp. 1689--1698.
- Albert J. P. Theuwissen, *Solid-State Imaging with Charge-Coupled Devices*,
  Kluwer, 1995.
- European Machine Vision Association, [EMVA Standard 1288, Release 4.0](https://www.emva.org/wp-content/uploads/EMVA1288General_4.0Release.pdf), 2021.
- Hamamatsu Photonics, [Photon Number Resolving Camera Technology](https://camera.hamamatsu.com/content/dam/hamamatsu-photonics/sites/documents/99_SALES_LIBRARY/sys/SCAS0149E_qCMOS_whitepaper.pdf),
  sections on BSI, DTI, QE, noise and MTF.
- Sony Semiconductor Solutions, [Pregius/Pregius S Global Shutter Technology](https://www.sony-semicon.com/en/technology/industry/pregius.html).
- Brian Guenter et al., [“Highly curved image sensors: a practical approach for
  improved optical performance”](https://www.microsoft.com/en-us/research/publication/highly-curved-image-sensors-practical-approach-improved-optical-performance/),
  *Optics Express* 25(12), 2017, pp. 13010--13023.

## 曝光、ISO 与计算成像

- ISO, [ISO 12232:2019, Photography -- Digital still cameras -- Determination
  of exposure index, ISO speed ratings, standard output sensitivity, and
  recommended exposure index](https://www.iso.org/standard/73758.html), edition 3,
  together with [Amendment 1:2020, determination of encoding-relative sensitivity
  (ERS)](https://www.iso.org/standard/79168.html). The base standard was confirmed
  in 2024 and remained published when checked in July 2026.
- Samuel W. Hasinoff et al., [“Burst photography for high dynamic range and
  low-light imaging on mobile cameras”](https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/),
  *ACM Transactions on Graphics* 35(6), 2016.
- Paul E. Debevec and Jitendra Malik, [“Recovering High Dynamic Range Radiance
  Maps from Photographs”](https://people.eecs.berkeley.edu/~malik/papers/debevec-malik97.pdf),
  SIGGRAPH 1997.
- Sony Semiconductor Solutions, [Hybrid Frame-HDR](https://www.sony-semicon.com/en/technology/mobile/hf-hdr.html),
  for a concrete combination of conversion-gain and multi-frame HDR.
- Canon, [Dual Gain Output Sensor White Paper](https://downloads.canon.com/cinemaeos/DGO-Sensor-White-Paper.pdf),
  2020, as a concrete dual-readout implementation.

## Log、RAW 与色彩编码

- ARRI, [ARRI LogC4 Logarithmic Color Space Specification](https://www.arri.com/resource/blob/278790/bea879ac0d041a925bed27a096ab3ec2/2022-05-arri-logc4-specification-data.pdf).
- ARRI, [Dynamic Range White Paper](https://www.arri.com/resource/blob/295460/e10ff8a5b3abf26c33f8754379b57442/2022-09-28-arri-dynamic-range-whitepaper-data.pdf).
- Sony, [Help Guide for Creators: Base ISO](https://helpguide.sony.net/di/pp/v1/en/contents/TP1000756719.html)
  and [Gamut](https://helpguide.sony.net/di/pp/v1/en/contents/TP1000756714.html).
- Adobe, [Digital Negative (DNG) Specification 1.7.1.0 landing page](https://helpx.adobe.com/camera-raw/digital-negative.html),
  September 2023 specification revision; official page last updated June 2026
  and checked July 2026.
- Apple, [Apple ProRes RAW White Paper](https://www.apple.com/final-cut-pro/docs/Apple_ProRes_RAW.pdf).
- Charles Poynton, *Digital Video and HD: Algorithms and Interfaces*, 2nd ed.,
  Morgan Kaufmann, 2012.
- R. W. G. Hunt and M. R. Pointer, *Measuring Colour*, 4th ed., Wiley, 2011.

## 几何光学、Fourier 光学与镜头

- Max Born and Emil Wolf, *Principles of Optics*, 7th expanded ed., Cambridge
  University Press, 1999.
- Joseph W. Goodman, *Introduction to Fourier Optics*, 4th ed., W. H. Freeman,
  2017.
- Warren J. Smith, *Modern Optical Engineering*, 4th ed., McGraw-Hill, 2008.
- Rudolf Kingslake and R. Barry Johnson, *Lens Design Fundamentals*, 2nd ed.,
  Academic Press, 2010.
- Virendra N. Mahajan, *Optical Imaging and Aberrations*, SPIE Press.
- H. Angus Macleod, *Thin-Film Optical Filters*, 4th ed., CRC Press, 2010.
- Carl Zeiss Camera Lens Division, [How to Read MTF Curves](https://lenspire.zeiss.com/photo/app/uploads/2018/04/Article-MTF-2008-EN.pdf).
- ISO, [ISO 12233:2024, Digital cameras -- Resolution and spatial frequency
  responses](https://www.iso.org/standard/88626.html), edition 5.
- Nikon, [Aberration Correction](https://www.nikon.com/company/technology/technology_fields/optics/aberration_correction/)
  and [Resin for Phase Fresnel Lenses](https://www.nikon.com/company/technology/technology_fields/materials/phase_fresnel_lens/).
- Nikon Research Report Vol. 4 (2022), [chromatic-aberration correction with
  glass maps and partial dispersion](https://www.nikon.com/company/technology/nrr/pdf/nrr_vol4_2022_04_e.pdf).
- Canon, [Interchangeable Lens Technologies](https://global.canon/en/technology/canon-tech/tech/iclenses/)
  and [Fluorite Lenses](https://global.canon/en/c-museum/special/exhibition2.html).
- 日本 Zeon, [Cyclo Olefin Polymer optical resin resources](https://www.zeon.co.jp/business/enterprise/resin/cop/).

## 标准边界

ISO 12232:2019/Amd 1:2020、ISO 12233:2024、CIE 色度学标准和若干编码标准的完整
规范受版权或获取条件约束。本书只重述主线所需定义和可公开核验部分，不复制标准
文本。标准中的合格性测试、容差和设备认证不属于本书内部闭包。
