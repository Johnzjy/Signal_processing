# signal processing
---
``【信号处理 python库】``
***
* [Signal](#Signal)<font color="#dd0000">  &nbsp;=> &nbsp;</font> { &nbsp; `` 产生信号 sin cos 方波 扫频``&nbsp; } 
* [FastFourier（快速傅里叶 FFT）](#FastFourier )<font color="#dd0000">  &nbsp;=> &nbsp;</font> { &nbsp; `` FFT ;IFFT;``&nbsp; } 
***

### Siganl

* Signal &nbsp; <'class'> &nbsp;&nbsp; { &nbsp; 产生信号 &nbsp; }  
  * Parameters
    FRQ &nbsp; : &nbsp; 频率 （hz kz GZ）
    time ：时长 （s）
    sampling_rate： 采样率
  * Returns &nbsp;: &nbsp;{'time'；'amp_s' }&nbsp;&nbsp;< dataform >
  * sin_signal &nbsp; <'funcation'>&nbsp; 
   > 正弦曲线，频率由signal.Freq
     时间轴由scale() 决定
  * sin_with_noise &nbsp; <'funcation'>&nbsp; 
  >正弦曲线伴随随机噪声，频率由signal.Freq
  * cos_signal &nbsp; <'funcation'>&nbsp; 
  >余弦曲线，频率由signal.Freq
     时间轴由scale() 决定  
  * sweep &nbsp; <'funcation'>&nbsp; 
  >扫频，设置起始频率=>start_f=0,终止频率=>end_f=5000
     时间轴由scale() 决定
  * noise &nbsp; <'funcation'>&nbsp; 
  >产生白噪声 幅度设置为 5 
     时间轴由scale() 决定
  * gauss_pulse &nbsp; <'funcation'>&nbsp; 
  >高斯脉冲 bandwith 设置带宽
     时间轴由scale() 决定
  * impulse &nbsp; <'funcation'>&nbsp; 
  >单位脉冲 幅度设置为 5 
     时间轴由scale() 决定
   * sawtooth &nbsp; <'funcation'>&nbsp; 
  >锯齿波《三角波》 width 设置倾斜角 0为下坡，1为上坡  0.5为等腰三角形  
     时间轴由scale() 决定

  * square_signal  &nbsp; <'funcation'>&nbsp; 
  >方波 频率需要单独设置 f=XXhz(int) 
     时间轴由scale() 决定

### FastFourier
* FastFourier &nbsp; <'class'> &nbsp;&nbsp; { &nbsp; 快速傅里叶 &nbsp; }  

  * fft   &nbsp; <'funcation'>&nbsp;
  > 返回一个未折叠的真实值
  input 格式 必须为 {'time'；'amp_s' }< dataform >
  * ifft   &nbsp; <'funcation'>&nbsp;
  > 傅里叶逆变换
  input 格式 必须为 {'time'；'amp_s' }< dataform >
  * fft_real   &nbsp; <'funcation'>&nbsp;
  > 傅里叶变换 real 值
  input 格式 必须为 {'time'；'amp_s' }< dataform >