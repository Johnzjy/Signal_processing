import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as csi_sig

class SigWave():
    def __init__(self,frq_input=None,time_input=1,sr_input=30000,ffts_input=512):
        self.FRQ_ = frq_input #输入频率
        self.time_ = time_input #原始信号时长
        self.sampling_rate_=sr_input #采样率
        self.fft_size_=ffts_input #fft长度  没用用到
        self.init()

    @property
    def FRQ(self): #频率
        if self.FRQ_ is None:
            return '50mz'
        else:
            return self.FRQ_
    @FRQ.setter
    def FRQ(self,value): 
        self.FRQ_=value
        self.init()
    @property
    def time(self):#时长
        return self.time_
    @time.setter
    def time(self,value):
        self.time_=value
        self.init()
        
    @property #FFt
    def fft_size(self):
        return self.fft_size_
    @property
    def sampling_rate(self):
        return self.sampling_rate_
    

    def init(self):
        self.period=1/self.frq_to_num() #周期
        self.cycle=self.time/self.period #时长内有多少个循环
        self.time_step=1/self.sampling_rate #采样时间间隔

    @property
    def frqnum(self):  #属性 将单位去掉 1kz =10000hz
        return self.frq_to_num()

    def scale(self): #生成一个时间轴
        list_=np.arange(0,self.time,self.time_step)
        return list_

    def noise(self,A=5): #在时间轴内生成一个噪声
        x_=self.scale()
        noise_=A*0.001*np.random.normal(-1.0,1.0,len(x_))
        d={'scale': x_, 'amp': noise_}
        df=pd.DataFrame(d)
        return df
    def sweep(self,start_f=0,end_f=5000):#扫频
        x_=self.scale()
        signal = csi_sig.chirp(x_, f0=start_f, f1=end_f, t1=self.time, method='linear')
        d={'scale': x_, 'amp': signal}
        df=pd.DataFrame(d)
        return df
    def gauss_pulse(self,bandwith=0.05):
        x_=self.scale()
        signal = csi_sig.gausspulse(x_-(self.time/2), fc=self.frqnum,bw=bandwith,bwr=-6)
        d={'scale': x_, 'amp': signal}
        df=pd.DataFrame(d)
        return df
    def impulse(self):
        x_=self.scale()
        signal=csi_sig.unit_impulse(len(x_),idx='mid')
        d={'scale': x_, 'amp': signal}
        df=pd.DataFrame(d)
        return df
    def sawtooth(self,width=1):
        x_=self.scale()
        signal=csi_sig.sawtooth(2 * np.pi * self.cycle* x_,width)
        d={'scale': x_, 'amp': signal}
        df=pd.DataFrame(d)
        return df
    def sin_window(self):
        x_=self.scale()
        sin_y=np.sin(2 * np.pi * self.frqnum * self.scale())/self.scale()
        d = {'scale': x_, 'amp': sin_y}
        df=pd.DataFrame(d)
        return df
    def square_signal(self,A=1,f=60.8):
        """
        产生一个方波

        Parameters
        ----------
        self: 
        A=1: 幅度 
        f=60000.8: 频率 

        Returns
        'time'；'amp'

        """  
        x_=self.scale()
        signal=[]
        for i in x_:
            if np.sin(2 * np.pi*f *i)>0:
                signal.append(1)
            else:
                signal.append(-1)
        signal =np.array(signal)
        d={'scale': x_, 'amp': signal}
        df=pd.DataFrame(d)
        return df

    def sin_signal(self,A=1):
        x_=self.scale()
        sin_y=A*np.sin(2 * np.pi * self.frqnum * self.scale())
        d = {'scale': x_, 'amp': sin_y}
        df=pd.DataFrame(d)
        return df
    

    def sin_with_noise(self):
        signal=self.sin_signal()
        nis=self.noise(A=100)
        signal.amp=signal.amp + nis.amp
        return signal

    def cos_signal(self):
        x_=self.scale()
        
        sin_y=np.cos(2 * np.pi * self.frqnum * self.scale())
        d = {'scale': x_, 'amp': sin_y}
        df=pd.DataFrame(d)
        return df
        
    def frq_to_num(self,): #转换数据类型 
        frq_str =self.FRQ
        unit = frq_str[-2:]
        num= frq_str[:-2]
        print(num)
        if num.isdigit():
            pass
        elif float(num):
            pass
        else:
            raise ValueError('the num is error')
            return None
        if unit.lower() == 'mz':
            #print ('1000000')
            return float(num)*1000000
        elif unit.lower() == 'kz':
            #print ('1000')
            return float(num)*1000 
        elif unit.lower() == 'gz':
            #print ('1000000000')
            return float(num)*1000000000
        elif unit.lower() == 'hz':
            #print ('0')
            return float(num)
        else:
            raise ValueError('the inputstr is error')
            return None


class FastFourier():
    """
    FFT运算

    Parameters
    ----------

    Returns
    -------

    """
    def __init__(self, *args, **kwargs):
    
        self.sampling_rate = 30000
        self.fft_size = 30000  #采样率 除以FFT长度，可以得到精确的信号频谱
    def fft_real(self,data): #实部FFT
        t=data.scale
        d=data.amp
        ys =d[:self.fft_size]
        yf = np.fft.rfft(ys)/self.fft_size
        #freq = np.linspace(0,self.sampling_rate/2, self.fft_size/2+1)
        freq = np.linspace(0,self.sampling_rate/2, len(d)/2+1) # 0- f/2 的范围内 等分fft_size/2+1 份
        freqs = np.array(map(lambda x : x/1e3, freq))
        yfp = 20*np.log10(np.clip(np.abs(yf),1e-20,1e100)) #转换成db值
        d={'scale':freq,"amp":yfp}
        return pd.DataFrame(d)
    
    def fft(self,data):#FFT 返回一个未折叠的真实值
        t=data.scale
        d=data.amp
        #d =d[:self.fft_size+1]
        print ('size:',len(data))
        fft_data=np.fft.fft(d,norm=None)
        #freq = np.linspace(0-self.sampling_rate/2,self.sampling_rate/2,len(fft_data))
        freq = np.fft.fftfreq(len(d),1/self.sampling_rate)#参数1 数据长度，参数二 频率步进

        #fft_data = 20*np.log10(np.clip(np.abs(fft_data),1e-20,1e100))#转换成db
        d={'scale':freq,"amp":fft_data}
        d=pd.DataFrame(d)
        d=d.sort_values('freq')
        return d
    def ifft_real(self,data):
        f=data.scale
        d=data.amp
        t=np.fft.ifft(data)
        return t

class IIR_Filter():
    def __init__(self, *args, **kwargs):
        pass
    def butter_filter(self,data):
        pass
        





s=SigWave()
s.FRQ='20.24hz'
s.time=0.5
s1=s.sweep(start_f=12000,end_f=500)
n1=s.noise(100)
c=SigWave()
c.FRQ='6370.12322HZ'
c.time=1
s2=c.sin_signal()
#s1.amp=s1.amp *s2.amp
#s1=s.sweep()
#s1.amp=s1.amp *s2.amp
#print (s1)
'''
s=SigWave()
s.FRQ='5HZ'
s.time=1
s1=s.sin_signal()
s.FRQ='15HZ'
s2=s.sin_signal()
s.FRQ='25HZ'
s3=s.sin_signal()

s1.amp=(1/4)*s1.amp+(1/4/3)*s2.amp+(1/4/5)*s3.amp

#s1.sin[10000:100000]=0.25
'''
f =FastFourier()
f1=f.fft_real(s1)

print ('\n 最大频率：\n \n',f1[f1.amp==f1.amp.max()],len(f1),'\n\n')
plt.figure(1)
plt.plot(s1.scale[0:],s1.amp[0:])
plt.figure(2)
mid=len(f1.scale)
#plt.plot(f1.freq[mid:mid+500],f1.amp[mid:mid+500])
plt.plot(f1.scale,f1.amp)
#print (f1)

#plt.semilogx(f1.freq, f1.amp)
#plt.margins(0, 0.1)
#plt.axvline(500, color='green')
plt.show()
