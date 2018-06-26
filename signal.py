import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as csi_sig

class Signal():
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
        d={'time': x_, 'amp_s': noise_}
        df=pd.DataFrame(d)
        return df
    def sweep(self,start_f=0,end_f=5000):#扫频
        x_=self.scale()
        signal = csi_sig.chirp(x_, f0=start_f, f1=end_f, t1=self.time, method='linear')
        d={'time': x_, 'amp_s': signal}
        df=pd.DataFrame(d)
        return df
    def gauss_pulse(self,bandwith=0.05):
        x_=self.scale()
        signal = csi_sig.gausspulse(x_-(self.time/2), fc=self.frqnum,bw=bandwith,bwr=-6)
        d={'time': x_, 'amp_s': signal}
        df=pd.DataFrame(d)
        return df
    def impulse(self):
        x_=self.scale()
        signal=csi_sig.unit_impulse(len(x_),idx='mid')
        d={'time': x_, 'amp_s': signal}
        df=pd.DataFrame(d)
        return df
    def sawtooth(self,width=1):
        x_=self.scale()
        signal=csi_sig.sawtooth(2 * np.pi * self.cycle* x_,width)
        d={'time': x_, 'amp_s': signal}
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
        'time'；'amp_s'

        """  
        x_=self.scale()
        signal=[]
        for i in x_:
            if np.sin(2 * np.pi*f *i)>0:
                signal.append(1)
            else:
                signal.append(-1)
        signal =np.array(signal)
        d={'time': x_, 'amp_s': signal}
        df=pd.DataFrame(d)
        return df

    def sin_signal(self,A=1):
        x_=self.scale()
        sin_y=A*np.sin(2 * np.pi * self.frqnum * self.scale())
        d = {'time': x_, 'amp_s': sin_y}
        df=pd.DataFrame(d)
        return df
    

    def sin_with_noise(self):
        signal=self.sin_signal()
        nis=self.noise(A=100)
        signal.amp_s=signal.amp_s + nis.amp_s
        return signal

    def cos_signal(self):
        x_=self.scale()
        
        sin_y=np.cos(2 * np.pi * self.frqnum * self.scale())
        d = {'time': x_, 'amp_s': sin_y}
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
        self.fft_size = 2**12  #采样率 除以FFT长度，可以得到精确的信号频谱
    def fft_real(self,data): #实部FFT
        t=data.time
        d=data.amp_s
        ys =d[:self.fft_size]
        yf = np.fft.rfft(ys)/self.fft_size
        #freq = np.linspace(0,self.sampling_rate/2, self.fft_size/2+1)
        freq = np.linspace(0,self.sampling_rate/2, self.fft_size/2+1) # 0- f/2 的范围内 等分fft_size/2+1 份
        freqs = np.array(map(lambda x : x/1e3, freq))
        yfp = 20*np.log10(np.clip(np.abs(yf),1e-20,1e100)) #转换成db值
        d={'freq':freq,"amp_f":yfp}
        return pd.DataFrame(d)
    
    def fft(self,data):#FFT 返回一个未折叠的真实值
        t=data.time
        d=data.amp_s
        #d =d[:self.fft_size+1]
        print ('size:',len(data))
        fft_data=np.fft.fft(d,norm=None)
        #freq = np.linspace(0-self.sampling_rate/2,self.sampling_rate/2,len(fft_data))
        freq = np.fft.fftfreq(len(d),1/self.sampling_rate)#参数1 数据长度，参数二 频率步进

        #fft_data = 20*np.log10(np.clip(np.abs(fft_data),1e-20,1e100))#转换成db
        d={'freq':freq,"amp_f":fft_data}
        d=pd.DataFrame(d)
        d=d.sort_values('freq')
        return d
    def ifft(self,data):
        #f=data.freq
        #d=data.amp_f
        t=np.fft.ifft(data)
        return t

class IIR_Filter():
    def __init__(self, *args, **kwargs):
        pass
    def butter_filter(self,data):
        pass
        




'''
s=Signal()
s.FRQ='12340.24hz'
s.time=0.5
s1=s.cos_signal()
n1=s.noise(100)
c=Signal()
c.FRQ='351.12322HZ'
c.time=1
s2=c.gauss_pulse()

#s1=s.sweep().,BGBV B
#s1.amp_s=s1.amp_s *s2.amp_s
print (s1)

s=Signal()
s.FRQ='5HZ'
s.time=1
s1=s.sin_signal()
s.FRQ='15HZ'
s2=s.sin_signal()
s.FRQ='25HZ'
s3=s.sin_signal()

s1.amp_s=(1/4)*s1.amp_s+(1/4/3)*s2.amp_s+(1/4/5)*s3.amp_s

#s1.sin[10000:100000]=0.25
f =FastFourier()
f1=f.fft(s1)
print (f1[f1.amp_f==f1.amp_f.max()],len(f1))
plt.figure(1)
plt.plot(s1.time[0:],s1.amp_s[0:])
plt.figure(2)
mid=len(f1.freq)
#plt.plot(f1.freq[mid:mid+500],f1.amp_f[mid:mid+500])
plt.plot(f1.freq,f1.amp_f)
print (f1)

#plt.semilogx(f1.freq, f1.amp_f)
plt.margins(0, 0.1)
#plt.axvline(500, color='green')
plt.show()
'''