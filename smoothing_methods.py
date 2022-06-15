##################################################
# Smoothing Methods (Holt-Winters)
##################################################
import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings("ignore")

############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data

y= y["co2"].resample("MS").mean()#Burada co2 değeleri için resample ile yeniden örneklemesi yapıldıktan sonra
##mean ile ortalamaları alınıyor.
#Bu işlem sayesinde y artık aylık olarak ortalama co2 değerlerini göstermektedir

#Bu işlem sornasında 1958-06-01 değeri gibi bazı değerlerde eksik değerler gözlemlenebilir.
##Bu eksik değelerin yerine medyan veya ortalama gibi değerler atamak yerine
###Önceki aya ya da sonraki aya ait değerlerle doldurulur.

y.isnull().sum()
#Out[9]: 5 Görüleceği üzere y verisinde 5 tane eksik gözlem var

y = y.fillna(y.bfill())
#Dersek bu method ile eksik değerlerden bir sonraki değeri eksik değerlere atamış oluruz

y.plot(figsize=(15,6))
plt.show()
#Burada grafikten trend ve mevsimsellik özelliklerini açık bir şekilde görebiliyoruz

############################
# Holdout
############################

#Holdout yöntemi ile verimizi train ve test olarak ayırıyoruz

train = y[:"1997-12-01"]
len(train) # 478 ay olarak train setimizi seçmiş oluyoruz.
#Öncelikle modelimizi train üzerinde kurduktan sonra başarısını kalan test kısmıyla başarısını test edeceğiz

test =y["1998-01-01":]
len(test) # 48 olarak test setimizi seçmiş oluyoruz.


##################################################
# Zaman Serisi Yapısal Analizi
##################################################

# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):

    #"H0": Non-stationary
    #"H1": Stationary
    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value <0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value : {round(p_value,3)})")
    else:
        print(F"Result: Stationary (H1 : stationary, p-value : {round(p_value,3)})")

is_stationary(y)

#P value değeri 0.05den olayı büyük olduğu için H0 doğrudur.
##Yani Serimiz durağan değildir. Biz bunu grafiği yorumlayarakta erişmiştik.


# Zaman Serisi Bileşenleri ve Durağanlık Testi

def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)

#Bu foksiyon sayesinde Trend, Mevsimlik ve artıkların nasıl şekillendiği görme imkanı verecektir.
#Durağın argümanınıda true yaparsak console kısmında çıktımızı verecektir.


##################################################
# Single Exponential Smoothing
##################################################

# SES = Level

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)
#Smoothing level ile alfa sayısını belirlemiş oluyoruz.
##Bu ileride diğer parametrelerde işin içine girdiğinde önemi daha iyi anlaşılacaktır.
###Biz el ile alfa parametresini belirlemezsek method en iyi başarı için otomatik olarak belirleyecektir.
train
y_pred = ses_model.forecast(48) #Makine modellerinin haricinde tahmin için predict kullanmıyoruz.
##Holdout yönteminde forecast kullanıyoruz.
###İçerisine yazdığımız sayı ise kaç adımlık tahmin yapılacağını belirliyor.

#Hatamızı kontrol etmek adına mae metodunu kullanarak inceliyoruz.

mean_absolute_error(test,y_pred)
#Out[29]: 5.706393369643809 ortalama hatamız bu şekilde oluyor

#Tahminlerde tekrar edenlerin değerlerin olmasının sebebi ise görsel üzerinden görüleceği üzere
##veride trend ve mevsimsellik olmasıdır.

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show()
#bu grafikte yeşil olanlar tahmin ettiğimiz değerlerdir.

def plot_co2(train,test,y_pred,title):
    mae= mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True,label="TRAIN",title=f"{title},MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST",figsize=(6,4))
    y_pred.plot(legend=True,label="PREDICTION")
    plt.show()

plot_co2(train,test,y_pred,"Single Exponential Smoothing")

ses_model.params
#smoothing_level 0.5 yerine diğer parametreleri denememiz lazım
##Hiperparametre optimizasyonu ile bunu yapıyor olacağız

############################
# Hyperparameter Optimization
############################

def ses_optimizer(train, alphas, step=48):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.8, 1, 0.01)

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

ses_optimizer(train, alphas)

best_alpha, best_mae = ses_optimizer(train, alphas)

############################
# Final SES Model
############################

best_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = best_model.forecast(48)
#Son bulduğumuz parametreler sonrasında modelimizi tekrar oluşturuyoruz
##Tahminleri tekrar yapdıktan sonra
###Daha sonra plot_co2 fonksyonumuz ile değişimi gözlemliyoruz
plot_co2(train,test,y_pred,"Single Exponential Smoothing")
#Görüldüğü üzere tahminler gerçek değerlere biraz daha yaklaşmış oldular.
##Single Exp Smoothingin durağan verilerde daha tutarlı olduğu grafiktende görülmüş oluyoruz.


##################################################
# Double Exponential Smoothing (DES)
##################################################
#Artık değerler gerçek değerler ile tahmin arasındaki farktır. Hatalar olarakda isimlendirilebilir.
# DES: Level (SES) + Trend

#Toplamsal model
# y(t) = Level + Trend + Seasonality + Noise
#Çarpımsal model
# y(t) = Level * Trend * Seasonality * Noise

#Mevsimsellik ve artık bileşenleri trendden bağımsız gözüküyorsa seri toplamsaldır.
##Ayrıca grafikten mevsimsellik ve artıklar 0'ın etrafında dağılıyorsada toplamsal diyebiliriz.

#Mevsimsellik ve artık bileşenler trende göre bağımlı gözüküyorsa seri çarpımsaldır.

#Trend etkisini göz önünde bulundurarak üssel düzenleme yapar.
#Temel yaklasım aynıdır. SES'e ek olarak trend de dikakte alınır

#Trend içeren ama mevsimsellik içermeyen tek değişkenli seriler için uygundur.

des_model = ExponentialSmoothing(train,trend="add").fit(smoothing_level=0.5,
                                                        smoothing_trend=0.5)

#Smoothing trend ise trendin yakın trende mi eski trende mi ağırlık verilmesi gerekiyor bunu gösteriyor.

y_pred = des_model.forecast(48)

plot_co2(train,test,y_pred,"Double Exponential Smoothing")
#Optimazyon yapılmamasına rağmen level ve Trend konusunda SES'e göre daha iyi noktadayız.
##Mevsimsellik uyumu olmadığından dolayı yine de başarısız sayılabilir bir yöntemdir.

############################
# Hyperparameter Optimization
############################


def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)
#Bu parametre optimizasyonları ile hata oranın 1.7411 kadar çekebildik


############################
# Final DES Model
############################

final_des_model = ExponentialSmoothing(train,trend="add").fit(smoothing_level=best_alpha,
                                                              smoothing_trend=best_beta)
y_pred = final_des_model.forecast(48)

plot_co2(train,test,y_pred,"Double Exponential Smoothing")
#Mevsimsellik uyumlu olmadığından dolayı dalgalanmaları yakalayamıyoruz.

##################################################
# Triple Exponential Smoothing (Holt-Winters)
##################################################

# TES = SES + DES + Mevsimsellik
# En gelişmiş smoothing yöntemidir.
# Bu yöntem dinamik olarak level,trend ve mevsimsellik etkilerini değerlendirerek tahmin yapmaktadır.
# Trend ve/veya mevsimsellik içeren tek değişkenli serilerde kullanılabilir.

tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)
y_pred =tes_model.forecast(48)

plot_co2(train,test,y_pred,"Triple Exponential Smoothig")


############################
# Hyperparameter Optimization
############################

alphas = betas = gammas = np.arange(0.20, 1, 0.10)

abg = list(itertools.product(alphas, betas, gammas))


def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


############################
# Final TES Model
############################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")