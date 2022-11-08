# program to slice signal into layers
# the signal can be raw/equalized/ bandpassed/ denoised

def slice_signal_exp_25(signal):
    signal_layer = []
    signal_layer_1 = signal[44100*0:44100*3]
    signal_layer_2 = signal[int(44100*8.4):int(44100*11.4)]
    signal_layer_3 = signal[int(44100*17.2):int(44100*20.2)]
    signal_layer_4 = signal[int(44100*26):int(44100*29)]
    signal_layer_5 = signal[int(44100*34.7):int(44100*37.7)]
    signal_layer_6 = signal[int(44100*43.4):int(44100*46.4)]
    signal_layer_7 = signal[int(44100*52.1):int(44100*55.1)]
    signal_layer_8 = signal[int(44100*60.9):int(44100*63.9)]
    signal_layer_9 = signal[int(44100*69.7):int(44100*72.7)]
    signal_layer_10 = signal[int(44100*78.5):int(44100*81.5)]
    signal_layer_11 = signal[int(44100*87):int(44100*90)]
    signal_layer_12 = signal[int(44100*95.7):int(44100*98.7)]
    signal_layer_13 = signal[int(44100*104.5):int(44100*107.5)]
    signal_layer_14 = signal[int(44100*113.2):int(44100*116.7)]
    signal_layer_15 = signal[int(44100*122):int(44100*125)]
    signal_layer_16 = signal[int(44100*130.8):int(44100*133.8)]
    signal_layer_17 = signal[int(44100*139.4):int(44100*142.4)]
    signal_layer_18 = signal[int(44100*148.2):int(44100*151.2)]
    signal_layer_19 = signal[int(44100*156.7):int(44100*159.7)]
    signal_layer_20 = signal[int(44100*165.6):int(44100*168.6)]
    signal_layer_21 = signal[int(44100*174.8):int(44100*177.8)]
    signal_layer_22 = signal[int(44100*182.9):int(44100*185.9)]
    signal_layer_23 = signal[int(44100*191.6):int(44100*194.6)]
    signal_layer_24 = signal[int(44100*200.4):int(44100*203.4)]
    signal_layer_25 = signal[int(44100*209.1):int(44100*212.1)]
    signal_layer_26 = signal[int(44100*217.8):int(44100*220.8)]
    signal_layer_27 = signal[int(44100*226.5):int(44100*229.5)]
    signal_layer_28 = signal[int(44100*235.2):int(44100*238.2)]
    signal_layer_29 = signal[int(44100*243.8):int(44100*246.8)]
    signal_layer_30 = signal[int(44100*252.8):int(44100*255.8)]
    signal_layer_31 = signal[int(44100*261.2):int(44100*264.2)]
    signal_layer_32 = signal[int(44100*270.1):int(44100*273.1)]
    signal_layer_33 = signal[int(44100*278.7):int(44100*281.7)]
    signal_layer_34 = signal[int(44100*287.5):int(44100*290.5)]
    signal_layer_35 = signal[int(44100*296.2):int(44100*299.2)]
    signal_layer_36 = signal[int(44100*304.8):int(44100*307.8)]
    signal_layer_37 = signal[int(44100*313.55):int(44100*316.55)]
    signal_layer_38 = signal[int(44100*322.2):int(44100*325.2)]
    signal_layer_39 = signal[int(44100*331):int(44100*334)]
    signal_layer_40 = signal[int(44100*339.7):int(44100*342.7)]
    signal_layer_41 = signal[int(44100*348.5):int(44100*351.5)]
    signal_layer_42 = signal[int(44100*357.1):int(44100*360.1)]
    signal_layer_43 = signal[int(44100*365.8):int(44100*368.8)]
    signal_layer_44 = signal[int(44100*374.5):int(44100*377.5)]
    signal_layer_45 = signal[int(44100*383.3):int(44100*386.3)]
    signal_layer_46 = signal[int(44100*391.95):int(44100*394.95)]
    signal_layer_47 = signal[int(44100*400.6):int(44100*403.6)]
    signal_layer_48 = signal[int(44100*409.4):int(44100*412.4)]
    signal_layer_49 = signal[int(44100*418.1):int(44100*421.1)]
    signal_layer_50 = signal[int(44100*426.8):int(44100*429.8)]

    signal_layer.append(signal_layer_1)
    signal_layer.append(signal_layer_2)
    signal_layer.append(signal_layer_3)
    signal_layer.append(signal_layer_4)
    signal_layer.append(signal_layer_5)
    signal_layer.append(signal_layer_6)
    signal_layer.append(signal_layer_7)
    signal_layer.append(signal_layer_8)
    signal_layer.append(signal_layer_9)
    signal_layer.append(signal_layer_10)
    signal_layer.append(signal_layer_11)
    signal_layer.append(signal_layer_12)
    signal_layer.append(signal_layer_13)
    signal_layer.append(signal_layer_14)
    signal_layer.append(signal_layer_15)
    signal_layer.append(signal_layer_16)
    signal_layer.append(signal_layer_17)
    signal_layer.append(signal_layer_18)
    signal_layer.append(signal_layer_19)
    signal_layer.append(signal_layer_20)
    signal_layer.append(signal_layer_21)
    signal_layer.append(signal_layer_22)
    signal_layer.append(signal_layer_23)
    signal_layer.append(signal_layer_24)
    signal_layer.append(signal_layer_25)
    signal_layer.append(signal_layer_26)
    signal_layer.append(signal_layer_27)
    signal_layer.append(signal_layer_28)
    signal_layer.append(signal_layer_29)
    signal_layer.append(signal_layer_30)
    signal_layer.append(signal_layer_31)
    signal_layer.append(signal_layer_32)
    signal_layer.append(signal_layer_33)
    signal_layer.append(signal_layer_34)
    signal_layer.append(signal_layer_35)
    signal_layer.append(signal_layer_36)
    signal_layer.append(signal_layer_37)
    signal_layer.append(signal_layer_38)
    signal_layer.append(signal_layer_39)
    signal_layer.append(signal_layer_40)
    signal_layer.append(signal_layer_41)
    signal_layer.append(signal_layer_42)
    signal_layer.append(signal_layer_43)
    signal_layer.append(signal_layer_44)
    signal_layer.append(signal_layer_45)
    signal_layer.append(signal_layer_46)
    signal_layer.append(signal_layer_47)
    signal_layer.append(signal_layer_48)
    signal_layer.append(signal_layer_49)
    signal_layer.append(signal_layer_50)

    return signal_layer




def slice_signal_exp_32(signal):
    signal_layer = []
    signal_layer_1 = signal[44100*0:44100*54]
    signal_layer_2 = signal[int(44100*55):int(44100*109)]
    signal_layer_3 = signal[int(44100*111):int(44100*165)]
    signal_layer_4 = signal[int(44100*165):int(44100*219)]
    signal_layer_5 = signal[int(44100*219):int(44100*273)]
    signal_layer_6 = signal[int(44100*273):int(44100*327)]
    signal_layer_7 = signal[int(44100*330):int(44100*384)]
    signal_layer_8 = signal[int(44100*384):int(44100*438)]
    signal_layer_9 = signal[int(44100*438):int(44100*492)]
    signal_layer_10 = signal[int(44100*492):int(44100*546)]

    signal_layer.append(signal_layer_1)
    signal_layer.append(signal_layer_2)
    signal_layer.append(signal_layer_3)
    signal_layer.append(signal_layer_4)
    signal_layer.append(signal_layer_5)
    signal_layer.append(signal_layer_6)
    signal_layer.append(signal_layer_7)
    signal_layer.append(signal_layer_8)
    signal_layer.append(signal_layer_9)
    signal_layer.append(signal_layer_10)

    return signal_layer


def slice_signal_exp(signal, layer_number, dwell_time):
    signal_layer = []
    layer_time = len(signal)/44100/layer_number # each layer in seconds, e.g., if 60 layers
    interval = layer_time - dwell_time
    
    for i in range(layer_number):
        current_layer = signal[int(44100*layer_time*i):int(44100*(layer_time*i+interval))]
        signal_layer.append(current_layer)
    return signal_layer


def slice_signal_exp_22(signal):
    signal_layer = []
    interval = 3.7
    signal_layer_1 = signal[int(44100*0):int(44100*interval)]
    signal_layer_2 = signal[int(44100*interval):int(44100*interval*2)]
    signal_layer_3 = signal[int(44100*interval*2):int(44100*interval*3)]
    signal_layer_4 = signal[int(44100*interval*3):int(44100*interval*4)]
    signal_layer_5 = signal[int(44100*interval*4):int(44100*interval*5)]
    signal_layer_6 = signal[int(44100*interval*5):int(44100*interval*6)]
    signal_layer_7 = signal[int(44100*interval*6):int(44100*interval*7)]
    signal_layer_8 = signal[int(44100*interval*7):int(44100*interval*8)]
    signal_layer_9 = signal[int(44100*interval*8):int(44100*interval*9)]

    signal_layer_10 = signal[int(44100*interval*9):int(44100*interval*10)]
    signal_layer_11 = signal[int(44100*interval*10):int(44100*interval*11)]
    signal_layer_12 = signal[int(44100*interval*11):int(44100*interval*12)]
    signal_layer_13 = signal[int(44100*interval*12):int(44100*interval*13)]
    signal_layer_14 = signal[int(44100*interval*13):int(44100*interval*14)]
    signal_layer_15 = signal[int(44100*interval*14):int(44100*interval*15)]
    signal_layer_16 = signal[int(44100*interval*15):int(44100*interval*16)]
    signal_layer_17 = signal[int(44100*interval*16):int(44100*interval*17)]
    signal_layer_18 = signal[int(44100*interval*17):int(44100*interval*18)]
    signal_layer_19 = signal[int(44100*interval*18):int(44100*interval*19)]

    signal_layer_20 = signal[int(44100*interval*19):int(44100*interval*20)]
    signal_layer_21 = signal[int(44100*interval*20):int(44100*interval*21)]
    signal_layer_22 = signal[int(44100*interval*21):int(44100*interval*22)]
    signal_layer_23 = signal[int(44100*interval*22):int(44100*interval*23)]
    signal_layer_24 = signal[int(44100*interval*23):int(44100*interval*24)]
    signal_layer_25 = signal[int(44100*interval*24):int(44100*interval*25)]
    signal_layer_26 = signal[int(44100*interval*25):int(44100*interval*26)]
    signal_layer_27 = signal[int(44100*interval*26):int(44100*interval*27)]
    signal_layer_28 = signal[int(44100*interval*27):int(44100*interval*28)]
    signal_layer_29 = signal[int(44100*interval*28):int(44100*interval*29)]

    signal_layer_30 = signal[int(44100*interval*29):int(44100*interval*30)]
    signal_layer_31 = signal[int(44100*interval*30):int(44100*interval*31)]
    signal_layer_32 = signal[int(44100*interval*31):int(44100*interval*32)]
    signal_layer_33 = signal[int(44100*interval*32):int(44100*interval*33)]
    signal_layer_34 = signal[int(44100*interval*33):int(44100*interval*34)]
    signal_layer_35 = signal[int(44100*interval*34):int(44100*interval*35)]
    signal_layer_36 = signal[int(44100*interval*35):int(44100*interval*36)]
    signal_layer_37 = signal[int(44100*interval*36):int(44100*interval*37)]
    signal_layer_38 = signal[int(44100*interval*37):int(44100*interval*38)]
    signal_layer_39 = signal[int(44100*interval*38):int(44100*interval*39)]

    signal_layer_40 = signal[int(44100*interval*39):int(44100*interval*40)]
    signal_layer_41 = signal[int(44100*interval*40):int(44100*interval*41)]
    signal_layer_42 = signal[int(44100*interval*41):int(44100*interval*42)]
    signal_layer_43 = signal[int(44100*interval*42):int(44100*interval*43)]
    signal_layer_44 = signal[int(44100*interval*43):int(44100*interval*44)]
    signal_layer_45 = signal[int(44100*interval*44):int(44100*interval*45)]
    signal_layer_46 = signal[int(44100*interval*45):int(44100*interval*46)]
    signal_layer_47 = signal[int(44100*interval*46):int(44100*interval*47)]
    signal_layer_48 = signal[int(44100*interval*47):int(44100*interval*48)]
    signal_layer_49 = signal[int(44100*interval*48):int(44100*interval*49)]
    signal_layer_50 = signal[int(44100*interval*49):int(44100*interval*50)]

    signal_layer.append(signal_layer_1)
    signal_layer.append(signal_layer_2)
    signal_layer.append(signal_layer_3)
    signal_layer.append(signal_layer_4)
    signal_layer.append(signal_layer_5)
    signal_layer.append(signal_layer_6)
    signal_layer.append(signal_layer_7)
    signal_layer.append(signal_layer_8)
    signal_layer.append(signal_layer_9)
    signal_layer.append(signal_layer_10)
    signal_layer.append(signal_layer_11)
    signal_layer.append(signal_layer_12)
    signal_layer.append(signal_layer_13)
    signal_layer.append(signal_layer_14)
    signal_layer.append(signal_layer_15)
    signal_layer.append(signal_layer_16)
    signal_layer.append(signal_layer_17)
    signal_layer.append(signal_layer_18)
    signal_layer.append(signal_layer_19)
    signal_layer.append(signal_layer_20)
    signal_layer.append(signal_layer_21)
    signal_layer.append(signal_layer_22)
    signal_layer.append(signal_layer_23)
    signal_layer.append(signal_layer_24)
    signal_layer.append(signal_layer_25)
    signal_layer.append(signal_layer_26)
    signal_layer.append(signal_layer_27)
    signal_layer.append(signal_layer_28)
    signal_layer.append(signal_layer_29)
    signal_layer.append(signal_layer_30)
    signal_layer.append(signal_layer_31)
    signal_layer.append(signal_layer_32)
    signal_layer.append(signal_layer_33)
    signal_layer.append(signal_layer_34)
    signal_layer.append(signal_layer_35)
    signal_layer.append(signal_layer_36)
    signal_layer.append(signal_layer_37)
    signal_layer.append(signal_layer_38)
    signal_layer.append(signal_layer_39)
    signal_layer.append(signal_layer_40)
    signal_layer.append(signal_layer_41)
    signal_layer.append(signal_layer_42)
    signal_layer.append(signal_layer_43)
    signal_layer.append(signal_layer_44)
    signal_layer.append(signal_layer_45)
    signal_layer.append(signal_layer_46)
    signal_layer.append(signal_layer_47)
    signal_layer.append(signal_layer_48)
    signal_layer.append(signal_layer_49)
    signal_layer.append(signal_layer_50)

    return signal_layer






