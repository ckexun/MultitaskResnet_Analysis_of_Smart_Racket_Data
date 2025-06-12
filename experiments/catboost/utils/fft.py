from pathlib import Path  # 用於處理檔案和目錄路徑
import numpy as np        # 匯入 numpy 進行數值運算
import math               # 匯入 math 模組進行數學相關操作
import csv                # 用於讀寫 CSV 檔案
import os                 # 提供檔案和目錄操作功能


def FFT(xreal, ximag):
    """
    對輸入之實部 (xreal) 與虛部 (ximag) 進行快速傅立葉轉換 (FFT)。

    參數:
        xreal (list[float]): 時域信號之實部
        ximag (list[float]): 時域信號之虛部，若無則可傳入全 0 列表

    回傳:
        n (int): FFT 點數
        xreal (list[float]): 頻域轉換後之實部
        ximag (list[float]): 頻域轉換後之虛部
    """
    # 找出最大可用的 2 的冪次 n，且 n 不超過輸入長度
    n = 2
    while (n * 2 <= len(xreal)):
        n *= 2

    # p 為 FFT 階層 (log2(n))
    p = int(math.log(n, 2))

    # 位址反轉 (bit-reversal) 置換
    for i in range(0, n):
        a, b = i, 0
        for j in range(0, p):
            b = int(b * 2 + a % 2)
            a = a / 2
        if b > i:
            xreal[i], xreal[b] = xreal[b], xreal[i]
            ximag[i], ximag[b] = ximag[b], ximag[i]

    # 初始化旋轉因子 (twiddle factors)
    wreal, wimag = [1.0], [0.0]
    arg = -2 * math.pi / n
    treal, timag = math.cos(arg), math.sin(arg)

    # 計算所有所需的旋轉因子
    for j in range(1, n // 2):
        wreal.append(wreal[-1] * treal - wimag[-1] * timag)
        wimag.append(wreal[-1] * timag + wimag[-1] * treal)

    # FFT 主迴圈：分段蝶形運算
    m = 2
    while m <= n:
        for k in range(0, n, m):
            for j in range(m // 2):
                idx1, idx2 = k + j, k + j + m // 2
                t = int(n * j / m)
                tre = wreal[t] * xreal[idx2] - wimag[t] * ximag[idx2]
                tim = wreal[t] * ximag[idx2] + wimag[t] * xreal[idx2]
                ure, uim = xreal[idx1], ximag[idx1]
                xreal[idx1], ximag[idx1] = ure + tre, uim + tim
                xreal[idx2], ximag[idx2] = ure - tre, uim - tim
        m *= 2

    return n, xreal, ximag


def FFT_data(input_data, swinging_times):
    """
    計算輸入感測資料每個擺動階段之加速度與角速度平均值

    參數:
        input_data (list[list[int]]): 讀入的感測資料，每筆包含六軸數值
        swinging_times (list[int]): 擺動階段起始索引列表

    回傳:
        a_mean (list[float]): 每階段之加速度均值
        g_mean (list[float]): 每階段之角速度均值
    """
    txtlength = swinging_times[-1] - swinging_times[0]
    a_mean = [0] * txtlength
    g_mean = [0] * txtlength

    for seg in range(len(swinging_times) - 1):
        a_vals, g_vals = [], []
        for i in range(swinging_times[seg], swinging_times[seg + 1]):
            # 三軸向量大小
            a_vals.append(math.sqrt(sum(input_data[i][0:3]) ** 2))
            g_vals.append(math.sqrt(sum(input_data[i][3:6]) ** 2))
        a_mean[seg] = sum(a_vals) / len(a_vals)
        g_mean[seg] = sum(g_vals) / len(g_vals)

    return a_mean, g_mean


def feature(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer):
    """
    計算單一擺動段之時域與頻域特徵，並將結果寫入 CSV

    參數:
        input_data (list[list[int]]): 該階段感測資料
        swinging_now (int): 當前階段索引 (從 0 開始)
        swinging_times (int): 總階段數
        n_fft (int): FFT 點數
        a_fft, g_fft (list[float]): 頻域實部
        a_fft_imag, g_fft_imag (list[float]): 頻域虛部
        writer (csv.writer): CSV 寫入器
    """
    # 時域統計特徵初始化
    allsum = [0] * len(input_data[0])
    a_vals, g_vals = [], []

    # 計算平均、變異數、RMS
    for row in input_data:
        allsum = [allsum[i] + row[i] for i in range(len(row))]
        a_vals.append(math.sqrt(sum(row[0:3]) ** 2))
        g_vals.append(math.sqrt(sum(row[3:6]) ** 2))
    mean = [s / len(input_data) for s in allsum]
    var = [0] * len(mean)
    rms = [0] * len(mean)
    for row in input_data:
        for i, v in enumerate(row):
            var[i] += (v - mean[i]) ** 2
            rms[i] += v ** 2
    var = [math.sqrt(v / len(input_data)) for v in var]
    rms = [math.sqrt(r / len(input_data)) for r in rms]

    # 加速度與角速度統計
    stats = lambda vals: [max(vals), sum(vals) / len(vals), min(vals)]
    a_max, a_mean, a_min = stats(a_vals)
    g_max, g_mean, g_min = stats(g_vals)

    # 峰度與偏度
    def moments(vals, m): return sum((v - sum(vals)/len(vals))**m for v in vals) / len(vals)
    a_kurt = moments(a_vals, 4) / (moments(a_vals, 2)**2)
    g_kurt = moments(g_vals, 4) / (moments(g_vals, 2)**2)
    a_skew = moments(a_vals, 3) / (moments(a_vals, 2)**1.5)
    g_skew = moments(g_vals, 3) / (moments(g_vals, 2)**1.5)

    # 頻域 PSD 與熵
    cut = n_fft // swinging_times
    a_psd = [a_fft[i]**2 + a_fft_imag[i]**2 for i in range(cut*swinging_now, cut*(swinging_now+1))]
    g_psd = [g_fft[i]**2 + g_fft_imag[i]**2 for i in range(cut*swinging_now, cut*(swinging_now+1))]
    a_fft_mean = sum(a_fft[cut*swinging_now:cut*(swinging_now+1)]) / cut
    g_fft_mean = sum(g_fft[cut*swinging_now:cut*(swinging_now+1)]) / cut
    a_psd_mean, g_psd_mean = sum(a_psd)/len(a_psd), sum(g_psd)/len(g_psd)
    e2 = sum(math.sqrt(p) for p in a_psd)
    e4 = sum(math.sqrt(p) for p in g_psd)
    entropy_a = sum((math.sqrt(a_psd[i])/e2)*math.log(math.sqrt(a_psd[i])/e2) for i in range(cut)) / cut
    entropy_g = sum((math.sqrt(g_psd[i])/e4)*math.log(math.sqrt(g_psd[i])/e4) for i in range(cut)) / cut

    # 合併特徵並寫入 CSV
    features = mean + var + rms + [a_max, a_mean, a_min] + [g_max, g_mean, g_min] + \
               [a_fft_mean, g_fft_mean, a_psd_mean, g_psd_mean, a_kurt, g_kurt, a_skew, g_skew, entropy_a, entropy_g]
    writer.writerow(features)


def data_generate(datapath: str, tar_dir: str):
    """
    批次讀取目錄下所有 .txt 感測資料檔，計算切段特徵並輸出對應 CSV。

    參數:
        datapath (str): 原始 .txt 檔案目錄
        tar_dir (str): 輸出 CSV 檔案的目錄
    """
    pathlist = Path(datapath).glob('**/*.txt')
    os.makedirs(tar_dir, exist_ok=True)

    for file in pathlist:
        with open(file) as f:
            lines = [l for l in f.readlines() if l.strip()]
        All_data = [[int(x) for x in line.split()[:6]] for line in lines[1:]]

        swing_idx = np.linspace(0, len(All_data), 28, dtype=int)
        output = Path(tar_dir) / f"{file.stem}.csv"
        with open(output, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [
                'ax_mean','ay_mean','az_mean','gx_mean','gy_mean','gz_mean',
                'ax_var','ay_var','az_var','gx_var','gy_var','gz_var',
                'ax_rms','ay_rms','az_rms','gx_rms','gy_rms','gz_rms',
                'a_max','a_mean','a_min','g_max','g_mean','g_min',
                'a_fft','g_fft','a_psd','g_psd','a_kurt','g_kurt','a_skew','g_skew','a_entropy','g_entropy'
            ]
            writer.writerow(header)
            try:
                a_fft, g_fft = FFT_data(All_data, swing_idx)
                a_imag = [0]*len(a_fft)
                g_imag = [0]*len(g_fft)
                n_fft, a_fft, a_imag = FFT(a_fft, a_imag)
                _,    g_fft, g_imag = FFT(g_fft, g_imag)
                for i in range(1, len(swing_idx)):
                    feature(All_data[swing_idx[i-1]:swing_idx[i]], i-1, len(swing_idx)-1, n_fft, a_fft, g_fft, a_imag, g_imag, writer)
            except Exception as e:
                print(f"Error in file: {file.stem} -> {e}")
                continue
