import pandas as pd
import matplotlib.pyplot as plt

# 讀取資料
file_path = "./dataset/39_Training_Dataset/train_data/40.txt"
df = pd.read_csv(file_path, sep=" ", header=None)
df.columns = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

# 建立時間軸（每秒85筆）
df["time"] = df.index / 85

colors = {
    "Ax": "red",
    "Ay": "green",
    "Az": "blue",
    "Gx": "orange",
    "Gy": "purple",
    "Gz": "cyan"
}

# 繪圖
plt.figure(figsize=(16, 8))
for col in ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]:
    plt.plot(df["time"], df[col], label=col, color=colors[col])

plt.xlabel("Time (seconds)")
plt.ylabel("Value")
plt.title("Player Swing Sensor Data of Waveform")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("sensor_data_plot.png")
