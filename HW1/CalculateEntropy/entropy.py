import numpy as np
import cv2
from math import log2
from collections import defaultdict

# ---------------------------
# 1. 讀取 raw 檔案並轉成彩色影像
# ---------------------------
def load_raw_image(filename, width=512, height=512, mode='interleaved'):
    """
    讀取 raw 圖像檔案並轉換成 numpy 陣列 (height x width x 3)
    :param filename: raw 檔案名稱
    :param width: 圖像寬度 (預設 512)
    :param height: 圖像高度 (預設 512)
    :param mode: 'interleaved' 或 'plane'
    :return: numpy 陣列 (height, width, 3)，RGB 格式
    """
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    
    if mode == 'interleaved':
        # 資料依序為 RGBRGB...
        img = data.reshape((height, width, 3))
    elif mode == 'plane':
        # 資料依序為 RRR... GGG... BBB...
        img = data.reshape((3, height, width))
        img = np.transpose(img, (1, 2, 0))
    else:
        raise ValueError("mode 必須是 'interleaved' 或 'plane'")
    return img

# ---------------------------
# 2. 轉換 RGB 至 YUV (依題目提供的公式)
# ---------------------------
def rgb_to_yuv(img):
    """
    依據公式將 RGB 影像轉換為 YUV 影像
    Y = R * 0.299000 + G * 0.587000 + B * 0.114000
    U = R * -0.168736 + G * -0.331264 + B * 0.500000 + 128
    V = R * 0.500000 + G * -0.418688 + B * -0.081312 + 128
    :param img: numpy 陣列 (height, width, 3) ，假設順序為 RGB
    :return: yuv_img: numpy 陣列 (height, width, 3) ，分別為 Y, U, V
    """
    # 將影像轉為 float 進行計算
    img = img.astype(np.float32)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    Y = R * 0.299000 + G * 0.587000 + B * 0.114000 
    U = R * -0.168736 + G * -0.331264 + B * 0.500000 + 128
    V = R * 0.500000 + G * -0.418688 + B * -0.081312 + 128

    # 轉回 uint8 並堆疊
    YUV = np.stack((Y, U, V), axis=-1).clip(0, 255).astype(np.uint8)
    return YUV

# ---------------------------
# 3. 一階熵計算
# ---------------------------
def compute_entropy(channel):
    """
    計算單一通道的 1st-order entropy
    熵公式： H = -sum(P(i)*log2(P(i)))，其中 i 為像素值 0~255
    :param channel: numpy 陣列 (height, width) 的單通道影像 (uint8)
    :return: entropy (bits per pixel)
    """
    hist, _ = np.histogram(channel, bins=256, range=(0, 256))
    total = channel.size
    probs = hist / total
    # 過濾掉機率為 0 的部分，避免 log(0)
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    return entropy

# ---------------------------
# 4. 聯合熵計算 (Joint Entropy)
# ---------------------------
def compute_joint_entropy(channels):
    """
    計算多通道的聯合熵，將每個像素的多個值視為一個 symbol
    :param channels: list of numpy arrays，每個 shape 為 (height, width)
    :return: joint entropy (bits per symbol)
    """
    # 將各通道合併成一個 symbol tuple (例如 (R,G,B))
    # 將每個像素轉成一維向量，再轉換成 tuple，然後統計出現次數
    height, width = channels[0].shape
    symbols = np.stack(channels, axis=-1).reshape(-1, len(channels))
    # 將每一列轉成 tuple 
    unique_symbols, counts = np.unique(symbols, axis=0, return_counts=True)
    probs = counts / symbols.shape[0]
    joint_entropy = -np.sum(probs * np.log2(probs))
    return joint_entropy

# ---------------------------
# 5. 條件熵計算 (Conditional Entropy)
# ---------------------------
def compute_conditional_entropy(channel, neighbor='left'):
    """
    計算單一通道的條件熵 H(S0|S_neighbor)
    - 如果 neighbor='left'，則每個像素的已知值為左邊像素，對第一欄使用 128 作為預設值
    - 如果 neighbor='upper'，則每個像素的已知值為上方像素，對第一行使用 128 作為預設值
    條件熵公式： H(X|Y) = -sum_{y} P(y) * sum_{x} P(x|y) log2(P(x|y))
    這裡我們直接用聯合統計來計算：
        H = - sum_{(y,x)} P(y,x) log2(P(x|y))
    :param channel: numpy 陣列 (height, width) 的單通道影像 (uint8)
    :param neighbor: 'left' 或 'upper'
    :return: conditional entropy (bits per pixel)
    """
    height, width = channel.shape
    # 使用 defaultdict 記錄 joint 次數： (neighbor_value, current_value)
    joint_counts = defaultdict(int)
    neighbor_counts = defaultdict(int)
    
    for i in range(height):
        for j in range(width):
            # 取得當前像素值
            current = int(channel[i, j])
            # 根據 neighbor 選擇相鄰像素值，邊界使用 128
            if neighbor == 'left':
                if j == 0:
                    nb = 128
                else:
                    nb = int(channel[i, j-1])
            elif neighbor == 'upper':
                if i == 0:
                    nb = 128
                else:
                    nb = int(channel[i-1, j])
            else:
                raise ValueError("neighbor 參數必須是 'left' 或 'upper'")
            
            joint_counts[(nb, current)] += 1
            neighbor_counts[nb] += 1

    total = height * width
    cond_entropy = 0.0
    # 計算聯合機率和條件機率
    for (nb, current), count in joint_counts.items():
        p_joint = count / total
        p_nb = neighbor_counts[nb] / total
        p_current_given_nb = p_joint / p_nb
        cond_entropy += p_joint * log2(1 / p_current_given_nb)
    
    return cond_entropy

# ---------------------------
# 6. 主程式
# ---------------------------
def main():
    # 設定圖像檔案名稱(測試.raw)
    filename = './Plane/LenaRGB.raw'
    
    # 載入 raw 影像（根據檔案格式選擇 'interleaved' 或 'plane'）
    #img = load_raw_image(filename, width=512, height=512, mode='interleaved')
    img = load_raw_image(filename, width=512, height=512, mode='plane')
    
    # 分離 RGB 通道
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    # 將 RGB 轉換為 YUV 
    yuv_img = rgb_to_yuv(img)
    Y = yuv_img[:, :, 0]
    U = yuv_img[:, :, 1]
    V = yuv_img[:, :, 2]
    
    # ---- 計算 1st order entropy ----
    print("==== 1st Order Entropy ====")
    print("RGB channels:")
    print(f"H(R) = {compute_entropy(R):.2f} bits/pixel")
    print(f"H(G) = {compute_entropy(G):.2f} bits/pixel")
    print(f"H(B) = {compute_entropy(B):.2f} bits/pixel")
    
    print("YUV channels:")
    print(f"H(Y) = {compute_entropy(Y):.2f} bits/pixel")
    print(f"H(U) = {compute_entropy(U):.2f} bits/pixel")
    print(f"H(V) = {compute_entropy(V):.2f} bits/pixel")
    
    # ---- 計算 Joint Entropy ----
    print("\n==== Joint Entropy ====")
    # Joint entropy of {R,G,B}
    joint_rgb = compute_joint_entropy([R, G, B])
    print(f"Joint Entropy of (R,G,B) = {joint_rgb:.2f} bits/pixel")
    # Joint entropy of {Y,U,V}
    joint_yuv = compute_joint_entropy([Y, U, V])
    print(f"Joint Entropy of (Y,U,V) = {joint_yuv:.2f} bits/pixel")
    
    # ---- 計算 Conditional Entropy ----
    print("\n==== Conditional Entropy ====")
    # 分別對 RGB 或 YUV 各通道計算Conditional Entropy
    cond_left_R = compute_conditional_entropy(R, neighbor='left')
    cond_upper_R = compute_conditional_entropy(R, neighbor='upper')
    cond_left_G = compute_conditional_entropy(G, neighbor='left')
    cond_upper_G = compute_conditional_entropy(G, neighbor='upper')
    cond_left_B = compute_conditional_entropy(B, neighbor='left')
    cond_upper_B = compute_conditional_entropy(B, neighbor='upper')
    
    cond_left_Y = compute_conditional_entropy(Y, neighbor='left')
    cond_upper_Y = compute_conditional_entropy(Y, neighbor='upper')
    cond_left_U = compute_conditional_entropy(U, neighbor='left')
    cond_upper_U = compute_conditional_entropy(U, neighbor='upper')
    cond_left_V = compute_conditional_entropy(V, neighbor='left')
    cond_upper_V = compute_conditional_entropy(V, neighbor='upper')

    print(f"Conditional Entropy of R given left pixel = {cond_left_R:.2f} bits/pixel")
    print(f"Conditional Entropy of R given upper pixel = {cond_upper_R:.2f} bits/pixel")
    print(f"Conditional Entropy of G given left pixel = {cond_left_G:.2f} bits/pixel")
    print(f"Conditional Entropy of G given upper pixel = {cond_upper_G:.2f} bits/pixel")
    print(f"Conditional Entropy of B given left pixel = {cond_left_B:.2f} bits/pixel")
    print(f"Conditional Entropy of B given upper pixel = {cond_upper_B:.2f} bits/pixel")
    print(f"Conditional Entropy of Y given left pixel = {cond_left_Y:.2f} bits/pixel")
    print(f"Conditional Entropy of Y given upper pixel = {cond_upper_Y:.2f} bits/pixel")
    print(f"Conditional Entropy of U given left pixel = {cond_left_U:.2f} bits/pixel")
    print(f"Conditional Entropy of U given upper pixel = {cond_upper_U:.2f} bits/pixel")
    print(f"Conditional Entropy of V given left pixel = {cond_left_V:.2f} bits/pixel")
    print(f"Conditional Entropy of V given upper pixel = {cond_upper_V:.2f} bits/pixel")
    
    output_lines = []
    output_lines.append("==== 1st Order Entropy ====")
    output_lines.append("RGB channels:")
    output_lines.append(f"H(R) = {compute_entropy(R):.2f} bits/pixel")
    output_lines.append(f"H(G) = {compute_entropy(G):.2f} bits/pixel")
    output_lines.append(f"H(B) = {compute_entropy(B):.2f} bits/pixel")

    output_lines.append("YUV channels:")
    output_lines.append(f"H(Y) = {compute_entropy(Y):.2f} bits/pixel")
    output_lines.append(f"H(U) = {compute_entropy(U):.2f} bits/pixel")
    output_lines.append(f"H(V) = {compute_entropy(V):.2f} bits/pixel")

    output_lines.append("\n==== Joint Entropy ====")
    output_lines.append(f"Joint Entropy of (R,G,B) = {joint_rgb:.2f} bits/pixel")
    output_lines.append(f"Joint Entropy of (Y,U,V) = {joint_yuv:.2f} bits/pixel")

    output_lines.append("\n==== Conditional Entropy ====")
    output_lines.append(f"Conditional Entropy of R given left pixel = {cond_left_R:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of R given upper pixel = {cond_upper_R:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of G given left pixel = {cond_left_G:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of G given upper pixel = {cond_upper_G:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of B given left pixel = {cond_left_B:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of B given upper pixel = {cond_upper_B:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of Y given left pixel = {cond_left_Y:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of Y given upper pixel = {cond_upper_Y:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of U given left pixel = {cond_left_U:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of U given upper pixel = {cond_upper_U:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of V given left pixel = {cond_left_V:.2f} bits/pixel")
    output_lines.append(f"Conditional Entropy of V given upper pixel = {cond_upper_V:.2f} bits/pixel")

    # 寫入文字檔
    with open("Lena_output.txt", "w") as f:
        for line in output_lines:
            print(line)      # 同時仍顯示在 console
            f.write(line + "\n")

if __name__ == '__main__':
    main()
