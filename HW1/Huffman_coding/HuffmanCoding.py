from PIL import Image
import os
import heapq

# ---------------------------------------------------------------------------- #
#                           Huffman 編碼相關函式                                #
# ---------------------------------------------------------------------------- #


class Node:
    """Huffman 樹的節點"""

    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol      # 符號（像素值或差值）
        self.freq = freq          # 出現頻率
        self.left = left          # 左子節點
        self.right = right        # 右子節點

    def __lt__(self, other):
        # 用於priority queue，以頻率排序
        return self.freq < other.freq


def compute_frequency(data):
    """
    計算資料中每個符號的頻率
    :param data: list of int, 原始資料或差分資料
    :return: dict, {symbol: frequency}
    """
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    print(
        f"Debug: 總計 {len(freq)} 個符號，頻率範例 (前 10 項): {list(freq.items())[:10]}")
    return freq


def build_huffman_tree(freq):
    """
    建立 Huffman 樹
    :param freq: dict, {symbol: frequency}
    :return: Node, Huffman 樹根節點
    """
    heap = [Node(symbol, f) for symbol, f in freq.items()]
    heapq.heapify(heap)
    print("Heap 最小值 (symbol, freq):", (heap[0].symbol, heap[0].freq))

    # 合併最小的兩個節點，直到只剩一個樹
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq, n1, n2)
        heapq.heappush(heap, merged)

    print("Debug: Huffman 樹構建完成")
    return heap[0]


def generate_codes(node, prefix="", code_map=None):
    """
    從 Huffman 樹生成初步的二進位碼
    :param node: Node, 樹節點
    :param prefix: str, 當前前綴碼
    :param code_map: dict, 儲存 {symbol: code}
    :return: dict, {symbol: code}
    """
    if code_map is None:
        code_map = {}

    # 如果是葉節點，將符號對應到前綴碼
    if node.symbol is not None:
        code_map[node.symbol] = prefix

    # 非葉節點，左右遞迴
    else:
        generate_codes(node.left, prefix + "0", code_map)
        generate_codes(node.right, prefix + "1", code_map)

    return code_map


def generate_canonical_codes(code_map):
    """
    將 Huffman 二進位碼轉為 Canonical Huffman 碼，並回傳每個符號的碼長
    :param code_map: dict, {symbol: binary_code}
    :return: (dict, dict), (canonical_code_map, lengths_map)
    """
    lengths = {s: len(c) for s, c in code_map.items()}
    sorted_syms = sorted(lengths.items(), key=lambda x: (x[1], x[0]))

    canonical_codes = {}
    code = 0
    prev_len = 0
    for sym, length in sorted_syms:
        code <<= (length - prev_len)
        canonical_codes[sym] = format(code, 'b').zfill(length)
        code += 1
        prev_len = length

    print(f"Debug: 初步碼長範例 (前 10 項): {sorted_syms[:10]}")
    print(
        f"Debug: Canonical 碼表示例 (前 10 項): {list(canonical_codes.items())[:10]}")
    return canonical_codes, lengths


# ---------------------------------------------------------------------------- #
#                        圖片讀取與 DPCM 前處理                                #
# ---------------------------------------------------------------------------- #

def load_luminance(path):
    """
    讀灰階影像即為 Y (luminance) 通道
    :param path: str, 影像檔路徑
    :return: list of int, 影像 Y 通道像素值平坦化列表
    """
    img = Image.open(path)
    y = list(img.getdata())
    print(f"Debug: 已載入 '{path}'，解析度 {img.size}，共 {len(y)} 個像素")
    return y


def dpcm_encode(data):
    """
    對資料做差分脈動編碼 (DPCM)，採用前一個樣本差分
    :param data: list of int, 原始像素值
    :return: list of int, 差分後資料 (範圍 0~510)
    """
    dpcm = []
    prev = 0
    for val in data:
        diff = val - prev
        dpcm.append(diff)
        prev = val
    print("Debug: DPCM 編碼完成，範圍:", (min(dpcm), max(dpcm)))
    return dpcm


# ---------------------------------------------------------------------------- #
#                        壓縮並寫入檔案                                      #
# ---------------------------------------------------------------------------- #

def compress_and_write(data, symbol_range, output_path):
    """
    壓縮資料並將結果寫入檔案，header 為每個符號的 Canonical Huffman 碼長
    :param data: list of int, 要壓縮的資料
    :param symbol_range: iterable, 所有可能符號範圍
    :param output_path: str, 輸出檔案路徑
    """
    freq = compute_frequency(data)
    root = build_huffman_tree(freq)
    init_codes = generate_codes(root)
    print(f"Debug: 初步 Huffman 碼表 (前 10 項): {list(init_codes.items())[:10]}")
    canon_codes, lengths = generate_canonical_codes(init_codes)

    symbol_list = list(symbol_range)
    length_table = [lengths.get(sym, 0) for sym in symbol_list]
    header_bytes = bytes(length_table)
    print(f"Debug: Header bytes 長度 = {len(header_bytes)}")

    bit_str = "".join(canon_codes[sym] for sym in data)
    padding = (8 - len(bit_str) % 8) % 8
    print(f"Debug: Bit 字串長度 = {len(bit_str)}，Padding = {padding}")
    bit_str += "0" * padding

    data_bytes = int(bit_str, 2).to_bytes(len(bit_str) // 8, byteorder='big')
    print(f"Debug: 寫入壓縮資料 bytes = {len(data_bytes)}")

    with open(output_path, "wb") as f:
        f.write(header_bytes)
        f.write(padding.to_bytes(1, 'big'))
        f.write(data_bytes)
    print(
        f"Debug: 檔案已寫入 '{output_path}'，大小 = {os.path.getsize(output_path)} bytes\n")


def test_compression(image_path, use_dpcm=False):
    """
    測試對單張影像的壓縮，並回傳檔案大小
    :param image_path: str, 影像檔路徑
    :param use_dpcm: bool, 是否使用 DPCM 前處理
    :return: int, 壓縮後檔案大小 (bytes)
    """
    y_data = load_luminance(image_path)
    if use_dpcm:
        data = [val + 255 for val in dpcm_encode(y_data)]
        symbol_range = range(511)
        out_name = f"{os.path.splitext(image_path)[0]}_dpcm_{os.path.splitext(image_path)[1][1:]}.bin"
    else:
        data = y_data
        symbol_range = range(256)
        out_name = f"{os.path.splitext(image_path)[0]}_orig_{os.path.splitext(image_path)[1][1:]}.bin"

    compress_and_write(data, symbol_range, out_name)
    return os.path.getsize(out_name)


# ---------------------------------------------------------------------------- #
#                        主測試流程與結果驗證                                  #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # 測試 Lena 和 Baboon
    lena_gray_size = os.path.getsize("Lena.png")
    baboon_gray_size = os.path.getsize("Baboon.png")
    lena_orig_size = test_compression("Lena.png", use_dpcm=False)
    lena_dpcm_size = test_compression("Lena.png", use_dpcm=True)
    baboon_orig_size = test_compression("Baboon.png", use_dpcm=False)
    baboon_dpcm_size = test_compression("Baboon.png", use_dpcm=True)

    print("最終壓縮結果 for png (bytes):")
    print(f"  Lena Gray: {lena_gray_size}")
    print(f"  Lena 原始: {lena_orig_size}")
    print(f"  Lena DPCM: {lena_dpcm_size}")
    print(f"  Baboon Gray: {baboon_gray_size}")
    print(f"  Baboon 原始: {baboon_orig_size}")
    print(f"  Baboon DPCM: {baboon_dpcm_size}")

    lena_gray_size = os.path.getsize("Lena.bmp")
    baboon_gray_size = os.path.getsize("Baboon.bmp")
    lena_orig_size = test_compression("Lena.bmp", use_dpcm=False)
    lena_dpcm_size = test_compression("Lena.bmp", use_dpcm=True)
    baboon_orig_size = test_compression("Baboon.bmp", use_dpcm=False)
    baboon_dpcm_size = test_compression("Baboon.bmp", use_dpcm=True)

    print("最終壓縮結果 for bmp (bytes):")
    print(f"  Lena Gray: {lena_gray_size}")
    print(f"  Lena 原始: {lena_orig_size}")
    print(f"  Lena DPCM: {lena_dpcm_size}")
    print(f"  Baboon Gray: {baboon_gray_size}")
    print(f"  Baboon 原始: {baboon_orig_size}")
    print(f"  Baboon DPCM: {baboon_dpcm_size}")
