#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.fftpack import dct, idct
import pickle
import math

# ------------------------------------------------------------
# JPEG 標準亮度量化表（8×8）
# ------------------------------------------------------------
JPEG_LUMA_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
])

def scale_quant_table(qt, Q):
    """
    根據 Quality Factor Q (1–100) 調整量化表
    參數:
      qt: 原始 8×8 量化表 (numpy array)
      Q: 1~100 的品質因子
    回傳:
      qt2: 調整後的整數量化表 (8×8 numpy array)
    演算法:
      Q<50 → scale=50/Q；Q>=50 → scale=2−(2Q/100)
      四捨五入後 clip 到 [1,255]
    """
    Q = max(1, min(Q, 100))
    scale = 50.0/Q if Q < 50 else 2 - (Q * 2 / 100)
    qt2 = np.floor(qt * scale + 0.5).astype(int)
    qt2 = np.clip(qt2, 1, 255)
    return qt2

def blockify(img, bs=8):
    """
    將影像切成 8×8 的 block
    參數:
      img: 2D 或 3D numpy array，shape=(H,W) 或 (H,W,C)
      bs: block size，預設 8
    回傳:
      blocks: shape=(nh, nw, bs, bs[, C]) 的區塊陣列
      H, W: 原始圖高度與寬度
    流程:
      1. pad 到可被 bs 整除
      2. reshape + transpose 拆成小區塊
    """
    H, W = img.shape[:2]
    pad_h = (-H) % bs
    pad_w = (-W) % bs
    if pad_h or pad_w:
        img = np.pad(img,
                     [(0, pad_h), (0, pad_w)] + ([(0,0)] if img.ndim==3 else []),
                     mode='constant')
    nh, nw = img.shape[0]//bs, img.shape[1]//bs
    if img.ndim == 2:
        blocks = img.reshape(nh, bs, nw, bs).transpose(0,2,1,3)
    else:
        C = img.shape[2]
        blocks = img.reshape(nh, bs, nw, bs, C).transpose(0,2,1,3,4)
    return blocks, H, W

def unblockify(blocks, H, W):
    """
    將 blocks 還原成圖像並移除 pad
    參數:
      blocks: shape=(nh,nw,bs,bs[,C]) 的區塊陣列
      H, W: 還原後要裁剪成的原始尺寸
    回傳:
      img: shape=(H,W) or (H,W,C) 的重建影像
    """
    nh, nw = blocks.shape[:2]
    bs = blocks.shape[2]
    if blocks.ndim == 4:
        img_p = blocks.transpose(0,2,1,3).reshape(nh*bs, nw*bs)
    else:
        img_p = blocks.transpose(0,2,1,3,4).reshape(nh*bs, nw*bs, blocks.shape[4])
    return img_p[:H, :W]

def make_zigzag_idx(n=8):
    """
    生成 n×n 的 Zig-Zag 掃描索引
    回傳:
      一維長度 n*n 的索引陣列，可用於重排 flat block
    """
    idx = np.arange(n*n).reshape(n,n)
    order = []
    for s in range(2*n-1):
        pts = [(i, s-i) for i in range(n) if 0 <= s-i < n]
        if s % 2 == 0:
            pts.reverse()
        order += pts
    return np.array([idx[i,j] for (i,j) in order], dtype=int)

ZZ_IDX = make_zigzag_idx(8)

def rle_encode_ac(row63):
    """
    對一個長度 63 的 AC vector 做 RLE 編碼
    參數:
      row63: 一維 numpy array，長度 63 (除去 DC)
    回傳:
      out: list of (zero_run, value) tuples，最後一筆 EOB=(0,0)
    流程:
      - 連續零計數，遇到非零則輸出 (zero_count, value)
      - 超過 16 個 0 時插入 ZRL=(15,0)
      - 最後補 EOB
    """
    out = []
    zero = 0
    for v in row63:
        if v == 0:
            zero += 1
            if zero == 16:
                out.append((15, 0))  # ZRL
                zero = 0
        else:
            out.append((zero, int(v)))
            zero = 0
    # 最後如果還有零，或尚無輸出，補 EOB
    if zero > 0 or not out or out[-1] != (0, 0):
        out.append((0, 0))
    # 確保全部都是 tuple
    return [tuple(x) for x in out]

def rle_decode_ac(ac_list):
    """
    將 RLE list 還原成長度 63 的 vector
    參數:
      ac_list: list of (zero_run, value)
    回傳:
      out: 一維 list，長度 63
    流程:
      - 遇 EOB=(0,0) 停止並補零
      - 依 zero_run 插入對應數量的 0
    """
    out = []
    for zr, val in ac_list:
        if (zr, val) == (0, 0):
            out += [0] * (63 - len(out))
            break
        out += [0] * zr + [val]
    out += [0] * (63 - len(out))
    return out

class HuffNode:
    """
    Huffman 樹節點
    sym: 符號 (tuple 或 int)
    freq: 該符號頻率
    left/right: 子節點
    """
    def __init__(self, sym=None, f=0):
        self.sym, self.freq = sym, f
        self.left = self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huff_tree(freq_map):
    """
    根據頻率表建 Huffman 樹
    參數:
      freq_map: dict，key=symbol, value=frequency
    回傳:
      HuffNode: 樹根
    """
    import heapq
    heap = [HuffNode(s, f) for s, f in freq_map.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        p = HuffNode(None, a.freq + b.freq)
        p.left, p.right = a, b
        heapq.heappush(heap, p)
    return heap[0]

def make_huff_codes(node, prefix='', cmap=None):
    """
    遞迴生成 Huffman code table
    參數:
      node: HuffNode
      prefix: 上層路徑的 0/1 字串
      cmap: dict，sym->code
    回傳:
      cmap: 填好所有符號對應的二進位字串
    """
    if cmap is None:
        cmap = {}
    if node.sym is not None:
        cmap[node.sym] = prefix
    else:
        make_huff_codes(node.left, prefix + '0', cmap)
        make_huff_codes(node.right, prefix + '1', cmap)
    return cmap

def huffman_encode(symbols):
    """
    給定 symbol list 做 Huffman 編碼
    參數:
      symbols: list of hashable (tuple 或 int)
    回傳:
      bitstr: 連接所有 code 的 bitstring
      cmap: symbol->code dict
    """
    # 1. 建頻率表
    freq = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    # 2. 建樹 & code table
    tree = build_huff_tree(freq)
    cmap = make_huff_codes(tree)
    # 3. 編碼
    bitstr = ''.join(cmap[s] for s in symbols)
    return bitstr, cmap

def huffman_decode(bitstr, cmap):
    """
    由 bitstring 與 code table 還原 symbols list
    參數:
      bitstr: 編碼後的 0/1 字串
      cmap: symbol->code dict
    回傳:
      out: 解碼後的 symbol list
    """
    inv = {v: k for k, v in cmap.items()}
    out = []
    buf = ''
    for b in bitstr:
        buf += b
        if buf in inv:
            out.append(inv[buf])
            buf = ''
    return out

def encode_channel(img_ch, qtable):
    """
    對單一 channel 做完整壓縮：
    1. blockify → 2D DCT → 量化
    2. Zig-Zag 扁平化 → 分離 DC/AC
    3. AC 做 RLE → 回傳 DC list, AC list
    """
    blocks, H, W = blockify(img_ch, 8)
    # DCT 需先減去 128 (將值域移到 [-128,127])
    data = blocks.astype(int) - 128
    # 2D DCT：沿 axis=2,3 做
    d1 = dct(dct(data, axis=2, norm='ortho'), axis=3, norm='ortho')
    # 量化
    q = np.round(d1 / qtable).astype(int)
    # flatten 每個 8×8 block → 長度 64
    flat = q.reshape(-1, 64)
    # Zig-Zag reorder
    zz = flat[:, ZZ_IDX]
    # DC = 第一個係數，AC = 後面 63
    dc = zz[:, 0].tolist()
    ac = [rle_encode_ac(zz[i,1:]) for i in range(zz.shape[0])]
    return dc, ac, H, W

def decode_channel(dc_list, ac_list, qtable, H, W):
    """
    將 DC list 與 AC list 還原成影像 channel：
    1. RLE decode → rebuild zigzag vector
    2. inverse zigzag → reshape 回 8×8 block
    3. dequant + 2D IDCT → blocks
    4. unblockify → 完整影像
    """
    n_blocks = len(dc_list)
    vecs = np.zeros((n_blocks, 64), dtype=int)
    for i, (d, ac) in enumerate(zip(dc_list, ac_list)):
        vecs[i, 0] = d
        vecs[i, 1:] = rle_decode_ac(ac)
    # inverse zigzag
    inv = np.zeros_like(vecs)
    inv[:, ZZ_IDX] = vecs
    blocks_q = inv.reshape(-1, 8, 8)
    # dequant + IDCT
    deq = blocks_q * qtable
    rec = idct(idct(deq, axis=2, norm='ortho'),
               axis=1, norm='ortho') + 128
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    nh = math.ceil(H/8); nw = math.ceil(W/8)
    blocks_rec = rec.reshape(nh, nw, 8, 8)
    return unblockify(blocks_rec, H, W)

def compute_psnr(orig, recon):
    """
    計算 PSNR（峰值訊號雜訊比）：
      PSNR = 10 * log10( MAX^2 / MSE )
      MAX = 255, MSE = 均方誤差
    """
    mse = np.mean((orig.astype(float) - recon.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10((255.0 ** 2) / mse)

def main(args):
    """
    主流程：
    1. 讀取原圖（gray or color）
    2. 針對每個 channel 做 encode_channel
    3. Huffman encode DC all_flat, decode 回來
    4. decode_channel 還原每個 channel
    5. 輸出重建 .raw + 印 PSNR
    """
    QF = args.quality
    qtable = scale_quant_table(JPEG_LUMA_QT, QF)

    # 讀檔
    if args.mode == 'gray':
        img = np.fromfile(args.input, dtype=np.uint8).reshape(args.size[0], args.size[1])
        channels = [img]
    else:
        arr = np.fromfile(args.input, dtype=np.uint8).reshape(*args.size, 3)
        channels = [arr[:,:,i] for i in range(3)]

    all_dc, all_ac, dims = [], [], []
    # 壓縮每個 channel
    for ch in channels:
        dc, ac, H, W = encode_channel(ch, qtable)
        all_dc.append(dc); all_ac.append(ac); dims.append((H,W))

    # 只對 DC 做 Huffman（示範）
    dc_flat = sum(all_dc, [])
    dc_bits, dc_map = huffman_encode(dc_flat)
    dc_dec = huffman_decode(dc_bits, dc_map)

    recon = []
    idx_dc = 0
    # 還原每個 channel
    for i, ch in enumerate(channels):
        nblk = len(all_dc[i])
        dlist = dc_dec[idx_dc : idx_dc + nblk]
        aclist = all_ac[i]
        idx_dc += nblk
        rec_ch = decode_channel(dlist, aclist, qtable, *dims[i])
        recon.append(rec_ch)
        print(f"[Q={QF}] Channel PSNR: {compute_psnr(ch, rec_ch):.2f} dB")

    # 合併並輸出
    recon_img = recon[0] if args.mode=='gray' else np.stack(recon, axis=2)
    recon_img.tofile(f"{args.input}.fast.Q{QF}.raw")
    with open(f"{args.input}.fast.Q{QF}.pkl", "wb") as f:
        pickle.dump({'dc_map': dc_map}, f)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode',    choices=['gray','color'], required=True)
    p.add_argument('--input',   required=True)
    p.add_argument('--size',    nargs=2, type=int, required=True, metavar=('H','W'))
    p.add_argument('--quality', '-q', type=int, default=90)
    args = p.parse_args()
    main(args)
