import subprocess

# 設定你要測試的 QF 清單
qf_list = [90, 80, 50, 20, 10 , 5]

# 設定你的輸入參數
mode = 'color'                # 或 'color'
input_file = 'LenaRGB.raw'      # 或 'baboonRGB.raw'
size = (256, 256)            # H, W
output_psnr_file = f'{input_file}.psnr.txt'

# 清空 PSNR 檔案
open(output_psnr_file, 'w').close()

# 執行每一個 QF
for qf in qf_list:
    print(f"\n🔧 Running QF={qf}...")
    result = subprocess.run(
        [
            'python', 'JPEG.py',
            '--mode', mode,
            '--input', input_file,
            '--size', str(size[0]), str(size[1]),
            '--quality', str(qf)
        ],
        capture_output=True,
        text=True
    )

    # 取出 PSNR 的印出結果
    lines = result.stdout.strip().split('\n')
    psnr_lines = [line for line in lines if 'PSNR' in line]
    print('\n'.join(psnr_lines))

    # 寫入文字檔
    with open(output_psnr_file, 'a') as f:
        f.write(f"QF={qf}\n")
        f.write('\n'.join(psnr_lines) + '\n\n')
