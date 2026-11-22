import numpy as np

# 1. 读取两个 npz，先看里面到底有哪些键
with np.load("output_tokens.npz", allow_pickle=False) as f:
    print("你的输出包含的键:", f.files)
    out_ids, mine = f["ids"], f["output_tokens"]

with np.load("assets/reference_outputs.npz", allow_pickle=False) as f:
    print("参考文件包含的键:", f.files)
    ref_ids, ref = f["ids"], f["output_tokens"]

# 2. 确认图片顺序一致
print("图片 ID 顺序一致:", list(out_ids) == list(ref_ids))
print("形状: 你的 =", mine.shape, " 参考 =", ref.shape)

# 3. 全局误差
diff = np.abs(mine - ref)
print(f"\n全局  max_abs = {diff.max():.3e}")
print(f"全局  mean_abs = {diff.mean():.3e}")

# 4. 每张图的最大误差 + 余弦相似度
print(f"\n{'图片ID':<12}{'max_abs':>12}{'mean_abs':>12}{'cos_sim':>12}")
for i, img_id in enumerate(out_ids):
    d = np.abs(mine[i] - ref[i])
    cos = float(np.dot(mine[i], ref[i]) /
                (np.linalg.norm(mine[i]) * np.linalg.norm(ref[i])))
    print(f"{str(img_id):<12}{d.max():>12.3e}{d.mean():>12.3e}{cos:>12.8f}")
