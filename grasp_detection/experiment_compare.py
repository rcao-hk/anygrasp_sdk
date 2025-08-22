import numpy as np
import os
import matplotlib.pyplot as plt

def load_res(experiment_root, model_name, split, camera_type):
    path = os.path.join(experiment_root, model_name, f"ap_test_{split}_{camera_type}.npy")
    return np.load(path)  # (S, A, K, M)

def compute_scene_ap(res, topk=50, metric_idx=None):
    """
    输入:
      res: ndarray, shape (S, A, K, M)
      topk: 取前K个抓取
      metric_idx: 选择最后一维(5个指标)中的哪一个；None 表示对最后一维取均值
    返回:
      scene_ap: ndarray, shape (S,), 每个 scene 的 AP
    """
    # 取前 topk
    res = res[:, :, :topk, :]  # (S, A, topk, M)

    # 选择 metric
    if metric_idx is None:
        # 对最后一维(5个metric)取均值
        res_metric = res.mean(axis=-1)  # (S, A, topk)
    else:
        res_metric = res[..., metric_idx]  # (S, A, topk)

    # 对 topk、anno 维度取平均，得到每个 scene 的分数
    scene_ap = res_metric.mean(axis=(1, 2))  # (S,)
    return scene_ap

# def plot_scene_bar(scene_ap_1, scene_ap_2, method1, method2, max_scenes=30, save_path='scene_ap_bar.png'):
#     """
#     将两种方法在前 max_scenes 个 scene 的 AP 画成并排柱状图
#     """
#     S = min(len(scene_ap_1), len(scene_ap_2))
#     S = min(S, max_scenes)
#     x = np.arange(S)

#     width = 0.4
#     plt.figure(figsize=(12, 4.5))
#     plt.bar(x - width/2, scene_ap_1[:S], width=width, label=method1)
#     plt.bar(x + width/2, scene_ap_2[:S], width=width, label=method2)

#     plt.xlabel('Scene ID')
#     plt.ylabel('AP')
#     plt.title(f'Per-scene AP (first {S} scenes)')
#     plt.xticks(x, [str(i) for i in range(S)], rotation=0)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     # plt.show()
#     print(f'[Saved] {save_path}')


from matplotlib.patches import Patch
def plot_scene_bar(scene_ap_1, scene_ap_2, method1, method2,
                   max_scenes=30, save_path='scene_ap_bar.png'):
    """
    将两种方法在前 max_scenes 个 scene 的 AP 画成并排柱状图；
    若 method1 - method2 > 0 视为上涨(蓝)，<0 视为下降(红)，=0 置灰；
    对应 scene 的 x 轴刻度标签按上述颜色着色。
    """
    S = min(len(scene_ap_1), len(scene_ap_2), max_scenes)
    x = np.arange(S)

    # 计算差值
    diff = scene_ap_1[:S] - scene_ap_2[:S]

    width = 0.4
    fig, ax = plt.subplots(figsize=(12, 4.5))
    b1 = ax.bar(x - width/2, scene_ap_1[:S], width=width, label=method1)
    b2 = ax.bar(x + width/2, scene_ap_2[:S], width=width, label=method2)

    ax.set_xlabel('Scene ID')
    ax.set_ylabel('AP')
    ax.set_title(f'Per-scene AP (first {S} scenes)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(S)], rotation=0)

    # 给x轴刻度标签按涨跌着色
    tick_colors = []
    for i, lab in enumerate(ax.get_xticklabels()):
        if diff[i] > 0:
            lab.set_color('blue')   # method1 > method2
            tick_colors.append('blue')
        elif diff[i] < 0:
            lab.set_color('red')    # method1 < method2
            tick_colors.append('red')
        else:
            lab.set_color('gray')   # 相等
            tick_colors.append('gray')

    # 可选：在每个scene上方标注Δ值（保留注释，按需开启）
    # for i in range(S):
    #     ax.text(i, max(scene_ap_1[i], scene_ap_2[i]) * 1.01,
    #             f"Δ={diff[i]:.3f}",
    #             ha='center', va='bottom', fontsize=8,
    #             color=('blue' if diff[i] > 0 else 'red' if diff[i] < 0 else 'gray'))

    # 图例：两方法 + 涨跌颜色说明
    handles = [Patch(color='blue', label=f'{method1} > {method2}'),
               Patch(color='red',  label=f'{method1} < {method2}')]
    leg1 = ax.legend(loc='upper left')
    ax.add_artist(leg1)
    # ax.legend(handles=handles, loc='upper right', title='Scene index color')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()
    print(f'[Saved] {save_path}')
    
    
def find_best_anno(experiment_root, method1, method2, split='seen', camera_type='realsense',
                   topk=50, metric_idx=None):
    """
    返回: 在 (scene, anno) 维度上 method1 - method2 提升最大的索引及分数
    """
    res1 = load_res(experiment_root, method1, split, camera_type)   # (S, A, K, M)
    res2 = load_res(experiment_root, method2, split, camera_type)   # (S, A, K, M)

    # 对齐 scene 数（以防不同文件S不一致）
    S = min(res1.shape[0], res2.shape[0])
    res1 = res1[:S]
    res2 = res2[:S]

    # 取前 topk
    res1 = res1[:, :, :topk, :]
    res2 = res2[:, :, :topk, :]

    # 选 metric
    if metric_idx is None:
        res1_metric = res1.mean(axis=-1)  # (S, A, K)
        res2_metric = res2.mean(axis=-1)  # (S, A, K)
    else:
        res1_metric = res1[..., metric_idx]  # (S, A, K)
        res2_metric = res2[..., metric_idx]  # (S, A, K)

    # 对 topk 取均值，得到每个 (scene, anno) 的分数
    res1_score = res1_metric.mean(axis=-1)  # (S, A)
    res2_score = res2_metric.mean(axis=-1)  # (S, A)

    diff = res1_score - res2_score          # (S, A)

    out = {}
    best_idx = np.unravel_index(np.argmax(diff), diff.shape)
    best_scene, best_anno = map(int, best_idx)
    out['mode'] = 'per-(scene, anno)'
    out['best_scene'] = best_scene
    out['best_anno'] = best_anno
    out['best_improvement'] = float(diff[best_scene, best_anno])

    m1_val = float(res1_score[best_scene, best_anno])
    m2_val = float(res2_score[best_scene, best_anno])
    out['method1_score_at_best'] = m1_val
    out['method2_score_at_best'] = m2_val

    return out


def compute_anno_ap_for_scene(res, scene_idx, topk=50, metric_idx=None):
    """
    res: ndarray (S, A, K, M)
    返回：该 scene 下每个 anno 的 AP，shape (A,)
    计算：先选 metric(或对M取均值)，再对前 topk 取均值
    """
    # 取该 scene
    scene_res = res[scene_idx]         # (A, K, M)
    scene_res = scene_res[:, :topk, :] # (A, topk, M)

    if metric_idx is None:
        # 对最后一维(5个metric)取均值 -> (A, topk)
        scene_metric = scene_res.mean(axis=-1)
    else:
        scene_metric = scene_res[..., metric_idx]  # (A, topk)

    # 对 topk 取均值 -> (A,)
    anno_ap = scene_metric.mean(axis=-1)
    return anno_ap


def plot_anno_ap_for_scene(experiment_root, method1, method2, split, camera_type,
                           scene_idx, topk=50, metric_idx=None,
                           color_ticks=True, save_path=None):
    """
    指定 scene_idx，绘制 method1 vs method2 在每个 anno 的 AP 折线图。
    color_ticks=True: 按差值给 x 轴 anno 索引着色（蓝/红/灰）
    """
    # 读取
    res1 = np.load(os.path.join(experiment_root, method1, f"ap_test_{split}_{camera_type}.npy"))
    res2 = np.load(os.path.join(experiment_root, method2, f"ap_test_{split}_{camera_type}.npy"))

    # 边界检查
    S = min(res1.shape[0], res2.shape[0])
    if not (0 <= scene_idx < S):
        raise IndexError(f"scene_idx 超界: 0 <= scene_idx < {S}, 但收到 {scene_idx}")

    # 计算 per-anno AP
    anno_ap_1 = compute_anno_ap_for_scene(res1, scene_idx, topk=topk, metric_idx=metric_idx)  # (A,)
    anno_ap_2 = compute_anno_ap_for_scene(res2, scene_idx, topk=topk, metric_idx=metric_idx)  # (A,)

    A = min(len(anno_ap_1), len(anno_ap_2))
    anno_ap_1 = anno_ap_1[:A]
    anno_ap_2 = anno_ap_2[:A]

    x = np.arange(A)
    diff = anno_ap_1 - anno_ap_2
    
    sort_idx = np.argsort(-diff)  # 从大到小
    print(f"\n[Scene {scene_idx}] Anno AP difference sorted (method1 - method2):")
    for idx in sort_idx:
        print(f"Anno {idx:03d}: diff={diff[idx]:+.4f}  "
              f"{method1}={anno_ap_1[idx]:.4f}, {method2}={anno_ap_2[idx]:.4f}")

    plt.figure(figsize=(12, 4.5))
    # 两条线（可按需把 marker 去掉或换成更稀疏的）
    plt.plot(x, anno_ap_1, label=method1, linewidth=2.5)
    plt.plot(x, anno_ap_2, label=method2, linewidth=2.5)

    plt.xlabel('Anno Index')
    plt.ylabel('AP')
    plt.title(f'Per-anno AP @ scene {scene_idx}')
    plt.xlim(0, A - 1)

    # x 轴刻度太多就稀释一下（避免过密）
    if A > 60:
        step = max(1, A // 30)
        plt.xticks(np.arange(0, A, step))
    else:
        plt.xticks(x)

    # 给 x 轴刻度按涨跌着色
    if color_ticks:
        ax = plt.gca()
        for i, lab in enumerate(ax.get_xticklabels()):
            if i >= len(diff): break
            if diff[i] > 0:
                lab.set_color('blue')
            elif diff[i] < 0:
                lab.set_color('red')
            else:
                lab.set_color('gray')

    plt.legend(loc='best')
    plt.tight_layout()

    if save_path is None:
        save_path = f"scene_{scene_idx:04d}_per_anno_ap_{method1}_vs_{method2}_{split}.png"
    plt.savefig(save_path, dpi=300)
    # plt.show()
    print(f"[Saved] {save_path}")
    
    
# ===== 使用示例 =====
if __name__ == "__main__":
    experiment_root = "/media/2TB/result/grasp/graspnet_trans_full/15000"
    method1 = "gsnet_virtual_ours_restored" 
    method2 = "gsnet_virtual_ours_restored_conf_0.5"
    # method1 = "gsnet_virtual_ours_restored_conf_0.5"
    # method2 = "gsnet_virtual_ours_restored"
    split = "seen"
    camera_type = "realsense"
    topk = 50
    metric_idx = None   # e.g., 3 表示 AP@0.8；None 表示对5个metric取均值

    # 找 (scene, anno) 上提升最大的样本
    result = find_best_anno(experiment_root, method1, method2, split, camera_type,
                            topk=topk, metric_idx=metric_idx)
    print("[Per (scene, anno)] Best pair:", result)

    # 计算 per-scene AP
    res1 = load_res(experiment_root, method1, split, camera_type)
    res2 = load_res(experiment_root, method2, split, camera_type)

    print('mean ap {}:'.format(method1), res1[:, :, :topk, :].mean())
    print('mean ap {}:'.format(method2), res2[:, :, :topk, :].mean())
    # 对齐 scene 数
    Smin = min(res1.shape[0], res2.shape[0])
    res1 = res1[:Smin]
    res2 = res2[:Smin]

    scene_ap_1 = compute_scene_ap(res1, topk=topk, metric_idx=metric_idx)  # (S,)
    scene_ap_2 = compute_scene_ap(res2, topk=topk, metric_idx=metric_idx)  # (S,)

    print(f'Per-scene AP {method1}:', scene_ap_1)
    print(f'Per-scene AP {method2}:', scene_ap_2)
    print('Per-scene AP diff:', scene_ap_1 - scene_ap_2)
    # 画出前 30 个 scene 的柱状图
    plot_scene_bar(scene_ap_1, scene_ap_2, method1, method2,
                   max_scenes=30,
                   save_path=f'scene_ap_bar_{method1}_vs_{method2}_{split}.png')

    scene_idx = result['best_scene']
    plot_anno_ap_for_scene(
        experiment_root, method1, method2, split, camera_type,
        scene_idx=scene_idx, topk=topk, metric_idx=metric_idx,
        color_ticks=True,
        save_path=f"scene_{scene_idx:04d}_per_anno_ap_{method1}_vs_{method2}_{split}.png"
    )