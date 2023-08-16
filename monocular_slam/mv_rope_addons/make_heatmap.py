import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 创建一个二维数组作为示例数据
def draw(path, name):
    plt.figure()
    data = np.loadtxt(path)
    data = data.reshape(4,6)
    # 使用seaborn库中的heatmap函数创建热力图
    # sns.heatmap(data)
    sns.heatmap(data, annot=True, xticklabels=["1.0", "1.5", "2.0", "2.5", "3.0", "3.5"], yticklabels=["4", "8", "16", "24"], annot_kws={"fontsize": 14}, cbar=False)
        # 显示图形
    plt.xlabel("Keyframe Threshold", fontsize=14)
    plt.ylabel("Frontend Window Size", fontsize=14)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title(name)
    plt.savefig(name + ".png", dpi=500, bbox_inches='tight')

draw("/home/yangjq/Projects/DROID_NOCS/DROID-SLAM/reconstructions/iou.txt", "Average_iou")
draw("/home/yangjq/Projects/DROID_NOCS/DROID-SLAM/reconstructions/rot.txt", "Average_rotation_error")
draw("/home/yangjq/Projects/DROID_NOCS/DROID-SLAM/reconstructions/trans.txt", "Average_translation_error")