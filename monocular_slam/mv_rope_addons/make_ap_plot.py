import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

legend_handles = []  # List to store legend handles
trans_dict1 = {
    1.0: "bottle",
    2.0: "bowl",
    3.0: "camera",
    4.0: "can",
    5.0: "laptop",
    6.0: "mug",
}
trans_dict = {
    "1.0": "bottle",
    "2.0": "bowl",
    "3.0": "camera",
    "4.0": "can",
    "5.0": "laptop",
    "6.0": "mug",
}
directory = "/home/yangjq/Projects/DROID_NOCS/NOCS/logs"
data = np.loadtxt("/home/yangjq/Projects/DROID_NOCS/NOCS/logs/test/best_result_zd.txt")
parser = argparse.ArgumentParser(description="nocs inference")
plt.figure()
parser.add_argument("--checkpoint", type=str, default="test")
args = parser.parse_args()
path = os.path.join(directory, args.checkpoint)
line_path = os.path.join(path, "curve1.png")
box_path = os.path.join(path, "boxplot.png")
with open("/home/yangjq/Projects/DROID_NOCS/DROID-SLAM/results.txt", "a+") as file:
    # Go to the beginning of the file
    file.seek(0)

    # Read the original content of the file
    original_content = file.read()

    # Write new data after the original content
    file.write(args.checkpoint)
cat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# Define the palette as a list to specify exact values
rocket_color = sns.color_palette("rocket")
for c in cat:
    each_cat_data_mask = data[:, 1] == c
    each_cat_data = data[each_cat_data_mask]
    x = []
    thres_iou = []
    thres_rot = []
    thres_trans = []
    ap_iou = []
    ap_rot = []
    ap_trans = []
    for i in range(100):
        percent = i / 100
        thres_iou.append(i)
        ap_iou.append(np.sum(each_cat_data[:, 2] > percent) / each_cat_data.shape[0])
    for i in range(100):
        percent = i * 60 / 100
        thres_rot.append(i * 60 / 100)
        ap_rot.append(np.sum(each_cat_data[:, 3] < percent) / each_cat_data.shape[0])

    for i in range(100):
        percent = i * 0.1 / 100
        thres_trans.append(i * 0.1 / 100)
        ap_trans.append(np.sum(each_cat_data[:, 4] < percent) / each_cat_data.shape[0])
    num = int(c)



    plt.figure(1, figsize=(2, 3))
    plt.grid(True)
    # Plotting the first subplot
    (line_iou,) = plt.plot(
        thres_iou, ap_iou, label=trans_dict1[c], color=rocket_color[num - 1], linewidth=2.5
    )
    plt.xlabel("IoU", fontsize=12)
    plt.ylabel("AP", fontsize=12)
    plt.xlim(0, 100)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)



    plt.figure(2)
    plt.grid(True)

    # Plotting the second subplot
    (line_rot,) = plt.plot(
        thres_rot, ap_rot, label=trans_dict1[c], color=rocket_color[num - 1], linewidth=2.5
    )
    plt.xlabel("Rot", fontsize=12)
    plt.xlim(0, 50)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)



    plt.figure(3)
    plt.grid(True)
    plt.xlim(0, 0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Plotting the third subplot
    (line_trans,) = plt.plot(
        thres_trans, ap_trans, label=trans_dict1[c], color=rocket_color[num - 1], linewidth=2.5
    )
    legend_handles.append(line_trans)  # Add line handle to legend_handles list
    plt.xlabel("Trans", fontsize=12)
# Adjusting the layout to make space for legends outside the plots
plt.figure(3)
# plt.subplots_adjust(right=0.78)

plt.legend(fontsize=12)
# plt.legend(handles=legend_handles, title="Coherence", bbox_to_anchor=(1.05, 1))
sns.set()


plt.figure(1)
plt.savefig('pictures/iou.png', dpi=400, bbox_inches='tight')
plt.figure(2)
plt.savefig('pictures/rot.png', dpi=400, bbox_inches='tight')
plt.figure(3)
plt.savefig("pictures/trans.png", dpi=400, bbox_inches='tight')
# plt.figure()
# data = np.loadtxt("ratio.txt")
# import seaborn as sns
# import pandas as pd

# df = pd.DataFrame(data, columns=["X", "Y"])

# ax = sns.boxplot(x="X", y="Y", data=df)


# ax.set_xticklabels([trans_dict[label.get_text()] for label in ax.get_xticklabels()])

# plt.xlabel("Category")
# plt.ylabel("inlier-ratio")
# plt.title("Boxplot")

# plt.savefig(box_path)
