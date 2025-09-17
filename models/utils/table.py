import matplotlib.pyplot as plt
import numpy as np

# 横轴区间标签
bins = ['[0,15)', '[15,30)', '[30,45)', '[45,60)', '[60,75)', '[75,90)', '[90,105)', '[105,∞)']
x = np.arange(len(bins))

# success 数据（假数据，请替换成你自己的）
seqtrack3d = [58, 64, 70, 69, 70, 79, 73, 74]
m2_track =   [54, 60, 63, 60, 61, 73, 68, 70]
st_tracker = [52, 59, 63, 62, 63, 75, 69, 70]
bat =        [38, 45, 48, 44, 45, 65, 41, 60]
cxtrack =    [47, 51, 50, 47, 48, 66, 42, 61]

# tracklet counts
counts = [3000, 192, 91, 45, 43, 26, 17, 246]

fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图（左轴）
bar = ax1.bar(x, counts, width=0.5, color='lightgray', label='Tracklet Counts')
ax1.set_ylabel('Tracklet Counts', color='gray')
ax1.set_ylim(0, 3200)
ax1.tick_params(axis='y', labelcolor='gray')

# 第二纵轴：Success
ax2 = ax1.twinx()
ax2.plot(x, seqtrack3d, 'go-', label='SeqTrack3D')
ax2.plot(x, m2_track, 'b^-', label='M2-Track')
ax2.plot(x, st_tracker, 'r*-', label='STTracker')
ax2.plot(x, bat, 's-', color='orange', label='BAT')
ax2.plot(x, cxtrack, 'p-', color='purple', label='CXTrack')

ax2.set_ylabel('Success')
ax2.set_ylim(30, 85)

# 横轴设置
ax1.set_xticks(x)
ax1.set_xticklabels(bins, rotation=30)
ax1.set_xlabel('Number of points in first template bounding box')

# 图例
fig.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))

plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()