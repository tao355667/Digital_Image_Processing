import matplotlib.pyplot as plt
font = {'family': 'SimHei', 'weight': 'bold', 'size': '16'}
plt.rc('font', **font)        # 步骤一（设置字体的更多属性）
plt.rc('axes', unicode_minus=False)  # 步骤二（解决坐标轴负数的负号显示问题）
# ...
plt.xlabel("x轴")
plt.ylabel("y轴")
plt.title("标题")
plt.show()
