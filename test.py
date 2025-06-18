# 定义两个证据源的基本概率分配
m1 = {'A': 0.6, 'B': 0.5, 'A∪B': 0.8}
m2 = {'A': 0.9, 'B': 0.8, 'A∪B': 0.1}

# 初始化组合后的置信度
m = {'A': 0, 'B': 0, 'A∪B': 0}
m['A'] = m1['A']*m2['A'] + m1['A']*m2['A∪B'] + m1['A∪B']*m2['A']
m['B'] = m1['B']*m2['B'] + m1['B']*m2['A∪B'] + m1['A∪B']*m2['B']
m['A∪B'] = m1['A∪B']*m2['A∪B']

# 计算归一化因子K
K = 1 - (m1['A']*m2['B'] + m1['B']*m2['A'])

# 归一化组合后的置信度
m_normalized = {k: v / K for k, v in m.items()}

# 输出归一化后的置信度结果
for hypothesis, belief in m_normalized.items():
    print(f"{hypothesis}: {belief:.4f}")
