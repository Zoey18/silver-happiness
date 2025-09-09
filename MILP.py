import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import os

# 定义路径
data_dir = "./30_118.5/filtered_data/3000_100_full_year_sorted.csv"
save_dir = "./Results_30_118.5"

# 连续 h 个小时最多调整一次高度
h = 1

# 额定风速
init_wind_rated = 12

# 把风速2次和3次都写在表里
df = pd.read_csv(data_dir)
df['ws2'] = df['ws'] ** 2
df['ws3'] = df['ws'] ** 3

# 设定高度范围和时间步长
heights = np.arange(0, 3001, 100)
timesteps_per_day = 24
days_per_interval = 7*51
timesteps = timesteps_per_day * days_per_interval

# 高度参数
delta_h_max_init = 3000 
init_height = 500  # 初始高度只在第一个周期使用
height_lb = 500
height_ub = 3000

# 能量计算参数
k1_parameter = 0.0579 * 1
k2_parameter = 0.09 * 1
k3_parameter = 0.15 * 1

# 对每个7天的间隔进行优化
num_intervals = len(df) // (timesteps * len(heights))
previous_final_height = init_height  # 初始时使用 init_height
pre_height = previous_final_height  # 仅用于计算目标函数

for interval in range(num_intervals):
    start_idx = interval * timesteps * len(heights)
    end_idx = start_idx + timesteps * len(heights)
    
    # 提取当前时间段的数据
    df_interval = df[start_idx:end_idx]
    
    # 确保数据大小匹配
    if df_interval.shape[0] != timesteps * len(heights):
        print(f"Skipping interval {interval + 1}: data size mismatch.")
        continue
    
    wind = df_interval['ws'].values.reshape(timesteps, len(heights))
    wind_square = df_interval['ws2'].values.reshape(timesteps, len(heights))
    wind_cube = df_interval['ws3'].values.reshape(timesteps, len(heights))
    
    # 创建模型
    model = gp.Model(f"FloatingWindPowerOptimization_Interval_{interval + 1}")
    
    # 添加变量
    height_vars = model.addVars(timesteps, lb=height_lb, ub=height_ub, vtype=GRB.CONTINUOUS, name="height")
    wind_interp = model.addVars(timesteps, name="wind_interp")
    wind_square_interp = model.addVars(timesteps, name="wind_square_interp")
    wind_cube_interp = model.addVars(timesteps, name="wind_cube_interp")
    wind_cube_min = model.addVars(timesteps, name="wind_cube_min")
    select = model.addVars(timesteps, len(heights), vtype=GRB.BINARY, name="select")
    pre_height = previous_final_height  # 仅用于计算目标函数

    # 改变高度功率积分用
    res1 = model.addVars(timesteps, len(heights), vtype=GRB.CONTINUOUS, name="res1")
    res2 = model.addVars(timesteps, len(heights), vtype=GRB.CONTINUOUS, name="res2")
    abs_diff = model.addVars(timesteps, len(heights), vtype=GRB.CONTINUOUS, name="abs_diff")
    
    # 添加约束
    for t in range(timesteps):
        model.addConstr(gp.quicksum(select[t, i] for i in range(len(heights) )) == 1)
    
    for t in range(timesteps):
        model.addConstr(gp.quicksum(select[t, i] * wind[t,i] for i in range(len(heights) )) - wind_interp[t]==0)

    for t in range(timesteps):
            model.addConstr(gp.quicksum(select[t, i] * wind_square[t,i] for i in range(len(heights) )) - wind_square_interp[t]==0)
    for t in range(timesteps):
            model.addConstr(gp.quicksum(select[t, i] * wind_cube[t,i] for i in range(len(heights) )) - wind_cube_interp[t]==0)

    for t in range(timesteps):
        model.addConstr(gp.quicksum(select[t, i] * heights[i] for i in range(len(heights))) - height_vars[t]==0)

    for t in range(timesteps):
        model.addConstr(wind_cube_interp[t] >= wind_cube_min[t], name=f"wind_cube_constr1_{t}")
        model.addConstr(init_wind_rated**3 >= wind_cube_min[t], name=f"wind_cube_constr2_{t}")
    
    for t in range(timesteps):
        for i in range(len(heights)):
            # 计算 res1 和 res2
            model.addConstr(gp.quicksum(select[t, i] for i in range(len(heights))) - res1[t, i]==0)

            if t > 0:
                model.addConstr(gp.quicksum(select[t-1, i] for i in range(len(heights))) - res2[t, i]==0)
            else:
                res2[t, i] = init_height/100  # 对于第一个时间步，没有前一个时间步的值

            # 线性化绝对值函数 |res2 - res1|
            model.addConstr(abs_diff[t, i] >= res2[t, i] - res1[t, i], name=f"abs_diff_pos_{t}_{i}")
            model.addConstr(abs_diff[t, i] >= res1[t, i] - res2[t, i], name=f"abs_diff_neg_{t}_{i}")

    height_diff = model.addVars(timesteps, lb=0, vtype=GRB.CONTINUOUS, name="height_diff")
    bool_height_diff = model.addVars(timesteps, vtype=GRB.BINARY, name="bool_height_diff")
    p_change = model.addVars(timesteps, lb=0, vtype=GRB.CONTINUOUS, name="p_change")
    


    def power_generation(wind_cube_min):
        k1 = k1_parameter
        p_tur = k1 * wind_cube_min
        return p_tur

    def energy_consumption(wind_square):
        k2 = k2_parameter
        p_maint = k2 * wind_square
        return p_maint 

    M = 1e8

    def energy_consumption2(t, abs_diff, wind_square, p_change):
        # k3 参数
        k3 = k3_parameter

        # 线性化 wind_square * height_diff_pos[t]
        wind_square_cum = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"wind_square_height_diff_{t}")
        
        # 线性化 wind_square * height_diff_pos[t]
        model.addConstr(gp.quicksum(abs_diff[t, i] * wind_square[t,i] for i in range(len(heights))) - wind_square_cum==0)
        
        # 将线性化的结果用于能量消耗计算
        model.addConstr(p_change[t] == k3 * wind_square_cum *100 / 3600, name=f"p_change_{t}")

        return p_change[t]

    total_energy = gp.quicksum(
        power_generation(wind_cube_min[t]) - energy_consumption(wind_square_interp[t])
            -energy_consumption2(t, abs_diff, wind_square, p_change)
        for t in range(timesteps)
    )
    
    model.setObjective(total_energy, GRB.MAXIMIZE)

    delta_h_max = delta_h_max_init
    for t in range(1, timesteps):
        model.addConstr(height_vars[t] - height_vars[t-1] <= delta_h_max, name=f"delta_h_max_pos_{t}")
        model.addConstr(height_vars[t-1] - height_vars[t] <= delta_h_max, name=f"delta_h_max_neg_{t}")

    for t in range(timesteps):
        model.addConstr(height_diff[t] <= bool_height_diff[t] * 1e6)
        model.addConstr(height_diff[t] >= bool_height_diff[t])

    for t in range(timesteps - h):
        model.addConstr(gp.quicksum(bool_height_diff[t + i] for i in range(h)) <= 1)

    model.setParam(GRB.Param.MIPGap, 0.01)
    model.setParam(GRB.Param.TimeLimit, 1800)

    model.optimize()
    print(f"Optimization status for interval {interval + 1}: {model.status}")

    optimal_heights = [round(height_vars[t].X, 2) for t in range(timesteps)]
    wind_interp_values = [round(wind_interp[t].X, 2) for t in range(timesteps)]
    height_diff_values = [round(height_diff[t].X, 2) for t in range(timesteps)]
    energy_contributions = [
        round(power_generation(wind_cube_min[t].X) - energy_consumption(wind_square_interp[t].X)
              - energy_consumption2(t, abs_diff, wind_square, p_change).X, 2)
        for t in range(timesteps)
    ]

    data = {
        'Time Step': list(range(timesteps)),
        'Height': optimal_heights,
        'Height Change': height_diff_values,
        'Wind Speed': wind_interp_values,
        'Energy Contribution': energy_contributions
    }
    result_df = pd.DataFrame(data)

    save_path = os.path.join(save_dir, f"3000m_h={h}_optimization_data_interval_{interval + 1}.csv")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_df.to_csv(save_path, index=False)
    print(f"Results saved for interval {interval + 1}")

    # 更新 previous_final_height 为当前间隔的最后高度
    previous_final_height = optimal_heights[-1]
