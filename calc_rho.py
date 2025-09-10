#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:11:13 2024
author：Wenting Wang
Peking University
email: wangwenting3000@gmail.com
"""

# 计算不同高度的air density


import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colorbar as cb
import matplotlib
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import metpy.calc as mpcalc
from metpy.calc import mixing_ratio_from_specific_humidity
from metpy.units import units
from scipy.interpolate import interp1d
from tqdm import tqdm  # 引入 tqdm 来显示进度条
import matplotlib as mpl



# 定义文件夹路径
data_folder = "/Users/wangwenting/working_papers/AE/data/raw/Apr"
output_folder = "/Users/wangwenting/working_papers/AE/data/outputs/air_density/Apr"

# 定义日期范围
date_range = pd.date_range(start="20230401", end="20230430")



ds_s_z = xr.open_dataset("/Users/wangwenting/working_papers/AE/data/raw/era5.surface_geopotential.nc")
# 地表的位势高度是不随时间变化的，所以只抽出一个静态的二维数据(时间随意给定的），有的海上位势高度为负，统一处理为0.
surface_geopotential = ds_s_z.sel(time='2023-01-01T23:00:00')
surface_geopotential = surface_geopotential.where(surface_geopotential >= 0, 0)
surface_geopotential['alt'] = surface_geopotential['z']/9.80665
# print(surface_geopotential)


# 创建 xarray.DataArray 对象
da_heights = xr.DataArray(data=np.arange(12000, -200, -200), 
                          dims=['height'],
                          coords={'height': np.arange(12000, -200, -200)}, 
                          name='height')

# 创建新的高度坐标：从0到8000米，每隔200米
new_heights = np.arange(0, 12200, 200)[::-1]  # 注意这里是8200，以包含8000米的点


Rd = 287.05

for current_date in date_range:
    date_str = current_date.strftime("%Y%m%d")

    # 打开原始的.nc文件，分别是比湿q和温度T
    ds_q = xr.open_dataset(f"{data_folder}/era5.specific_humidity.{date_str}.nc")
    ds_t = xr.open_dataset(f"{data_folder}/era5.temperature.{date_str}.nc")
    ds_z = xr.open_dataset(f"{data_folder}/era5.geopotential.{date_str}.nc")

    
    rho = ds_q['level']*100 / ( Rd * (1+0.608*ds_q['q'])*ds_t['t'] )
    
    # 将保存为原始的格式
    air_density = xr.DataArray(
        data=rho,
        dims=['level', 'time', 'latitude', 'longitude'],
        coords={
            'level': ds_q['level'],
            'time': ds_q['time'],
            'latitude': ds_q['latitude'],
            'longitude': ds_q['longitude'],
        },
        name='rho')
    ds_rho = xr.Dataset({'rho': air_density})
    # 将风速保存为nc文件
    ds_rho.to_netcdf(f"{output_folder}/era5.air_density.{date_str}.nc")    
    
    
    # 得到位势高度
    ds_z['alt'] = ds_z['z']/9.80665
    
        
    
    # 获取经纬度和时间点
    lons = ds_q.longitude.values
    lats = ds_q.latitude.values
    times = ds_q.time.values
    
    # lon = 80
    # lat = 40
    # time = '2023-01-01T23:00:00'
    
    # 进度条设置
    total_iterations = len(lons) * len(lats) * len(times)
    pbar = tqdm(total=total_iterations, desc='Processing')
    
    
    
    # 对风速进行线性插值处理
    def linear_interpolation(data, old_coords, new_coords):
        # 创建一个插值函数
        interpolator = interp1d(old_coords, data, kind='linear', fill_value='extrapolate')
        # 对新的坐标进行插值
        interpolated_data = interpolator(new_coords)
        return interpolated_data
    
    
    
    
    # 将空气密度保存为原始的格式
    interp_rho = xr.DataArray(
        data=np.nan,
        dims=['time', 'height', 'latitude', 'longitude'],
        coords={
            'time': ds_q['time'],
            'height': da_heights,
            'latitude': ds_q['latitude'],
            'longitude': ds_q['longitude'],
        },
        name='rho')
    interp_rho = xr.Dataset({'rho': interp_rho})
    
    
    
    # 读取气压层
    pressure_levels = ds_q['level'].values
    

    
    # 对每个经纬度点和时间点进行高度计算和插值
    for lon in lons:
        for lat in lats:
            for time in times:
                test_ds_rho = ds_rho.rho.sel(longitude=lon,latitude=lat,time=time)
                test_ds_z  = ds_z.alt.sel(longitude=lon,latitude=lat,time=time)
                
    
    
                # 计算出每个气压层对应的高度
                heights = test_ds_z.values
                # 这里会出现位势高度为负的情况，是因为地表气压小于个pressure levels的值，即在地表以下，为负的层没有实际意义。
                s_height = heights - surface_geopotential.alt.sel(longitude=lon,latitude=lat).values
                
                # 添加height维度,每个 'level' 对应一个特定的距离地表的高度值 
                test_ds_rho.coords['height'] = ('level', s_height)
    
                # 在 test_ds_ws 上进行插值处理
                interpolated_rho = xr.DataArray(
                    linear_interpolation(test_ds_rho, test_ds_rho.height, new_heights),
                    dims=('height',),
                    coords={'height': new_heights},)
    
                # 创建新的数据集并保存
                # interpolated_ds = xr.Dataset({'ws': interpolated_ws})
                interp_rho['rho'].loc[dict(longitude=lon, latitude=lat, time=time)] = interpolated_rho.values
    
                # 更新进度条
                pbar.update(1)
    
    
    # interp_ws.to_netcdf('/Users/wangwenting/working_papers/AE/data/outputs/interp_ws.20230810.nc')


    output_filename = f"{output_folder}/interp_rho.{date_str}.nc"
    interp_rho.to_netcdf(output_filename)



    # 关闭进度条
    pbar.close()


    print(date_str)






# # 可视化----------------------------------------------------------------------------------------
# matplotlib.rcParams['axes.unicode_minus'] = False
# font = {'font.family': 'Times New Roman',
#         'mathtext.fontset': 'stix',
#         'font.size': 12}
# # matplotlib.rcParams.update(font)
# plt.rcParams.update(font)

# # mpl.rcParams['pcolor.shading'] = 'auto'

# def plot_ahi(Lon, Lat, type, ax, date='', **kwargs):
#     ax.coastlines(resolution='50m', zorder=3, lw=0.5)
#     ax.add_feature(cfeature.BORDERS, zorder=3, lw=0.5)
#     reader = shpreader.Reader('/Users/wangwenting/material/map/NCL-Chinamap-master/cnmap/cnhimap.shp')
#     counties = list(reader.geometries())
#     COUNTIES = cfeature.ShapelyFeature(counties, ccrs.PlateCarree())
#     ax.add_feature(COUNTIES, facecolor='none', lw=0.5, edgecolor='black', zorder=9)

#     box = [70, 140, 0, 60]
#     res = 10
#     xi = np.linspace(box[0], box[1], int((box[1] - box[0]) / res + 1))
#     yi = np.linspace(box[2], box[3], int((box[3] - box[2]) / res + 1))

#     lon_formatter = LongitudeFormatter()
#     lat_formatter = LatitudeFormatter()
#     ax.xaxis.set_major_formatter(lon_formatter)
#     ax.yaxis.set_major_formatter(lat_formatter)
#     ax.set_xticks(xi, crs=ccrs.PlateCarree())
#     ax.set_yticks(yi, crs=ccrs.PlateCarree())
#     ax.set_extent(box, crs=ccrs.PlateCarree())
    
#     sc = ax.pcolormesh(Lon-105, Lat, type, **kwargs)
#     ax.set_title(date,fontsize=12,color="white",
#                     bbox=dict(boxstyle='square,pad=0.3', fc='black', ec='k',lw=0 ,alpha=0.8, pad=0.5),
#                     fontdict={'family': 'Times New Roman', 'size': 14})
#     return sc, ax


# interp_rho = xr.open_dataset('/Users/wangwenting/working_papers/AE/data/outputs/air_density/Jan/interp_rho.20230101.nc')

# rho_400 = interp_rho.sel(time='2023-01-01T12:00:00', height = 5000).rho.values
# rho_1000 = interp_rho.sel(time='2023-01-01T12:00:00', height = 8000).rho.values


# Lon4 = interp_rho.longitude.values
# Lat4 = interp_rho.latitude.values


# proj = ccrs.PlateCarree(central_longitude=105)
# fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection=proj), figsize=(12.5, 5), dpi=200)


# sc, axes = plot_ahi(Lon4, Lat4, rho_400, ax=ax[0], date='              Air density 2023-01-01 12:00:00 height = 5000 m          ', cmap='Spectral_r', vmax=2, vmin=0)


# sc, axes = plot_ahi(Lon4, Lat4, rho_1000, ax=ax[1], date='            Air density 2023-01-01 12:00:00 height = 8000 m           ', cmap='Spectral_r', vmax=2, vmin=0)


# plt.subplots_adjust(wspace=0.03, hspace=0.01) 





























