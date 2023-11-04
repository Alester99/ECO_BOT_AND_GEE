import http.client
import json
import pandas as pd 
import numpy as np
import sys
import os
import matplotlib.pyplot  as plt
from io import StringIO
import warnings
import inspect
from datetime import datetime
warnings.filterwarnings("ignore")

n_file = "Ukraine_07"
# id	city_name	region_id	region_name	latitude	longitude

def is_json(myjson):
  try:
    json.loads(myjson)
  except ValueError as e:
    return False
  return True
# dirname = os.getcwd()
# print(dirname)
print(sys.argv)
print(os.path.realpath(sys.argv[0]))

# __file__ = '__file__'
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"

sensors = pd.read_excel(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\parsing_4.xlsx")
print(sensors.columns)
sensors = sensors[['id','region_id',"region_name","center_latitude","center_longitude"]]
sensors = sensors.dropna(subset=["center_longitude"])
sensors = sensors.sort_values(by=['id'])


n_file = "Ukraine_10"
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
filename = os.path.join(dirname, f"regions\\data\\all_sensors_data_{n_file}.xlsx")

stations = pd.DataFrame(columns = ['id','region_id',"region_name", 'lon',	'lat',	'phenomenon',	'logged_at',	'value'])
stations.to_excel(filename,index=False)



stations = pd.read_excel(filename)

a = len(stations); c = 10
b = a + c

def elements_to_download(g1,g2,t):

  if len(g2) == 0: return g1 [t+1:]
  else:
    last = g2[len(g2)-1]
    m = max([last,t])
    # return g1[g1.index(last)+1:]
    return g1[np.where(g1 == m)[0][0]+1:]

t = -1


# ['id', 'center_latitude', 'center_longitude', 'city_name',
#        'hromada_name', 'hromada_id', 'region_name', 'region_id', 'type_name',
#        'aqi', 'aqi_is_old', 'aqi_updated_at', 'cnt_devices_with_aqi',
#        'rating_aqi', 'link_maps_aqi', 'link_maps_gamma', 'link', 'wind_power',
#        'wind_power_is_old', 'wind_power_updated_at', 'wind_direction',
#        'wind_direction_is_old', 'wind_direction_updated_at',
#        'wind_direction_grad', 'wind_direction_grad_is_old',
#        'wind_direction_grad_updated_at', 'temperature', 'temperature_is_old',
#        'temperature_updated_at', 'humidity', 'humidity_is_old',
#        'humidity_updated_at', 'pressure', 'pressure_is_old',
#        'pressure_updated_at', 'pressure_pa', 'gamma', 'gamma_is_old',
#        'gamma_updated_at']

s = 0
a = [   2,   18,   19,   20,   21,   22,   23,   24,   25,   26,   27,   28,   29,   30,\
   31,  218,  233,  274,  282, 8775, 8777, 8778, 8779, 8780, 8782, 8786, 8787, 8897,\
 8926, 8927, 8928, 8931, 8934, 8963, 9006, 9034, 9035, 9036, 9039, 9044, 9047, 9055,\
 9104, 9157, 9176, 9187, 9192, 9228, 9341, 9342, 9345, 9347, 9356, 9360, 9363, 9367,\
 9368, 9369, 9371, 9372, 9409, 9448, 9477, 9494, 9495, 9688, 9753]
    
a = [1538, 1411, 1540, 1412, 1542, 1157, 1401, 1290, 1402, 1556, 1175, 1561, 1306,\
     1434, 1436, 1182, 1568, 1569, 1313, 1315, 1444, 1316, 1317, 32, 1577, 1578, 1579,\
    1586, 1590, 1339, 1595, 1341, 1342, 1217, 1345, 1603, 1220, 1605, 1607, 1226, 1611,\
    1228, 1229, 1227, 1359, 1620, 1365, 1498, 1242, 1372, 1371, 1247, 1248, 1249, 1376, 1379,\
    3428, 1381, 1386, 1645, 1646, 1648, 1392, 1522, 1395, 1394, 1269, 1268, 1399, 1529, 1274]     
# satelite_points = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\Satelite_data.csv" ) 
# print(satelite_points[['id','value']])
# a = satelite_points['id'].values
print (len(a))
c = int(len(a)/4)
a1 = a[:c]
a2 = a[c:2*c]
a3 = a[2*c:3*c]
a4 = a[3*c:4*c]

s = 0
conn = http.client.HTTPSConnection("www.saveecobot.com")
for k in a4:#elements_to_download(a,b,0): #download_by_id:
    # f = f + 1
    # print(f"|k = {k} {f}/{l}", end = " ")
    
    #print(sensors["id"].values)
    #for i in sensors["region_id"].values):
    #print(sensors.head())
    # print(sensors)
    i = np.where(sensors["id"] == k)[0][0]
    #print(i)
    #print(sensors["city_name"].values[i])
    id = sensors["id"].values[i]#[0]
    # city_id = sensors["city_id"].values[i]
    # city_name = sensors["city_name"].values[i]
    region_id = sensors["region_id"].values[i]#[0]
    region_name = sensors["region_name"].values[i]
    lon = sensors["center_longitude"].values[i]
    lat = sensors["center_latitude"].values[i]

    headers = {
        'Content-Type': "application/json",
        'apikey': "SORRY api key is a secret"
        }
    
    
    conn.request("GET", f"/api/v1/sensor-archives/{id}", headers=headers)
    
    res = conn.getresponse()
    data = res.read()
    
    
    csvStringIO = StringIO(data.decode("utf-8"))
    
    if len(csvStringIO.getvalue()[0:10]) == 0: continue
    df = pd.read_csv(csvStringIO, sep=",", header=None, on_bad_lines='skip')
    if df[0][0] == 'device_id':
        df.columns = df[0:1].values[0]
        df = df.drop(0)
        phenomenon = "temperature"
        df_ = df.copy()
        df_ = df[df["phenomenon"] == phenomenon]        
        df_ = df_.drop(columns=['value_text'])
        df_["logged_at"] = pd.to_datetime(df_["logged_at"],errors = 'coerce')
        s_left = pd.to_datetime('2020-03-18 08:40:00')
        s_right = pd.to_datetime('2020-03-18 9:00:00')
        df_ = df_[(s_left <= df_["logged_at"]) & (s_right >= df_["logged_at"])]
        df_['value']=pd.to_numeric(df_['value'], errors='coerce')
        df_["id"] = [int(k) for _ in range(len(df_))]
        df_["region_id"] = [region_id for _ in range(len(df_))]
        df_["region_name"] = [region_name for _ in range(len(df_))]
        df_["lon"] = [lon for _ in range(len(df_))]
        df_["lat"] = [lat for _ in range(len(df_))]      
        stations = pd.concat([stations,df_], ignore_index=True)
        filename1 = os.path.join(dirname, f"regions\\\data\\all_sensors_data_{n_file}.xlsx")
        stations.to_excel(filename1,index=False)
        s += 1
        if s == 1000:break

b = pd.read_excel(f"C:/Users/Alex/Desktop/EcoBotApi/regions/data/all_sensors_data_{n_file}.xlsx" )
b = b['id'].unique()

print(elements_to_download(a,b,0))

l = len(elements_to_download(a,b,0))
print(l)

print (filename)


            # print(df_["logged_at"].values[0], df_["logged_at"].values[1])
            # print(f"columns:\n{df_.columns.values},\n index:\n{df_.index.values}")
# dirname = os.path.dirname(__file__)

n_file = "Ukraine_07"
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
filename1 = os.path.join(dirname, f"regions\\data\\all_sensors_data_{n_file}.xlsx")
stations1 = pd.read_excel(filename1)
print(stations1)

n_file = "Ukraine_08"
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
filename1 = os.path.join(dirname, f"regions\\data\\all_sensors_data_{n_file}.xlsx")
stations2 = pd.read_excel(filename1)
print(stations2)

n_file = "Ukraine_09"
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
filename1 = os.path.join(dirname, f"regions\\data\\all_sensors_data_{n_file}.xlsx")
stations3 = pd.read_excel(filename1)
print(stations3)

n_file = "Ukraine_10"
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
filename1 = os.path.join(dirname, f"regions\\data\\all_sensors_data_{n_file}.xlsx")
stations4 = pd.read_excel(filename1)
print(stations4)

stations = pd.concat([stations1,stations2,stations3,stations4], axis=0)

print(stations)
left = pd.to_datetime('2020-03-18 08:40:00')
right = pd.to_datetime('2020-03-18 9:00:00')

stations = stations.drop_duplicates()
print(stations["id"].drop_duplicates().values)
# h1 = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\Satelite_data_16-19.csv")
# h2 = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\Satelite_data_16-19_1.csv")
# s = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_c.csv")

# print(h1.columns)
# # s = pd.concat([df1,df2,df3,df4], axis=0).drop_duplicates()
# h = pd.concat([h1,h2], axis=0)


print(h[h["id"] == 1540][["id","value"]])
print(h[h["id"] == 1611][["id","value"]])
print(h[h["id"] == 1395][["id","value"]])
print(h["id"].values)

a = h["id"].values

a = stations["id"].drop_duplicates().values

for i in a:
    eco = stations[(left <= stations["logged_at"]) & (right >= stations["logged_at"]) & (stations["id"] == i)][["id","lon","lat","logged_at","value"]]
    sate = h[h["id"] == i][["id","value"]].drop_duplicates()
    
    print( sate["value"])
    eco["Eco"] = float(sate["value"])
    eco["Diff"] = abs(eco["value"] - eco["Eco"])
    closest = eco[eco["Diff"] == eco["Diff"].min()]
    
    datetime_obj = datetime.strptime('2020-03-18 08:50:00', '%Y-%m-%d %H:%M:%S')

    # Extract hour and minute and format as HH:MM
    hour_minute_str = datetime_obj.strftime('%H:%M')
    X = eco["logged_at"].dt.strftime('%H:%M')
    new_row = pd.DataFrame([{"logged_at": hour_minute_str}])
    X1 = pd.concat([X,new_row], ignore_index=True)

    
    plt.figure(figsize=(10, 10))
    plt.plot(X, eco["value"], marker='o', linestyle='-',label='EcoBot')
    plt.plot(X,[closest["value"]]*len(X),  linestyle='-',label='Closest value')
    plt.plot(X, [sate["value"]]*len(X), linestyle='-', color = 'red',label='Geospatial')
    plt.plot(X, [sate["value"]]*len(X), linestyle='-', color = 'yellow',label='Geospatial')
    #plt.plot(eco["logged_at"].mean().dt.strftime('%H:%M'),eco["value"].mean(), marker='o', linestyle='-',label='Mean')
    plt.title('EcoBot vs Geospatial data 8:40 - 8:50', fontsize=20)
    
    
    plt.xticks(rotation=90) 
    plt.xlabel('Time', fontsize=14, labelpad=10)
    plt.ylabel('Temperature', fontsize=14, labelpad=10)
    
    
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dirname, f"regions\\data\\Eco_Geo\\{i}.png"))
    plt.show()
 
print (h)

dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
filename = os.path.join(dirname, f"regions\\data\\all_sensors_data_{n_file}.xlsx")

table = pd.DataFrame()    
for i in a:
    eco = stations[(left <= stations["logged_at"]) & (right >= stations["logged_at"]) & (stations["id"] == i)][["id","lon","lat","logged_at","value"]]
    sate = h[h["id"] == i][["id","value"]].drop_duplicates()
    
    

    eco["Eco"] = float(sate["value"])
    eco["Diff"] = abs(eco["value"] - eco["Eco"])
    closest = eco[eco["Diff"] == eco["Diff"].min()]
    
    datetime_obj = datetime.strptime('2020-03-18 08:50:00', '%Y-%m-%d %H:%M:%S')

    # Extract hour and minute and format as HH:MM
    hour_minute_str = datetime_obj.strftime('%H:%M')
    X = eco["logged_at"].dt.strftime('%H:%M')
    new_row = pd.DataFrame([{"logged_at": hour_minute_str}])
    X1 = pd.concat([X,new_row], ignore_index=True)
    
    h_ = h.reset_index(drop=True)
    closest_ = closest.reset_index(drop=True)
    
    # print(h_)
    # print(closest_)
    a1 = h_[h_["id"] ==i][["value"]].values[0][0]
    a2 = closest["value"].values[0]
    print("----------------------------------")
    print(i,a1)
    print(i,a2)
    print(f"|{a1} - {a2}| = {abs(a1-a2)}")
    new_row = pd.DataFrame([{"id" : i, "satelite" : a1, "eco" : a2, "diff" : abs(a1-a2)}])
    table = pd.concat([table,new_row], ignore_index=True)
print (table)
filename1 = os.path.join(dirname, f"regions\\\data\\table.xlsx")
table.to_excel(filename1,index=False)

print(stations[["lon","lat","id"]].drop_duplicates())
# print(stations['id'].unique())
# for i in stations['id'].unique():
#    a = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))][0:1]
#    b = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))][-2:-1]
#    print(pd.concat([a,b])[['id','logged_at','value', 'lon','lat']])
   
# for i,j in zip(stations['lon'].unique(),stations['lat'].unique()):
#     print(i,j)
    
print(filename)
stations['logged_at'] = pd.to_datetime(stations['logged_at'])
start_date = pd.to_datetime('2020-03-15')
end_date = pd.to_datetime('2020-03-20')

start_date = pd.to_datetime('2019-03-15')
end_date = pd.to_datetime('2020-03-20')

print(stations['phenomenon'])

s = stations[['id','lon', 'lat','value','logged_at','phenomenon']]
s = s[s['phenomenon'] =='temperature'][['id','lon', 'lat','value','logged_at']]
print(s)

# print(stations[['id','lon', 'lat','value','logged_at']].iloc[(stations['phenomenon']=='temperature')])
# print(stations[['id','lon', 'lat','value','logged_at']].iloc[(stations['phenomenon']=='temperature')]\
# [(stations['logged_at'] >= start_date) & (stations['logged_at'] <= end_date)])
# print(satelite_points[["id","value"]])

start_date = pd.to_datetime('2020-03-15')
end_date = pd.to_datetime('2020-03-20')


print(s[(s['logged_at'] >= start_date) & (s['logged_at'] <= end_date)])

start_date = pd.to_datetime('2019-03-15')
end_date = pd.to_datetime('2020-03-20')


##################################################################
##################################################################    
##################################################################    

# from osgeo import gdal   
# rastername = f"raster\\Downscaled_LST_usingS2_10m_2020-3-15_20_square.tif"
# raster_ds =  gdal.Open(os.path.join(dirname, rastername))
# print("Відкрито файл")

# # Отримуємо кількість стовпців та рядків у растровому файлі
# cols = raster_ds.RasterXSize
# rows = raster_ds.RasterYSize



    
# print(f"{cols} {rows}")
# print(f"cols x rows = {cols*rows}")
# # Отримуємо інформацію про геопросторове положення растрового файлу
# transform = raster_ds.GetGeoTransform()
# print("Трансформовано")
# #Отримуємо координати центру кожного пікселя та значення 
# d = 1
# data = []
# # for y in range(rows)[::d]:
# #     for x in range(cols)[::d]:
    
# # for y in stations['lon'].unique():
# #     for x in stations['lat'].unique():
# for x, y in zip(stations['lon'].unique(),stations['lat'].unique()):
#         # Отримуємо координати центру пікселя
#         px = x + 0.5
#         py = y + 0.5
#         x_geo = transform[0] + px * transform[1] + py * transform[2]
#         y_geo = transform[3] + px * transform[4] + py * transform[5]
        
#         # Отримуємо значення пікселя
#         band = raster_ds.GetRasterBand(1)
#         pixel_value = band.ReadAsArray(x, y, 1, 1)[0,0]
        
#         # Виводимо отримані значення
#         data.append([x, y, pixel_value])
#         #print(f'Координати: ({x_geo}, {y_geo}), Значення пікселя: {pixel_value}')
        
# # Створення DataFrame Pandas зі списку даних
# df = pd.DataFrame(data, columns=['Longitude', 'Latitude', 'Value'])
# print(df)
    
# print(band.ReadAsArray(49.82478829999999, 49.82478829999999, 1, 1)[0,0])


stations['logged_at'] = pd.to_datetime(stations['logged_at'])



print(stations['id'].unique())
for i in stations['id'].unique():
    # a = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))][0:4]
    # b = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))][-4:-1]
    # print(pd.concat([a,b])[['id','logged_at','value', 'lon','lat']])
    result = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))]\
    [(stations['logged_at'] >= start_date) & (stations['logged_at'] <= end_date)]
    print(result)

for i in [27,28,30]:
    # a = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))][0:4]
    # b = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))][-4:-1]
    # print(pd.concat([a,b])[['id','logged_at','value', 'lon','lat']])
    result = stations.iloc[np.where((stations['id']==i) & (stations['phenomenon']=='temperature'))]\
    [(stations['logged_at'] >= start_date) & (stations['logged_at'] <= end_date)]
    print(result)
    
   
# df1 = pd.read_excel(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_Ukraine_02.xlsx" )
# df2 = pd.read_excel(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_Ukraine_03.xlsx" )
# df3 = pd.read_excel(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_Ukraine_04.xlsx" )
# df4 = pd.read_excel(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_Ukraine_05.xlsx" )
h1 = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\Satelite_data_16-19.csv")
h2 = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\Satelite_data_16-19_1.csv")
s = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_c.csv")

print(h1.columns)
# s = pd.concat([df1,df2,df3,df4], axis=0).drop_duplicates()
h = pd.concat([h1,h2], axis=0)
h = pd.read_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\Satelite_data_18-19_r1.csv")


print(h[h["id"] == 1540][["id","value"]])
print(h[h["id"] == 1611][["id","value"]])
print(h[h["id"] == 1395][["id","value"]])

print(s.columns)

print(s['Unnamed: 0'])

print(s["id"].drop_duplicates())
print(s[["lon","lat"]])
print(s[s["id"]<= 1565])
print(h.columns)
print(h["id"].values)
for i in a:
    v = h[h["id"] == i][["id","value"]]
    # if is_number(v):
    #     print(v)
    print(v)
    
a = [1540,1611,1395]    
print(h[h["id"] == 1540][["id","value"]])
print(h[h["id"] == 1611][["id","value"]])
print(h[h["id"] == 1395][["id","value"]])

print(stations[["id","lon","lat"]].drop_duplicates())
print(stations)

#print(h)
# t = s.iloc[np.where((s['id']==1584) & (s['phenomenon'] =='temperature'))]
# t = t.iloc[np.where((t['logged_at'] >= f'2020-3-16') & (t['logged_at'] <= f'2020-3-19'))]
# t = t.drop_duplicates()

# print(t[["lon","lat"]])

# print(s.iloc[np.where((s['id']==1584) & (s['phenomenon'] =='temperature') \
#               & (s['logged_at'] >= f'2020-3-16') & (s['logged_at'] <= f'2020-3-19')\
#              )])


# def get_lon_json (row):
#     # print(type(row[".geo"]))
#     return json.loads(row[".geo"])["coordinates"][0]

# def get_lat_json (row):
#     return json.loads(row[".geo"])["coordinates"][1]

# # h["lon"] = json.loads(h[".geo"].values[0])["coordinates"][0]
# # h["lat"] = json.loads(h[".geo"].values[0])["coordinates"][1]
# h["lon1"] = h.apply(get_lon_json,axis = 1)
# h["lat1"] = h.apply(get_lat_json,axis = 1)
# print(h[["lon1","lat1"]])

# h = pd.merge(s,h[["id","lon1","lat1"]], on='id', how='inner').drop_duplicates()#[["lon","lat","lon1","lat1"]]
# h["lon"] = h["lon1"]
# h["lat"] = h["lat1"]
# print(h.columns)

# h =  h[['id', 'region_id', 'region_name', 'lon', 'lat', 'phenomenon','logged_at', 'value']]

# print(h)


# h.to_csv(r"C:\Users\Alex\Desktop\EcoBotApi\regions\data\all_sensors_data_c.csv")


# c = h.iloc[np.where((h['id']==1584))][".geo"].values[0]
# c = json.loads(c)
# print(c["coordinates"][0],c["coordinates"][1])
# print(type(c))
            
dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
df = pd.DataFrame()
# df = pd.DataFrame(columns = ['id', 'TOTAL% diff1','TOTAL% diff2', '% diff1 max','% diff1 min', '% diff2 max','% diff2 min'])
a = []

for i in range(1,31):
    start = f'2020-3-{i}'
    end = f'2020-3-{i}'
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    filename2 = os.path.join(dirname, f"regions\\data\\compare\\{start}.xlsx")
    # print(filename2)
    
    s1 = s[["id","lon","lat","value","logged_at","phenomenon"]][s['phenomenon']=='temperature']
    # print(s1['logged_at'])
    
    s1['logged_at'] = pd.to_datetime(s1['logged_at'])
    s1 = s1[(s1['logged_at'] >= start_date) & (s1['logged_at'] <= end_date)].drop_duplicates()
        
    
    h1= h[["id","value"]].drop_duplicates()
    
    
    r = pd.merge(s1,h1, on='id', how='inner').drop_duplicates()
    length = len(r)
    print("DATE ",start)
    print("START LENGHT ",length)
    print(r["id"].values)
    a = list(set(r["id"].values))
    r = r.rename(columns={'value_x': 'ECO_BOT_BATA','value_y': 'SATELIE_DATA'})
    r = r[['id', 'lon', 'lat',  'logged_at','ECO_BOT_BATA', 'SATELIE_DATA']]
    # print(r)
    # r['square of the difference'] = (r['ECO_BOT_BATA'] - r['SATELIE_DATA']) ** 2
    # r['deviation -**2'] = (r['square of the difference'].sum())/len(r)
    # print(r)
    r['% diff1'] = abs(round((r['ECO_BOT_BATA'] - r['SATELIE_DATA'])/r['ECO_BOT_BATA']*100,2))
    r = r.iloc[np.where(r['% diff1']<=100)]
    T1 = r['% diff1'].sum()/len(r)
    r["TOTAL% diff1"] = T1
    # print(r[['ECO_BOT_BATA','SATELIE_DATA','% diff1',"TOTAL% diff1"]].head(1))
    # r['% diff2'] = abs(round((r['ECO_BOT_BATA'] - r['SATELIE_DATA'])/r['SATELIE_DATA']*100,2))
    # T2 = r['% diff2'].sum()/len(r)
    # r["TOTAL% diff2"] = T2
    # print(r[['ECO_BOT_BATA','SATELIE_DATA','% diff2',"TOTAL% diff2"]].head(1))
    # print()
    
    r['(max + min)/2_1'] = (r['% diff1'].max() + r['% diff1'].min())/2
    # r['(max + min)/2_2'] = (r['% diff2'].max() + r['% diff2'].min())/2
    # print(r[['(max + min)/2_1','(max + min)/2_2']].head(1) )
    max1 = r['% diff1'].max(); min1 = r['% diff1'].min()
    # print(max1,min1)
    # max2 = r['% diff2'].max(); min2 = r['% diff2'].min()
    # print(max2,min2)
    new_row1 = {'day': start, 'TOTAL% diff1': T1,'% diff1 max':max1,'% diff1 min': min1}
    # new_row1 = {'id': i, 'TOTAL% diff1': T1,'TOTAL% diff2': T2, '% diff1 max':max1,'% diff1 min': min1,\
    #             '% diff2 max': max2,'% diff2 min': min2}

#///////////////////////////////////////////////////////////////////////////////
    # c1 = round(((r['% diff1'] >= 0.00) & (r['% diff1'] <= 20.00)).sum()/len(r),1)*100
    # c2 = round(((r['% diff1'] >= 20.00) & (r['% diff1'] <= 40.30)).sum()/len(r),1)*100
    # c3 = round(((r['% diff1'] >= 40.00) & (r['% diff1'] <= 50.30)).sum()/len(r),1)*100
    # c4 = round(((r['% diff1'] >= 50.00) & (r['% diff1'] <= 80.30)).sum()/len(r) ,1)*100
    # c5 = round(((r['% diff1'] >= 80.00)).sum()/len(r),2)*100
    # # print(c1,c2,c3,c4,c5)
    # new_row1["0-20_1"] = c1
    # new_row1["20-40_1"] = c2
    # new_row1["40-60_1"] = c3
    # new_row1["60-80_1"] = c4
    # new_row1["80_1"] = c5
    # new_row1["Not del"] = round(len(r)/length ,1)*100
#///////////////////////////////////////////////////////////////////////////////    
    n1 = 10.00; n2 = 20.00; n3 = 30.00; n4 = 40.00; n5 = 50.00; n6 = 60.00; n7 = 70.00; n8 = 80.00; n9 = 90.00
    
    c1 = round(((r['% diff1'] >= 0.00) & (r['% diff1'] <= n1)).sum()/len(r),1)*100
    c2 = round(((r['% diff1'] >= n1) & (r['% diff1'] <= n2)).sum()/len(r),1)*100
    c3 = round(((r['% diff1'] >= n2) & (r['% diff1'] <= n3)).sum()/len(r),1)*100
    c4 = round(((r['% diff1'] >= n3) & (r['% diff1'] <= n4)).sum()/len(r) ,1)*100
    c5 = round(((r['% diff1'] >= n4)).sum()/len(r),2)*100
    
    c6 = round(((r['% diff1'] >= n4) & (r['% diff1'] <= n5)).sum()/len(r) ,1)*100
    c7 = round(((r['% diff1'] >= n5) & (r['% diff1'] <= n6)).sum()/len(r) ,1)*100
    c8 = round(((r['% diff1'] >= n6) & (r['% diff1'] <= n7)).sum()/len(r) ,1)*100
    c9 = round(((r['% diff1'] >= n8) & (r['% diff1'] <= n9)).sum()/len(r) ,1)*100
    c10 = round(((r['% diff1'] >= n8) & (r['% diff1'] <= n9)).sum()/len(r) ,1)*100
    # print(c1,c2,c3,c4,c5)
    new_row1[f"0-{n1}"] = c1
    new_row1[f"{n1}-{n2}"] = c2
    new_row1[f"{n2}-{n3}"] = c3
    new_row1[f"{n3}-{n4}"] = c4
    new_row1[f"{n4}+"] = c5
    new_row1[f"{n4}-{n5}"] = c6
    new_row1[f"{n5}-{n6}"] = c7
    new_row1[f"{n5}-{n7}"] = c8
    new_row1[f"{n7}-{n8}"] = c9
    new_row1[f"{n9}-100.0"] = c10
    
    
    new_row1["Not del"] = round(len(r)/length ,3)*100
    new_row1[f"<{n4}"] = c1+c2+c3+c4
    new_row1[f"<{n5}"] = c1+c2+c3+c4+c6
    
    # c1 = round(((r['% diff2'] >= 0.00) & (r['% diff2'] <= 20.00)).sum()/len(r),2)*100
    # c2 = round(((r['% diff2'] >= 20.00) & (r['% diff2'] <= 40.30)).sum()/len(r),2)*100
    # c3 = round(((r['% diff2'] >= 40.00) & (r['% diff2'] <= 60.30)).sum()/len(r),2)*100
    # c4 = round(((r['% diff2'] >= 60.00) & (r['% diff2'] <= 80.30)).sum()/len(r) ,2) *100
    # c5 = round(((r['% diff2'] >= 80.00)).sum()/len(r),2)*100
    # print(c1,c2,c3,c4,c5)
    # new_row1["0-20_2"] = c1
    # new_row1["20-40_2"] = c2
    # new_row1["40-60_2"] = c3
    # new_row1["60-80_2"] = c4
    # new_row1["80_2"] = c5
    
    
    
    r.to_excel(filename2)
    df = pd.concat([df, pd.DataFrame([new_row1])], ignore_index=True)
    print("LENGHT",len(r))
    # df.append(new_row1, ignore_index=True)
df.to_excel(os.path.join(dirname, f"regions\\data\\compare\\stat.xlsx"))

print(df.columns)
print(df[[ f"0-{n1}", f"{n1}-{n2}", f"{n2}-{n3}",f"{n3}-{n4}", f"{n4}+", 'Not del',f"<{n4}",f"<{n5}"]])
print ("END",len(df))

def plot_stat_not_del (df):
    
    
    df['day_'] = pd.to_datetime(df['day'])
    day = df['day']
    not_del = df['Not del']
    
    start_date = pd.to_datetime(f'2020-3-16')
    end_date = pd.to_datetime(f'2020-3-19')
    
    
    df1 = df[(df['day_'] >= start_date) & (df['day_'] <= end_date)]
    
    
    day_ = df1['day']
    not_del_ = df1['Not del']    
    

    plt.figure(figsize=(10, 8))
    plt.plot(day, not_del, marker='o', linestyle='-',label='Accuracy')
    plt.plot(day_, not_del_, marker='o', linestyle='-', color = 'red',label='Data from satelites')
    plt.title('Percent of points not removed due to large inaccuracy', fontsize=20)
    
    plt.xticks(rotation=90) 
    plt.xlabel('Day', fontsize=14, labelpad=10)
    plt.ylabel('Count %', fontsize=14, labelpad=10)
    
    
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dirname, f"regions\\data\\compare\\Percentage_correct.png"))
    plt.show()
    
    
    
    
plot_stat_not_del (df)

def plot_stat_40 (df):
    
    
    df['day_'] = pd.to_datetime(df['day'])
    day = df['day']
    not_del = df['<40.0']
    
    start_date = pd.to_datetime(f'2020-3-16')
    end_date = pd.to_datetime(f'2020-3-19')
    
    
    df1 = df[(df['day_'] >= start_date) & (df['day_'] <= end_date)]
    
    
    day_ = df1['day']
    not_del_ = df1['<40.0']    
    


    plt.figure(figsize=(10, 8))
    plt.plot(day, not_del, marker='o', linestyle='-',label='Accuracy')
    plt.plot(day_, not_del_, marker='o', linestyle='-', color = 'red',label='Data from satelites')
    plt.title('Percent of points where percentage difference is not too big<40.0', fontsize=20)
    
    plt.xticks(rotation=90) 
    plt.xlabel('Day', fontsize=14, labelpad=10)
    plt.ylabel('Count', fontsize=14, labelpad=10)
    
    
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(dirname, f"regions\\data\\compare\\less 40.0.png"))
    plt.show()
    
    
    
plot_stat_40 (df.iloc[np.where(df['Not del']>=40)])

def plot_stat_50 (df):
    
    
    df['day_'] = pd.to_datetime(df['day'])
    day = df['day']
    not_del = df['<50.0']
    
    start_date = pd.to_datetime(f'2020-3-16')
    end_date = pd.to_datetime(f'2020-3-19')
    
    
    df1 = df[(df['day_'] >= start_date) & (df['day_'] <= end_date)]
    
    
    day_ = df1['day']
    not_del_ = df1['<50.0']    
    


    plt.figure(figsize=(10, 8))
    plt.plot(day, not_del, marker='o', linestyle='-',label='Accuracy')
    plt.plot(day_, not_del_, marker='o', linestyle='-', color = 'red',label='Data from satelites')
    plt.title('Percent of points where percentage difference is not too big<50.0', fontsize=20)
    
    plt.xticks(rotation=90) 
    plt.xlabel('Day', fontsize=14, labelpad=10)
    plt.ylabel('Count', fontsize=14, labelpad=10)
    
    
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(dirname, f"regions\\data\\compare\\less 50.0.png"))
    plt.show()
    
    
plot_stat_50 (df.iloc[np.where(df['Not del']>=40)])

print(len(df))
print(df)
print(a)



# def plot_stat_not_del (df):
    
    
#     df['day_'] = pd.to_datetime(df['day'])
#     day = df['day']
#     not_del = df['Not del']
    
#     start_date = pd.to_datetime(f'2020-3-16')
#     end_date = pd.to_datetime(f'2020-3-19')
    
    
#     df1 = df[(df['day_'] >= start_date) & (df['day_'] <= end_date)]
    
    
#     day_ = df1['day']
#     not_del_ = df1['Not del']    
    
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

#     ax1.plot(day, not_del, marker='o', linestyle='-',label='Accuracy')
#     ax1.plot(day_, not_del_, marker='o', linestyle='-', color = 'red',label='Data from satelites')
#     ax1.set_title('Number of points not removed due to large inaccuracy', fontsize=20)
    
#     ax1.tick_params(axis='x', rotation=90)
#     ax1.set_xlabel('Day', fontsize=14, labelpad=10)
#     plt.set_ylabel('Count', fontsize=14, labelpad=10)
    
    
#     ax1.grid(True)
#     ax1.legend()
#     plt.show()
    
    
    
# plot_stat_not_del (df)
    
# import ee
# import folium

# ee.Authenticate()
# ee.Initialize()

# # Load the Landsat image
# image = ee.Image('LANDSAT/LC08/C01/T1_TOA/LC08_044034_20140318')

# # Define the visualization parameters.
# vizParams = {
#     'bands': ['B5', 'B4', 'B3'],
#     'min': 0,
#     'max': 0.5,
#     'gamma': [0.95, 1.1, 1]
# }

# # Get the Earth Engine image as a URL
# image_url = image.visualize(**vizParams).getThumbURL()

# # Create a map centered on San Francisco Bay
# map_center = [37.5010, -122.1899]
# Map = folium.Map(location=map_center, zoom_start=10)

# # Add the image URL as a TileLayer to the map
# folium.TileLayer(
#     tiles=image_url,
#     attr='Google Earth Engine',
#     name='false color composite',
# ).add_to(Map)

# # Display the map
# display(Map)



# dirname = r"C:\Users\Alex\Desktop\EcoBotApi"
# excel = f"regions\data\Kiev_teritory_1.xlsx"
# csv =  f"regions\data\Kiev_teritory_1.csv"

# filename = os.path.join(dirname, excel)

# df = pd.read_excel(filename)

# csv_filename = os.path.join(dirname, csv)
# df.to_csv(csv_filename, index=False)
