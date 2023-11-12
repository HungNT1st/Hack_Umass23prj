from scan import *
from pose import *
from merge import *

model_SAM()     
res = model(res)
texture_path = "./"
scale_data = res["lengths"]()
angle_data = res["angles"]()
export_path = "./models/export.obj"
convert2dto3d(texture_path, scale_data, angle_data, export_path)
run(export_path)