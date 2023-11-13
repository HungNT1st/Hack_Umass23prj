from scan import *
from pose import *
from merge import *

def main():
  name = model_SAM()   
  input_from_bro(p) 
  res = [] 
  res = model(res)
  scale_data = res["lengths"]()
  angle_data = res["angles"]()
  export_path = "./models/export.obj"
  texture_path = "./clothes/" + name + "_front_body.jpg"
  convert2dto3d(texture_path, scale_data, angle_data, export_path)
  run(export_path, 1)

if __name__ == '__main__':
	main()