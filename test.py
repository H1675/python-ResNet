import cst
import cst.results
import cst.interface
import os
import numpy as np

# a = 10
# for i in range(10):
#     print(i)

# print('StoreParameter','StoreParameter("D1","%d")' % 2)

# # 获取当前路径
# path = os.getcwd()
#
# # 临时文件路径
# temp = path + "\\temp"
# temp_name = temp + "\\temp.cst"
#
# # 工程文件路径
# filename = path + "\model\yagi_antenna.cst"
#
# para_name = ["D1","D2","L0","L1","L2","R1"]
#
# cst = cst.interface.DesignEnvironment()
# # 打开目标工程
# project = cst.open_project(filename)
# # 另存目标工程
# project.save(temp_name)
# # 修改参数开始仿真
# modeler = project.modeler
#
# temp = 'StoreParameter(\"' + para_name[0] + '\",' + '\"%f\"' % 20 + ')'
# modeler.add_to_history('StoreParameter', temp)


project = cst.results.ProjectFile(r"D:\py_project\my_resnet\model\yagi_antenna.cst")
dir = project.get_3d().get_result_item("Tables\\1D Results\\Directivity,Phi=0.0")

dir_x = dir.get_xdata()
dir_y = np.array(dir.get_ydata()).reshape(-1,1)
print(dir_y)
# project.load_subproject()
# print(project.list_subprojects())