import cst
import cst.results
import cst.interface
import numpy as np

def cst_simulation(sample,sample_num,para_name,para_dim,filename,temp_name):

    for i in range(sample_num):
        # 创建cst环境
        cst_project = cst.interface.DesignEnvironment()
        # 打开目标工程
        project = cst_project.open_project(filename)
        # 另存目标工程
        project.save(temp_name)
        # 修改参数开始仿真
        modeler = project.modeler

        for j in range(para_dim):
            temp = 'StoreParameter(\"' + para_name[j] + '\",' + '\"%f\"'%sample[i][j] + ')'
            modeler.add_to_history('StoreParameter',temp)

        modeler.run_solver()
        project.save(temp_name)
        project.close()
        cst_project.close()

        project = cst.results.ProjectFile(temp_name)
        s11 = project.get_3d().get_result_item("1D Results\\S-Parameters\\S1,1")
        dir = project.get_3d().get_result_item("Tables\\1D Results\\Directivity,Phi=0.0")

        S11_temp = abs(np.array([s11.get_ydata()])).T
        dir_temp = np.array([dir.get_ydata()]).T

        if i == 0:
            S11 = S11_temp.reshape(-1,1)
            Dir = dir_temp.reshape(-1,1)
            frequency = s11.get_xdata()
            angle = dir.get_xdata()
        else:
            S11 = np.hstack((S11,S11_temp))
            Dir = np.hstack((Dir,dir_temp))

    return S11,Dir,frequency,angle


