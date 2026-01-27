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
        s11 = project.get_3d().get_result_item('Tables\\0D Results\\S1,1_0D_GlobalyMax')
        ftbr = project.get_3d().get_result_item('Tables\\1D Results\\Front-to-back ratio over frequency')
        gain = project.get_3d().get_result_item('Tables\\0D Results\\Max Gain at selected Frequency')

        s11_temp = np.array([s11.get_data()])
        ftbr_temp = np.array([ftbr.get_ydata()])
        gain_temp = np.array([gain.get_data()])

        if i == 0:
            S11 = s11_temp.reshape(-1,1)
            FTBR = ftbr_temp.reshape(-1,1)
            Gain = gain_temp.reshape(-1,1)

        else:
            S11 = np.vstack((S11,s11_temp))
            FTBR = np.vstack((FTBR,ftbr_temp))
            Gain = np.vstack((Gain,gain_temp))

    return S11, FTBR, Gain


