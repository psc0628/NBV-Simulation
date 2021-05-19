# NBV-Simulation
This is a nbv simulation system for comparative experiment. Including the method of our paper "An Improved Global Adaptive-multiresolution Next-best-view Method for Reconstruction of 3D Unknown Objects".
## Installion
For our nbv_simulation c++ code, these libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6.
For nbv_net, please follow https://github.com/irvingvasquez/nbv-net.
We tested our codes on Windows 10. For other system, please check the file read/write or multithreading commands in the codes.
## Note
Change "const static size_t maxSize = 100000;" to "const static size_t maxSize = 1000" in file OcTreeKey.h, so that the code will run faster.
## Usage
1. Sample your 3d object model from *.obj or *.ply to *.pcd, and there is an example in 3d_models. For the sampling method, please follow   https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp.
2. Change the directory and model name in file DefaultConfiguration.yaml.
3. Put the file DefaultConfiguration.yaml in the correct path.
4. Run compiled program of main.cpp.
5. For the method_of_IG: Ours is 0, OA is 1, UV is 2, RSE is 3, APORA is 4, Kr is 5, NBVNET is 6. If you want to run with NBV-Net, please run both compiled program of main.cpp and "python run_test.py model_name" in pytorch environment.
6. There is a parameter "show", by default is 1, which means that the middle cloud will be shown in a pcl window, close it to continue. If you don't want to show the middle cloud, change it to 0.
## Questions
Please contact 18210240033@fudan.edu.cn
