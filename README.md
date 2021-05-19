# NBV-Simulation
   This is a nbv simulation system for comparative experiment. Including the method of our paper "An Improved Global Adaptive-multiresolution Next-best-view Method for Reconstruction of 3D Unknown Objects".
## Installion
   <br> For our nbv_simulation c++ code. These libraries need to be installed: opencv 4.4.0, PCL 1.9.1, Eigen 3.3.9, OctoMap 1.9.6.<br>
   For nbv_net, please follow https://github.com/irvingvasquez/nbv-net.
   We tested our codes on Windows 10. For other system, please check the file read/write or multithreading commands in the codes.
## Note
   Change "const static size_t maxSize = 100000;" to "const static size_t maxSize = 1000" in file OcTreeKey.h, so that the code will run faster.
