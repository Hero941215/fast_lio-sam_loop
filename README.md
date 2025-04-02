# fast_lio-sam_loop
**Fast-lio with loop closing function. isam2 is used for pose graph optimization. The loop and odometry are integrated into a unified file. Supports full operation on long sequence datasets, such as urbanNAV dataset ([FAST_LIO_SAM](https://github.com/kahowang/FAST_LIO_SAM) will crash in about 10 minutes).**

## 0. Features
In the config file, if keyframe_pub_en and loop_en are true, the loop-closing function is enabled. Then, when correct_fe_en is "true", it means that the front-end ikd-tree will reconstruct when a loop occurs, and "false" means it will not reconstruct. For "false" case, drawing on the LOAM concept, the transform from the map coordinate system to the odom coordinate system is maintained and used to correct the FAST-LIO pose to the map system, providing initial pose graph. Therefore, Fast-lio2's ikd tree does not require reconstruction, ensuring that the front-end can always run efficiently.

Sparse_raw_point_cloud_en is set to true, which allows for downsampling of keyframes stored globally and controlling the running memory to make the system run longer.

**Important Notes**: 
  - **A Bug is fixed (2025-04-02)**: we fixed bug in the reconstruct() function provided by [FAST_LIO_LC](https://github.com/yanliang-wang/FAST_LIO_LC), [FAST_LIO_SAM](https://github.com/kahowang/FAST_LIO_SAM). Now, both correct_fe_en == true or correct_fe_en == false,  the system can run stably. Howerver, when correct_fe_en == true, reconstruct a ikd-tree map usually need more time for odometry, and the odometry pose will jump. So, we recommend correct_fe_en == false.

![Run on Public UrbanNav Dataset](https://github.com/Hero941215/fast_lio-sam_loop/blob/main/2025-04-02%2013-31-24.jpg)

## 1. Prerequisites
### 1.0 **gcc** and **g++**

gcc and g++ 7.5.0 are tested OK. However, gcc and g++ 10.3.0 are not OK for the gtsam we use.

### 1.1 **Ubuntu** and **ROS**
**Ubuntu >= 18.04**

ROS    >= Melodic. [ROS Installation](http://wiki.ros.org/ROS/Installation)

### 1.2. **PCL && Eigen**
PCL    >= 1.8,   Follow [PCL Installation](http://www.pointclouds.org/downloads/linux.html).

Eigen  >= 3.3.3, Follow [Eigen Installation](http://eigen.tuxfamily.org/index.php?title=Main_Page).

### 1.3. **livox_ros_driver**
Follow [livox_ros_driver Installation](https://github.com/Livox-SDK/livox_ros_driver).

## 2. Build

Clone the repository and catkin_make:

```
    cd ~/$A_ROS_DIR$/src
    git clone https://github.com/Hero941215/fast_lio-sam_loop
    cd fast_lio-sam_loop
    git submodule update --init
    cd ../..
    catkin_make
    source devel/setup.bash
```
- Remember to source the livox_ros_driver before build (follow 1.3 **livox_ros_driver**)

## 3. Run

### 3.1. Download Dataset ([UrbanNAV](https://github.com/IPNL-POLYU/UrbanNavDataset)): 

rosbag play XXX.bag

### 3.2. Run SLAM system: 

roslaunch fast_lio_sam_loop mapping_velodyne.launch

## 4. Acknowledgments

Thanks for LOAM(J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time), [FAST-LIO2](https://github.com/hku-mars/FAST_LIO)，[FAST_LIO_SAM](https://github.com/kahowang/FAST_LIO_SAM), [FAST_LIO_LC](https://github.com/yanliang-wang/FAST_LIO_LC).
