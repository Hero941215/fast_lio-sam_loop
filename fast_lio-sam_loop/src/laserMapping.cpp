// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>

#include <Eigen/Core>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include <std_srvs/Empty.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "IMU_Processing.hpp"
#include "preprocess.h"
#include "utility.h"
#include <ikd-Tree/ikd_Tree.h>

#include "fast_lio_sam_loop/save_map.h"

#include "tic_toc.h"

// using namespace gtsam;

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;

// T1为雷达初始时间戳，s_plot为整个流程耗时，s_plot2特征点数量,s_plot3为kdtree增量时间，s_plot4为kdtree搜索耗时，s_plot5为kdtree删除点数量
//，s_plot6为kdtree删除耗时，s_plot7为kdtree初始大小，s_plot8为kdtree结束大小,s_plot9为平均消耗时间，s_plot10为添加点数量，s_plot11为点云预处理的总时间
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];

// 定义全局变量，用于记录时间,match_time为匹配时间，solve_time为求解时间，solve_const_H_time为求解H矩阵时间
double match_time = 0, solve_time = 0, solve_const_H_time = 0;

// kdtree_size_st为ikd-tree获得的节点数，kdtree_size_end为ikd-tree结束时的节点数，add_point_size为添加点的数量，kdtree_delete_counter为删除点的数量
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;

// runtime_pos_log运行时的log是否开启，pcd_save_en是否保存pcd文件，time_sync_en是否同步时间
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};          // 残差，点到面距离平方和
float DET_RANGE = 300.0f;                // 设置的当前雷达系中心到各个地图边缘的距离 
const float MOV_THRESHOLD = 1.5f;        // 设置的当前雷达系中心到各个地图边缘的权重
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;           // https://blog.csdn.net/weixin_43369786/article/details/129225369

string root_dir = ROOT_DIR;                  // 设置根目录
string map_file_path, lid_topic, imu_topic;  // 设置地图文件路径，雷达topic，imu topic


double res_mean_last = 0.05, total_residual = 0.0;              // 残差平均值，残差总和
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;     // 雷达时间戳，imu时间戳

double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;  // imu预设参数

double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;   // 0.5 具体作用再看
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;  // 设置立方体长度，视野一半的角度，视野总角度，总距离，雷达结束时间，雷达初始时间
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;         // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);    // 外参
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;     // 激光雷达数据时间戳缓存队列，存储的激光雷达一帧的初始时刻
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());   // 去畸变的特征，lidar系
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());   // 畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());  // 地图系单帧点云
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));  
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterICP;     // loop, icp matching, down sampling
pcl::VoxelGrid<PointType> downSizeFilterSurf;    // fast-lio, measure point cloud, down sampling
// pcl::VoxelGrid<PointType> downSizeFilterMap;      

KD_TREE<PointType> ikdtree;   // ikd-tree地图

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);    
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;                                  // 当前的欧拉角
V3D position_last(Zero3d);                      // 上一帧的位置
V3D Lidar_T_wrt_IMU(Zero3d);                    // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;                            // 点云和激光雷达数据
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;  
state_ikfom state_point;                          // ! 估计器的当前状态
vect3 pos_lid;                                    // world系下lidar坐标

nav_msgs::Path path;                              // ! odom path
nav_msgs::Odometry odomAftMapped;                 // imu里程计
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

// ! add-mtc-20250116, for bundle_adjustment_test, add keyframe detection
bool   scan_lidar_pub_en = false, keyframe_pub_en = false;          // 是否发布雷达坐标系下的激光点云，是否发布关键帧
bool   first_keyframe = true;                                       // 第一帧强制插入
double last_timestamp_keyframe = 0;                                 // 上一次关键帧插入的时刻
double keyframe_timestamp_th = 0.25;                                // 关键帧插入的时间间隔
bool timestamp_save_en = false;                                     // 存储激光雷达时间戳

state_ikfom last_odom_kf_state;                                     // 上一里程计坐标系下的关键帧状态量
double keyframe_trans_th = 0, keyframe_rot_th = 0;                  // 关键帧位姿增量检测
nav_msgs::Odometry lidarOdom;                                       // lidar里程计
// ! ---------------------------- add-mtc ---------------------------------

// ! add-mtc-20250331, for loop closing with odom change mode

// 点云存储服务 
string save_directory;                  // ? need init
float globalMapServerLeafSize = 0.4;    // ? need init，0.4

// frame for message and tf
string odometryFrame = "odom";          // ? need init
string MapFrame = "map";                // ? need init 

// gtsam
gtsam::NonlinearFactorGraph gtSAMgraph; 
gtsam::Values initialEstimate;
gtsam::Values optimizedEstimate;
gtsam::ISAM2 *isam;                     // ? need init
gtsam::Values isamCurrentEstimate;
Eigen::MatrixXd correctPoseCovariance;  // 从isam中拿到优化后的最新关键帧的协方差矩阵。

// 关键帧对应的点云
// 注意，如果启动显示界面，则需要是两个容器。否则显示线程可能会导致里程计线程丢帧。存储降采样点云是为了更好的控制运行内存
bool SparseRawPointCloudFlag = true;   // ? need init, default: true
vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;     // 存的是降采样点云
vector<pcl::PointCloud<PointType>::Ptr> laserCloudRawKeyFrames; // 默认存降采样，可以改成所有点

// 里程计线程中的里程计位姿
float transformTobeMapped[6];   //  当前帧的位姿(odometryFrame系下)

// 关键帧位姿
pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

// Loop closure
bool loopClosureEnableFlag;     // ? need init，true，开了关键帧检测进而判断是否使用loop
float loopClosureFrequency;     // ? need init，1.0 
float historyKeyframeSearchRadius;   // ? need init
float historyKeyframeSearchTimeDiff; // ? need init
int historyKeyframeSearchNum;        // ? need init
float historyKeyframeFitnessScore;   // ? need init
float surroundingKeyframeSearchRadius;      // ? need init，重建ikd-tree的关键帧搜索范围，50m
float surroundingKeyframeDensity;           // ? need init，重建ikd-tree的关键帧密度，1.0m
float mappingSurfLeafSize;                  // ? need init，重建ikd-tree的点云密度，0.4m
pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<PointType>());

// global map visualization radius
float globalMapVisualizationSearchRadius;  // ? need init
float globalMapVisualizationPoseDensity;   // ? need init
float globalMapVisualizationLeafSize;      // ? need init

bool aLoopIsClosed = false;
map<int, int> loopIndexContainer;           // from new to old
vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

// fe-pose correct by isam
bool correct_fe_flag;          // 默认开启，但是如果需要进行路径规划等任务，需要关闭。
gtsam::Pose3 correct_Tmo;      // correct_fe_flag = false时使用，需要初始化
nav_msgs::Path globalPath;     // path in MapFrame

// Ros Publisher
ros::Publisher pubHistoryKeyFrames;
ros::Publisher pubIcpKeyFrames;
ros::Publisher pubLoopConstraintEdge;
ros::Publisher pubLaserCloudSurround;
ros::ServiceServer srvSaveMap;

std::mutex mtx;

// 回环线程
void loopClosureThread();
bool detectLoopClosureDistance(int *latestID, int *closestID);
bool detectLoopClosureMultiCond(int *latestID, int *closestID); // plus（检测静止启动、避免临近关键帧（lio-sam-6-axis），使用局部窗口里面的最老关键帧(slict)）
void performLoopClosure();
void visualizeLoopClosure();
void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum);

// 里程计线程
void getCurPose(state_ikfom cur_state);     // fast_lio_sam
void saveKeyFramesAndFactor();
void saveKeyFramesAndFactor_wtUpdateIKF();

void addOdomFactor();
void addOdomFactor_Fastlio();

void addLoopFactor();

void correctPoses();  // 回环成功的话，更新轨迹。并且if correct_fe_flag == true, 重建Ikd-tree地图
void correctPoses_wtRebuildIKD();

void updatePath(const PointTypePose &pose_in);

void recontructIKdTree();    // ikd-tree地图重建，使用lio-sam中的局部地图搜索方案	
void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract);

// 发布全局地图
void visualizeGlobalMapThread();
void publishGlobalMap();

// 存储地图
bool saveMapService(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);

// !  ------------------------------------- add-mtc ---------------------------------------------

shared_ptr<Preprocess> p_pre(new Preprocess());   // 点云预处理器
shared_ptr<ImuProcess> p_imu(new ImuProcess());   // imu预处理器

// 接受到中断信号会触发
void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

// 存储状态信息
inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

// 没有用
void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);   //返回被剔除的点
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

// admin.guyuehome.com/38803
// 地图点被组织成一个ikd-Tree，该树以里程计速率合并新的点云扫描而动态增长。
// 为了防止地图的大小不受约束，ikid - tree中只保留LiDAR当前位置周围长为L的大型局部区域中的地图点
BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;         // global系下lidar位置
    // 初始化局部地图包围盒角点，以为w系下lidar位置为中心,得到长宽高200*200*200的局部地图
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);   // 计算到边界的距离
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){        // 可能会有三个cube需要被移除
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    // mtx_buffer.lock();  // 同一时刻，只做点云或者imu的数据处理
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    std::cout << "cur pc size: " << ptr->points.size() << std::endl;

    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    // mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    // mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    // mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();  // 帧初时间戳
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time; // 帧末时间戳

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)  // imu高频率，因此，在进行数据对齐时，理论上应该已经有大于激光雷达的imu数据了
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec(); // 拿出lidar_beg_time到lidar_end_time之间的所有IMU数据
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

// 发布里程计坐标系下的点云
PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en) // 默认发布
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body); // 默认发布降采样点
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = odometryFrame;
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

// 发布imu坐标系下的局部点云
void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "imu_link";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// 发布lidar坐标系下的局部点云
void publish_frame_lidar(const ros::Publisher & pubLaserCloudFull_lidar)
{
    sensor_msgs::PointCloud2 laserCloudmsg;
    if(dense_pub_en)
    {
        pcl::toROSMsg(*feats_undistort, laserCloudmsg);
        // std::cout << "keyframe pc size: " << feats_undistort->points.size() << std::endl;
    } 
    else
    {
        pcl::toROSMsg(*feats_down_body, laserCloudmsg);
        // std::cout << "keyframe pc size: " << feats_down_body->points.size() << std::endl;
    }
        
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "laser_link";
    pubLaserCloudFull_lidar.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

// 默认不使用
void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = odometryFrame;
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

// 默认不使用
void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = odometryFrame;
    pubLaserCloudMap.publish(laserCloudMap);
}

// 设置输出的t,q，在publish_odometry，publish_path调用
template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_lidar_odometry(const ros::Publisher & pubLidarOdom)
{
    Eigen::Vector3d imu_toi(state_point.pos(0), state_point.pos(1), state_point.pos(2)); 
    Eigen::Quaterniond imu_qoi(geoQuat.w, geoQuat.x, geoQuat.y, geoQuat.z); 

    Eigen::Isometry3d Toi; Toi.setIdentity();
    Toi.rotate(imu_qoi);
    Toi.pretranslate(imu_toi);
    
    Eigen::Quaterniond qil = state_point.offset_R_L_I;
    Eigen::Vector3d til = state_point.offset_T_L_I;
    Eigen::Isometry3d Til; Til.setIdentity();
    Til.rotate(qil);
    Til.pretranslate(til);

    Eigen::Isometry3d Tol = Toi*Til;
    Eigen::Quaterniond qol; qol = Tol.linear();
    Eigen::Vector3d tol = Tol.translation();

    lidarOdom.header.frame_id = odometryFrame;
    lidarOdom.child_frame_id = "laser_link";
    lidarOdom.header.stamp = ros::Time().fromSec(lidar_end_time);
    lidarOdom.pose.pose.position.x =  tol.x();
    lidarOdom.pose.pose.position.y = tol.y();
    lidarOdom.pose.pose.position.z = tol.z();
    lidarOdom.pose.pose.orientation.x = qol.x();
    lidarOdom.pose.pose.orientation.y = qol.y();
    lidarOdom.pose.pose.orientation.z = qol.z();
    lidarOdom.pose.pose.orientation.w = qol.w();
    pubLidarOdom.publish(lidarOdom);

    if(0)
    {
        ofstream f;
        std::string res_dir = string(ROOT_DIR) + "result/fe_lio_pose.txt";
        f.open(res_dir, ios::app);
        f << fixed;
        f << setprecision(6) << lidar_end_time << setprecision(7) << " " << tol.x() << " " 
        << tol.y() << " " << tol.z() << " " << qol.x() << " " << qol.y() << " " << qol.z() << " " << qol.w() << std::endl;
        f.close();
    }

    // ! add-mtc-20250227, 发布tf变换 lidar_to_odom 的tf变换
    static tf::TransformBroadcaster br2;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(lidarOdom.pose.pose.position.x, \
                                    lidarOdom.pose.pose.position.y, \
                                    lidarOdom.pose.pose.position.z));
    q.setW(lidarOdom.pose.pose.orientation.w);
    q.setX(lidarOdom.pose.pose.orientation.x);
    q.setY(lidarOdom.pose.pose.orientation.y);
    q.setZ(lidarOdom.pose.pose.orientation.z);
    transform.setRotation( q );
    br2.sendTransform( tf::StampedTransform( transform, lidarOdom.header.stamp, odometryFrame, "laser_link" ) );
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = odometryFrame;
    odomAftMapped.child_frame_id = "imu_link";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    // 设置协方差 P里面先是旋转后是位置 这个POSE里面先是位置后是旋转 所以对应的协方差要对调一下
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    std::cout << "imu pose.position: " << odomAftMapped.pose.pose.position.x << " " << odomAftMapped.pose.pose.position.y << " " << 
                           odomAftMapped.pose.pose.position.z << std::endl;

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, odometryFrame, "imu_link" ) );

    if(0)
    {
        ofstream f;
        std::string res_dir = string(ROOT_DIR) + "result/fe_lio_pose.txt";
        f.open(res_dir, ios::app);
        f << fixed;
        f << setprecision(6) << lidar_end_time << setprecision(7) << " " << odomAftMapped.pose.pose.position.x << " " 
        << odomAftMapped.pose.pose.position.y << " " << odomAftMapped.pose.pose.position.z << " " <<
                        odomAftMapped.pose.pose.orientation.x << " " << odomAftMapped.pose.pose.orientation.y << " "
                         << odomAftMapped.pose.pose.orientation.z << " " << odomAftMapped.pose.pose.orientation.w << std::endl;
        f.close();

    }
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = MapFrame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

// 通过函数h_share_model（h_dyn_share_in）同时计算残差（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    // std::cout << "run iekf, cur feats_down_size: " << feats_down_size << std::endl;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)  // 如果收敛了
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // 如果最近邻的点数小于NUM_MATCH_POINTS或者最近邻的点到特征点的距离大于5m，则认为该点不是有效点
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue; // 周围点的数量充足

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);               //将残差存储至res_last
            }
        }
    }
    
    effct_feat_num = 0;  // 统计有效数据关联的数量

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;  // 平均误差
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;  // imu坐标系
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

// 这里直接使用Tmi的增量进行判断即可
bool need_new_keyframe()
{
    if(first_keyframe)
    {	
        first_keyframe = false;
        return true;
    }
    else
    {
        // 计算时间差，并判断
        double time_diff = std::fabs(lidar_end_time-last_timestamp_keyframe);
        if(time_diff<keyframe_timestamp_th)
            return false;

        // 计算位姿增量差，并判断
        Eigen::Affine3f last_Toi = stateIkfomToAffine3f(last_odom_kf_state);
        Eigen::Affine3f cur_Toi = stateIkfomToAffine3f(state_point);

        Eigen::Affine3f transBetween = last_Toi.inverse() * cur_Toi;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        std::cout << "delta_euler: " << roll << " " << pitch << " " << yaw << std::endl;
        std::cout << "delta_t: " << x << " " << y << " " << z << std::endl << std::endl;

        if(sqrt(x*x + y*y + z*z) > keyframe_trans_th || fabs(roll) > keyframe_rot_th ||
                            fabs(pitch) > keyframe_rot_th || fabs(yaw) > keyframe_rot_th)
            return true;
        else
            return false;
    }
}

void keyframe_detection(const ros::Publisher & pubLidarOdom, const ros::Publisher & pubLaserCloudFull_lidar)
{
    if(keyframe_pub_en)
    {
        if(need_new_keyframe())
        {
            publish_lidar_odometry(pubLidarOdom);
            publish_frame_lidar(pubLaserCloudFull_lidar);
            ROS_INFO("\033[1;32m----> Create new keyframe by LIO.\033[0m");

            // ! add-mtc-20250329, 添加位姿图后端（ref: lio-sam）
            std::lock_guard<std::mutex> lock(mtx);

            if(loopClosureEnableFlag)
            {
                if(correct_fe_flag)
                {
                    saveKeyFramesAndFactor();
                    correctPoses();
                }
                else
                {
                    saveKeyFramesAndFactor_wtUpdateIKF();
                    correctPoses_wtRebuildIKD();
                }
            }

            // ! 更新时间戳和状态量, 不能放在前面
            last_timestamp_keyframe = lidar_end_time;
            last_odom_kf_state = state_point;
        }
    }
    else
    {
        if (scan_pub_en && scan_lidar_pub_en) 
        {
            publish_frame_lidar(pubLaserCloudFull_lidar);
        }   
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    
    /***************************** fast-lio *************************/
    nh.param<bool>("publish/path_en", path_en, true);
    nh.param<bool>("publish/scan_publish_en", scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en", dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en, true);
    nh.param<bool>("publish/scan_lidarframe_pub_en", scan_lidar_pub_en, true);
    nh.param<int>("max_iteration", NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path", map_file_path,"");
    nh.param<string>("common/lid_topic", lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner", filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf", filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map", filter_size_map_min,0.5);
    nh.param<double>("cube_side_length", cube_len,500);
    nh.param<float>("mapping/det_range", DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree", fov_deg,180);
    nh.param<double>("mapping/gyr_cov", gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov", acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov", b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov", b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<bool>("time_save/timestamp_save_en", timestamp_save_en, false);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

    // ***************************** frame coordinate *****************************
    nh.param<string>("frame/odometry_frame", odometryFrame, "odom");
    nh.param<string>("frame/map_frame", MapFrame, "map");

    // ***************************** keyframe detection *****************************
    nh.param<bool>("publish/scan_lidarframe_pub_en", scan_lidar_pub_en, true);
	nh.param<bool>("publish/keyframe_pub_en", keyframe_pub_en, true);
	nh.param<double>("keyframe/interval_th", keyframe_timestamp_th, 0.25);
	nh.param<double>("keyframe/trans_th", keyframe_trans_th, 0.1);
	nh.param<double>("keyframe/rot_rad_th", keyframe_rot_th, 0.0872);

    // ***************************** loop closing *****************************
    nh.param<bool>("odometry_mode/correct_fe_en", correct_fe_flag, false);
    nh.param<bool>("loop/loop_en", loopClosureEnableFlag, false);
    nh.param<float>("loop/loop_frequency", loopClosureFrequency, 1.0);
    nh.param<bool>("loop/sparse_raw_point_cloud_en", SparseRawPointCloudFlag, true);
    nh.param<float>("loop/history_kf_search_radius", historyKeyframeSearchRadius, 25.0);
    nh.param<float>("loop/history_kf_search_time_diff", historyKeyframeSearchTimeDiff, 25.0);    
    nh.param<int>("loop/history_kf_search_num", historyKeyframeSearchNum, 25);   
    nh.param<float>("loop/history_kf_fitness_score", historyKeyframeFitnessScore, 0.4);    
    nh.param<float>("loop/surrounding_kf_search_radius", surroundingKeyframeSearchRadius, 30.0);
    nh.param<float>("loop/surrounding_kf_density", surroundingKeyframeDensity, 2.0);
    nh.param<float>("loop/mapping_surf_leaf_size", mappingSurfLeafSize, 0.4);

    if(keyframe_pub_en && loopClosureEnableFlag)
    {
        ROS_INFO("\033[1;32m----> LIO system run with LOOP Function.\033[0m");
        if(correct_fe_flag)
            ROS_INFO("\033[1;34m----> LOOP Function with rebuild ikd-tree.\033[0m");
        else
            ROS_INFO("\033[1;34m----> LOOP Function without rebuild ikd-tree.\033[0m");
    }
    else
        ROS_INFO("\033[1;32m----> LIO system run without LOOP Function.\033[0m");

    correct_Tmo = gtsam::Pose3();

    // ***************************** visualization *****************************
    nh.param<float>("visualization/global_map_visualization_search_radius", globalMapVisualizationSearchRadius, 1000.0);
    nh.param<float>("visualization/global_map_visualization_pose_density", globalMapVisualizationPoseDensity, 10.0);
    nh.param<float>("visualization/global_map_visualization_leaf_size", globalMapVisualizationLeafSize, 0.5);

    // ******************************* service *******************************
    nh.param<string>("service/save_directory", save_directory, "/home/mtcjyb/Documents/");
    nh.param<float>("service/global_map_server_leaf_size", globalMapServerLeafSize, 0.4);
    
    // ***************************** optimization *****************************
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    // ***************************** allocate Memory **************************
    cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
    copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
    copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

    // ***************************** init odometry pose **************************
    for (int i = 0; i < 6; ++i)
    {
        transformTobeMapped[i] = 0;
    }


    path.header.stamp    = ros::Time::now();
    path.header.frame_id = MapFrame;

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0; // 雷达总帧数 
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    //这里没用到
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    // std::cout << "filter_size_surf_min: " << filter_size_surf_min << std::endl;

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);

    // 将函数地址传入kf对象中，用于接收特定于系统的模型及其差异
    // 通过一个函数（h_dyn_share_in）同时计算测量（z）、估计测量（h）、偏微分矩阵（h_x，h_v）和噪声协方差（R）
    // get_f: fast-lio2 eq(1) 动力学模型
    // df_dx, df_dw: fast-lio2 eq(7) 误差状态传递
    // h_share_model: 测量模型，用于计算残差, h_share_model需要进行数据关联
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    /*** ROS publisher initialization ***/
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    ros::Publisher pubLaserCloudFull_lidar = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_lidar", 100000);
	ros::Publisher pubLidarOdom = nh.advertise<nav_msgs::Odometry> 
            ("/LidarOdometry", 100000);

    pubHistoryKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("loop/icp_loop_closure_history_cloud", 1);
    pubIcpKeyFrames = nh.advertise<sensor_msgs::PointCloud2>("loop/icp_loop_closure_corrected_cloud", 1);
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop/loop_closure_constraints", 1);
    pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("visualization/map_global", 1);

    srvSaveMap = nh.advertiseService("service/save_map", &saveMapService);

    std::thread loopthread(&loopClosureThread);
    std::thread visualizeMapThread(&visualizeGlobalMapThread);

//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;  // 如果有中断产生，则结束主循环
        ros::spinOnce();      // ROS消息回调处理函数，放在ROS的主循环中
        if(sync_packages(Measures))   // ROS消息回调处理函数，放在ROS的主循环中
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;       // 这一步操作，可以让imu中的数据于lidar内部对齐，因此，第一帧实际上是被扔掉了。
                continue;
            }

            TicToc t_odom;

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort); // 点云去畸变到帧尾
            state_point = kf.get_x();  // 这里拿到的是预测的Tmi
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;  // 雷达坐标系下的机器人位置

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment(); // 维护局部地图

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);        // 0.5， 控制参与前端的点云数量

            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/ 
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            getCurPose(state_point);                    // ! 更新transformTobeMapped

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            std::cout << "frame_id: " << frame_num << std::endl;
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();    // 分为需要和不需要降采样的点
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);

            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /******* Keyframe detection (And save imu key pose for pose_graph) *******/
            // ! ******* rebuild ikd-tree based on your need, use config to change the mode !!!! *******/
            keyframe_detection(pubLidarOdom, pubLaserCloudFull_lidar);

            /******* 激光雷达时间戳存储 *******/
            if(timestamp_save_en)
            {
                ofstream f;
				std::string res_dir = string(ROOT_DIR) + "result/timestamps.txt";
				f.open(res_dir, ios::app);
				f << fixed;
				f << setprecision(6) << lidar_end_time << std::endl;
				f.close();
            }

            /*** Debug variables ***/
            frame_num ++;
            if (runtime_pos_log)
            {
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }

            std::cout << "odometry cost time: " << t_odom.toc() << " ms" << std::endl << std::endl;
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    loopthread.join(); //  分离线程
    visualizeMapThread.join();

    return 0;
}

void loopClosureThread()
{
    if (loopClosureEnableFlag == false)
        return;

    ros::Rate rate(loopClosureFrequency);
    while (ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
        // std::cout << "run loop closure!" << std::endl;
        performLoopClosure();
        visualizeLoopClosure();
    }
}

bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
    int loopKeyPre = -1;

    // check loop constraint added before
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
      return false;

    // find the closest history key frame
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
      int id = pointSearchIndLoop[i];
      if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
      {
        loopKeyPre = id;
        break;
      }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
      return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

void performLoopClosure()
{
    if (cloudKeyPoses3D->points.empty() == true)
        return;
    
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳

    mtx.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx.unlock();

    // find keys
    int loopKeyCur;
    int loopKeyPre;

    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;
    
    // extract cloud
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    mtx.lock();
    loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
    loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
    mtx.unlock();

    if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
        return;

    if (pubHistoryKeyFrames.getNumSubscribers() != 0)
        publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, MapFrame);

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;
    else
        ROS_INFO("\033[1;32m----> Loop ICP Check Pass!!.\033[0m");

    // publish corrected cloud
    if (pubIcpKeyFrames.getNumSubscribers() != 0)
    {
      pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
      pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
      publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, MapFrame);
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx.unlock();

    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;
}

bool detectLoopClosureMultiCond(int *latestID, int *closestID)
{

    return false;
}

void visualizeLoopClosure()
{
    if (loopIndexContainer.empty())
      return;
    
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳

    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = odometryFrame;
    markerNode.header.stamp = timeLaserInfoStamp;
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3;
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = odometryFrame;
    markerEdge.header.stamp = timeLaserInfoStamp;
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
      int key_cur = it->first;
      int key_pre = it->second;
      geometry_msgs::Point p;
      p.x = copy_cloudKeyPoses6D->points[key_cur].x;
      p.y = copy_cloudKeyPoses6D->points[key_cur].y;
      p.z = copy_cloudKeyPoses6D->points[key_cur].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
      p.x = copy_cloudKeyPoses6D->points[key_pre].x;
      p.y = copy_cloudKeyPoses6D->points[key_pre].y;
      p.z = copy_cloudKeyPoses6D->points[key_pre].z;
      markerNode.points.push_back(p);
      markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
{
    // extract near keyframes
    nearKeyframes->clear();
    int cloudSize = copy_cloudKeyPoses6D->size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
      int keyNear = key + i;
      if (keyNear < 0 || keyNear >= cloudSize )
        continue;
        
      *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
      return;

    // downsample near keyframes
    pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    downSizeFilterICP.setInputCloud(nearKeyframes);
    downSizeFilterICP.filter(*cloud_temp);
    *nearKeyframes = *cloud_temp;
}

// 里程计线程
void getCurPose(state_ikfom cur_state)     // fast_lio_sam
{
    //  欧拉角是没有群的性质，所以从SO3还是一般的rotation matrix 转换过来的结果一样
    Eigen::Vector3d eulerAngle = cur_state.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
    
    transformTobeMapped[0] = eulerAngle(2);             //  roll  使用 eulerAngles(2,1,0) 方法时，顺序是 ypr
    transformTobeMapped[1] = eulerAngle(1);             //  pitch
    transformTobeMapped[2] = eulerAngle(0);             //  yaw
    transformTobeMapped[3] = cur_state.pos(0);          //  x
    transformTobeMapped[4] = cur_state.pos(1);          //   y
    transformTobeMapped[5] = cur_state.pos(2);          // z
}

void saveKeyFramesAndFactor()
{
    // 激光里程计因子(from fast-lio),  输入的是frame_relative pose  帧间位姿(body 系下)
    addOdomFactor();

    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();

    gtsam::Pose3 latestRawEstimate = stateIkfomTogtsamPose3(state_point); // ! 记录优化前的位姿

    // update iSAM
    // std::cout << "gtSAMgraph.size(): " << gtSAMgraph.size() << " initialEstimate.size(): " << initialEstimate.size() << std::endl;
    // gtSAMgraph.print();
    // initialEstimate.print();
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();     // 这一步很重要，如果后端发生了回环优化，则这里可以把前端修过来

    if (aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();  
        isam->update();
        isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);
    correct_Tmo = latestEstimate * latestRawEstimate.inverse();        // ! 更新矫正矩阵
    
    static tf::TransformBroadcaster br3;
    tf::Transform                   transform;
    tf::Quaternion                  qmo;
    Eigen::Vector3d Correct_tmo = correct_Tmo.translation();
    Eigen::Quaterniond Correct_qmo = correct_Tmo.rotation().toQuaternion();
    transform.setOrigin(tf::Vector3(Correct_tmo(0), \
                                    Correct_tmo(1), \
                                    Correct_tmo(2)));
    qmo.setW(Correct_qmo.w());
    qmo.setX(Correct_qmo.x());
    qmo.setY(Correct_qmo.y());
    qmo.setZ(Correct_qmo.z());
    transform.setRotation( qmo );
    br3.sendTransform( tf::StampedTransform( transform, lidarOdom.header.stamp, MapFrame, odometryFrame ) );

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);

    // ESKF状态和方差  更新
    state_ikfom state_updated = kf.get_x(); //  获取cur_pose (还没修正)
    Eigen::Vector3d pos(latestEstimate.translation().x(), latestEstimate.translation().y(), latestEstimate.translation().z());
    Eigen::Quaterniond q = EulerToQuat(latestEstimate.rotation().roll(), latestEstimate.rotation().pitch(), latestEstimate.rotation().yaw());

    //  更新状态量
    state_updated.pos = pos;
    state_updated.rot =  q;
    state_point = state_updated; // 对state_point进行更新，state_point可视化用到
    kf.change_x(state_updated);  // 对cur_pose 进行isam2优化后的修正

    // 更新协方差
    auto P_updated = kf.get_P();  // ikf中前6个维度是先旋转后平移
    correctPoseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);  // 6*6 先旋转后平移
    std::cout << "P_update: ";
    for (int k = 0; k < 6; k++)
    {
        P_updated(k, 0) = correctPoseCovariance(k, 0);
        P_updated(k, 1) = correctPoseCovariance(k, 1);
        P_updated(k, 2) = correctPoseCovariance(k, 2);
        P_updated(k, 3) = correctPoseCovariance(k, 3);
        P_updated(k, 4) = correctPoseCovariance(k, 4);
        P_updated(k, 5) = correctPoseCovariance(k, 5);
        std::cout << correctPoseCovariance(k, k) << " ";
    }
    std::cout << std::endl<< std::endl;

    // 存储降采样点云和原始（降采样）点云
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thislaserCloudRawKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*feats_down_body, *thisSurfKeyFrame);
    if(SparseRawPointCloudFlag)
        pcl::copyPointCloud(*feats_down_body, *thislaserCloudRawKeyFrame);
    else
        pcl::copyPointCloud(*feats_undistort, *thislaserCloudRawKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    laserCloudRawKeyFrames.push_back(thislaserCloudRawKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
}

void saveKeyFramesAndFactor_wtUpdateIKF()
{
    // 激光里程计因子(from fast-lio),  输入的是frame_relative pose  帧间位姿(body 系下)
    addOdomFactor_Fastlio();

    // 闭环因子 (rs-loop-detect)  基于欧氏距离的检测
    addLoopFactor();

    gtsam::Pose3 latestRawEstimate = stateIkfomTogtsamPose3(state_point); // ! 记录优化前的位姿

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();     // 这一步很重要，如果后端发生了回环优化，则这里可以把前端修过来

    if (aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();  
        isam->update();
        isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    // save key poses
    PointType thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);
    correct_Tmo = latestEstimate * latestRawEstimate.inverse();        // ! 更新矫正矩阵

    static tf::TransformBroadcaster br3;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    Eigen::Vector3d Correct_tmo = correct_Tmo.translation();
    Eigen::Quaterniond Correct_qmo = correct_Tmo.rotation().toQuaternion();
    transform.setOrigin(tf::Vector3(Correct_tmo(0), \
                                    Correct_tmo(1), \
                                    Correct_tmo(2)));
    q.setW(Correct_qmo.w());
    q.setX(Correct_qmo.x());
    q.setY(Correct_qmo.y());
    q.setZ(Correct_qmo.z());
    transform.setRotation( q );
    br3.sendTransform( tf::StampedTransform( transform, lidarOdom.header.stamp, MapFrame, odometryFrame ) );

    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
    thisPose6D.roll = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);

    correctPoseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);  // 6*6 先旋转后平移

    // 存储降采样点云和原始（降采样）点云
    pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr thislaserCloudRawKeyFrame(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*feats_down_body, *thisSurfKeyFrame);
    if(SparseRawPointCloudFlag)
        pcl::copyPointCloud(*feats_down_body, *thislaserCloudRawKeyFrame);
    else
        pcl::copyPointCloud(*feats_undistort, *thislaserCloudRawKeyFrame);
    surfCloudKeyFrames.push_back(thisSurfKeyFrame);
    laserCloudRawKeyFrames.push_back(thislaserCloudRawKeyFrame);

    // save path for visualization
    updatePath(thisPose6D);
}

void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        // std::cout << "PriorFactor:" << trans2gtsamPose(transformTobeMapped) << std::endl;
        // std::cout << "add-prior factor!" << std::endl;
    }else{
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

void addOdomFactor_Fastlio()
{
    if (cloudKeyPoses3D->points.empty())
    {
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }else{
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = stateIkfomTogtsamPose3(last_odom_kf_state);  // diff from above function
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);        // 这里是里程计估计的位姿
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        // 进行预测矫正, diff from above function
        poseTo = correct_Tmo * poseTo;
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);             // diff from above function
    }
}

void addLoopFactor()
{
    if (loopIndexQueue.empty())
      return;

    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
      int indexFrom = loopIndexQueue[i].first;
      int indexTo = loopIndexQueue[i].second;
      gtsam::Pose3 poseBetween = loopPoseQueue[i];
      gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
      gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

void correctPoses()  // 回环成功的话，更新轨迹。并且if correct_fe_flag == true, 重建Ikd-tree地图
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // 清空里程计轨迹
        globalPath.poses.clear();
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D->points[i]);
        }

        // 清空局部map， reconstruct  ikdtree submap
        recontructIKdTree();
    
        ROS_INFO("\033[1;32m----> ISAM2 Big Update after loop.\033[0m");
        aLoopIsClosed = false;
    }
}

void correctPoses_wtRebuildIKD()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // 清空里程计轨迹
        globalPath.poses.clear();
        // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            // 更新里程计轨迹
            updatePath(cloudKeyPoses6D->points[i]);
        }

        ROS_INFO("ISAM2 Update");
        aLoopIsClosed = false;
    }
}

void recontructIKdTree()    // ikd-tree地图重建，使用lio-sam中的局部地图搜索方案	
{
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses(new pcl::KdTreeFLANN<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    // extract all the nearby key poses and downsample them
    kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
    kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
    for (int i = 0; i < (int)pointSearchInd.size(); ++i)
    {
      int id = pointSearchInd[i];
      surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
    }

    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses;
    downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity);
    downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
    downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
    for(auto& pt : surroundingKeyPosesDS->points)
    {
      kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
      pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
    }

    extractCloud(surroundingKeyPosesDS);
}

void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
{
    pcl::PointCloud<PointType>::Ptr subMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr subMapKeyFramesDS(new pcl::PointCloud<PointType>());
        
    for (int i = 0; i < (int)cloudToExtract->size(); ++i)
    {
        if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
            continue;

        int thisKeyInd = (int)cloudToExtract->points[i].intensity;
        pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd]);
        *subMapKeyFrames += laserCloudSurfTemp;
    }

    // 降采样
    pcl::VoxelGrid<PointType> downSizeFilterSubMapKeyFrames;      // for global map visualization
    downSizeFilterSubMapKeyFrames.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
    downSizeFilterSubMapKeyFrames.setInputCloud(subMapKeyFrames);
    downSizeFilterSubMapKeyFrames.filter(*subMapKeyFramesDS);

    std::cout << "subMapKeyFramesDS->points.size(): " << subMapKeyFramesDS->points.size() << std::endl;

    ROS_INFO("\033[1;32m----> Reconstructed ikd-tree Map.\033[0m");
    ikdtree.reconstruct(subMapKeyFramesDS->points);
    int featsFromMapNum = ikdtree.validnum();
    int kdtree_size_st = ikdtree.size();
    std::cout << "featsFromMapNum  =  "   << featsFromMapNum   <<  "\t" << " kdtree_size_st   =  "  <<  kdtree_size_st  << std::endl;
}

// 发布全局地图
void visualizeGlobalMapThread()
{
    ros::Rate rate(0.2);
    while (ros::ok()){
      rate.sleep();
      publishGlobalMap();
    }
}

void publishGlobalMap()
{   
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time);

    if (pubLaserCloudSurround.getNumSubscribers() == 0)
      return;

    if (cloudKeyPoses3D->points.empty() == true)
      return;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
    pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

    // kd-tree to find near key frames to visualize
    std::vector<int> pointSearchIndGlobalMap;
    std::vector<float> pointSearchSqDisGlobalMap;
    // search near key frames to visualize
    mtx.lock();
    kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
    kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
    mtx.unlock();

    for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
      globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
    // downsample near selected key frames
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
    downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
    downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
    for(auto& pt : globalMapKeyPosesDS->points)
    {
      kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
      pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
    }

    // extract visualized and downsampled key frames
    for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
        if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
        continue;
        int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
        *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
    }
    // downsample visualized points
    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
    publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, MapFrame);
}

// 存储地图
bool saveMapService(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    if (cloudKeyPoses6D->size() < 1) {
			ROS_INFO("NO ENCOUGH POSE!");
			return false;
		}
		pcl::PointCloud<PointType>::Ptr globalRawCloud(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalRawCloudDS(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());

        mtx.lock();
		for (int i = 0; i < (int) cloudKeyPoses3D->size(); i++) {
			*globalRawCloud += *transformPointCloud(laserCloudRawKeyFrames[i], &cloudKeyPoses6D->points[i]);
			cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size()<< " ...";
		}
        pcl::VoxelGrid<PointType> downSizeFilterSurf2;
        downSizeFilterSurf2.setLeafSize(globalMapServerLeafSize, globalMapServerLeafSize, globalMapServerLeafSize);
		downSizeFilterSurf2.setInputCloud(globalRawCloud);	
		downSizeFilterSurf2.filter(*globalRawCloudDS);
        mtx.lock();

		*globalMapCloud += *globalRawCloudDS;
		std::cout << "map size: " << globalMapCloud->size() << std::endl;

		if (globalMapCloud->empty()) {
			std::cout << "empty global map cloud!" << std::endl;
			return false;
		}
		pcl::io::savePCDFileASCII(save_directory + "globalmap_lidar_feature.pcd", *globalMapCloud);
		cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed: " << endl;

		return true;
}

/**
 * 更新里程计轨迹
 */
void updatePath(const PointTypePose &pose_in)
{
    string odometryFrame = "camera_init";
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);

    pose_stamped.header.frame_id = odometryFrame;
    pose_stamped.pose.position.x =  pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z =  pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}