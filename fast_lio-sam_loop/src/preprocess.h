#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE{AVIA = 1, VELO16, OUST64, PANDAR32, RS16}; //{1, 2, 3, 4, 5}
enum TIME_UNIT{SEC = 0, MS = 1, US = 2, NS = 3};
enum Feature{Nor, Poss_Plane, Real_Plane, Edge_Jump, Edge_Plane, Wire, ZeroPoint};

// 枚举类型：位置标识
enum Surround{Prev, Next};

// 枚举类型：表示有跨越边的类型
enum E_jump{Nr_nor, Nr_zero, Nr_180, Nr_inf, Nr_blind};

// orgtype类：用于存储激光雷达点的一些其他属性
struct orgtype
{
  double range;     // 点云在xy平面离雷达中心的距离
  double dista;     // 当前点与后一个点之间的距离
  //假设雷达原点为O 前一个点为M 当前点为A 后一个点为N
  double angle[2];  // 这个是角OAM和角OAN的cos值
  double intersect; // 这个是角MAN的cos值
  E_jump edj[2];
  Feature ftype;
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor;
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

namespace velodyne_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      float time;
      uint16_t ring;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

namespace ouster_ros {
  struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint32_t t;
      uint16_t reflectivity;
      uint8_t  ring;
      uint16_t ambient;
      uint32_t range;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

namespace pandar_pcl {
    struct PointXYZIT {
    PCL_ADD_POINT4D   //添加pcl里xyz
    float intensity;
    double timestamp;
    uint16_t ring;                   ///< laser ring number
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned,确保定义新类型点云内存与SSE对齐
    } EIGEN_ALIGN16;                   // 强制SSE填充以正确对齐内存
}// namespace pandar_pcl 

POINT_CLOUD_REGISTER_POINT_STRUCT(
    pandar_pcl::PointXYZIT,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (double, timestamp, timestamp)
    (uint16_t, ring, ring)
)

//ANCHOR robosense modify
namespace robosense_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        std::uint16_t ring;
        double timestamp;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}

// namespace robosense_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(robosense_ros::Point,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                          (float, intensity, intensity)
                                          (std::uint16_t, ring, ring)
                                          (double, timestamp, timestamp)
)

struct PointXYZIRPYT
{
  PCL_ADD_POINT4D
      PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
  float roll;
  float pitch;
  float yaw;
  double time;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
(float, x, x) (float, y, y)
(float, z, z) (float, intensity, intensity)
(float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
(double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class Preprocess
{
  public:
//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();

  // 对Livox自定义Msg格式的激光雷达数据进行处理
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf;
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale;
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit;
  double blind;
  bool feature_enabled, given_offset_time;
  ros::Publisher pub_full, pub_surf, pub_corn;
    

  private:
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);  // 用于对Livox激光雷达数据进行处理
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void rslidar_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);   // 对rs-lidar进行处理，不提取特征 
  void pandar_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
