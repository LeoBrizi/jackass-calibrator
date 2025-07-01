#include "mad_icp_ros/odometry.h"

#include <yaml-cpp/yaml.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <iostream>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

Eigen::Matrix4d parseMatrix(const std::vector<std::vector<double>>& vec);

int num_cores = 4;
int num_keyframes = 4;
bool realtime = false;
bool kitti = false;

mad_icp_ros::Odometry::Odometry(const rclcpp::NodeOptions& options)
    : Node("mad_icp", options) {
  // Load parameters file
  std::string package_share_dir =
      ament_index_cpp::get_package_share_directory("mad_icp_ros");

  this->declare_parameter<std::string>(
      "dataset_config_file",
      package_share_dir + "/config/datasets/vbr_os0.cfg");

  this->get_parameter("dataset_config_file", dataset_config_file_path_);

  this->declare_parameter<std::string>(
      "mad_icp_config_file", package_share_dir + "/config/default.cfg");

  this->get_parameter("mad_icp_config_file", mad_icp_config_file_path_);

  // Parse the parameters
  YAML::Node yaml_dataset_config, yaml_mad_icp_config;
  // RCLCPP_INFO(get_logger(), dataset_config_file_path_.c_str());
  // RCLCPP_INFO(get_logger(), mad_icp_config_file_path_.c_str());

  yaml_dataset_config = YAML::LoadFile(dataset_config_file_path_);
  yaml_mad_icp_config = YAML::LoadFile(mad_icp_config_file_path_);

  min_range_ = yaml_dataset_config["min_range"].as<double>();
  max_range_ = yaml_dataset_config["max_range"].as<double>();
  const double sensor_hz = yaml_dataset_config["sensor_hz"].as<double>();
  const bool deskew = yaml_dataset_config["deskew"].as<bool>();
  // parsing lidar in base homogenous transformation
  const auto lidar_to_base_vec = yaml_dataset_config["lidar_to_base"]
                                     .as<std::vector<std::vector<double>>>();
  const Eigen::Matrix4d lidar_to_base = parseMatrix(lidar_to_base_vec);

  // parse mad-icp configuration
  const double b_max = yaml_mad_icp_config["b_max"].as<double>();
  const double b_min = yaml_mad_icp_config["b_min"].as<double>();
  const double b_ratio = yaml_mad_icp_config["b_ratio"].as<double>();
  const double p_th = yaml_mad_icp_config["p_th"].as<double>();
  const double rho_ker = yaml_mad_icp_config["rho_ker"].as<double>();
  const int n = yaml_mad_icp_config["n"].as<int>();

  // Instance a Pipeline
  pipeline_ =
      std::make_unique<Pipeline>(sensor_hz, deskew, b_max, rho_ker, p_th, b_min,
                                 b_ratio, num_keyframes, num_cores, realtime);

  // Instance the PointCloud2 subscriber

  // set the qos to match the ouster
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
                 .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
                 .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
  pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "points", qos,
      std::bind(&Odometry::pointcloud_callback, this, std::placeholders::_1));
}

void mad_icp_ros::Odometry::pointcloud_callback(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  // Convert the PointCloud2 message to mad_icp's ContainerType (std::vector
  // of 3d points)
  auto stamp = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

  // mad icp uses unordered clouds
  auto height = msg->height;
  auto width = msg->width;

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
  // TODO filter on intensity
  // sensor_msgs::PointCloud2ConstIterator<float> iter_int(*msg, "intensity");

  pc_container_.clear();  // is this necessary?
  pc_container_.reserve(msg->height * msg->width);

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    auto point = Eigen::Vector3d{*iter_x, *iter_y, *iter_z};
    if (point.norm() < min_range_ || point.norm() > max_range_ ||
        std::isnan(point.x()) || std::isnan(point.y()) || std::isnan(point.z()))
      continue;

    pc_container_.emplace_back(point);
  }

  pipeline_->compute(stamp, pc_container_);
  std::cout << pipeline_->currentPose() << std::endl;
}

Eigen::Matrix4d parseMatrix(const std::vector<std::vector<double>>& vec) {
  // this need to be done to respect config file <shit>
  std::vector<double> mat_vec;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      mat_vec.push_back(vec[r][c]);
    }
  }
  return Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(
      mat_vec.data());
}

RCLCPP_COMPONENTS_REGISTER_NODE(mad_icp_ros::Odometry)
