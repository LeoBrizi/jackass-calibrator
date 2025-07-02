#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "odometry/pipeline.h"

namespace mad_icp_ros {

class Odometry : public rclcpp::Node {
 public:
  Odometry(const rclcpp::NodeOptions& options);

 protected:
  std::string frame_id_{"base_link"};

  std::string dataset_config_file_path_;
  std::string mad_icp_config_file_path_;

  std::unique_ptr<Pipeline> pipeline_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;

  void pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // store each message here
  ContainerType pc_container_;
  // PointCloud2 -> std::vector -> madtree (can I save 1 conversion if I write
  // PointCloud2 to madtree?)

  double min_range_{0};
  double max_range_{0};

  rclcpp::Time stamp_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;

  void publish_odom() const;
};
}  // namespace mad_icp_ros