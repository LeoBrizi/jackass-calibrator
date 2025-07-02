# mad-icp ROS2 node (Jackal 3D SLAM stack)
- Thinking about forking mad-icp and adding PointCloud2 -> MADTree conversion directly.
- Specifically for the Jackal: figure out a way to isolate from the ClearPath stack (odometry, tf publishing), perhaps namespace everything that we add (`/jackass` for example).
- The slam stack should be organized as ROS2 components. If we load components into the same process, we can:
  - Organize the execution of callbacks through `executors`.
  - bypass the DDS layer in between components and achieve zero copies.
