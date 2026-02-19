#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    # Updated paths for your specific files
    pkg_td3 = get_package_share_directory('td3')
    world = os.path.join(pkg_td3, 'worlds', 'td3.world')
    urdf_file = os.path.join(pkg_td3, 'urdf', 'td_robot.urdf') # Matches your filename
    
    # Updated RViz config
    rviz_file = os.path.join(pkg_td3, 'launch', 'pioneer3dx.rviz') 

    return LaunchDescription([
        # 1. Start Gazebo Server with your custom world
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ),

        # 2. Start Gazebo Client
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gzclient.launch.py')
            ),
        ),

        # 3. Publish Robot State (Sends URDF to Gazebo and RViz)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 
                         'robot_description': open(urdf_file).read()}]
        ),

        # 4. Spawn the Robot as 'r1'
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-topic', 'robot_description',
                       '-entity', 'r1',
                       '-x', '0.0', '-y', '0.0', '-z', '0.1'],
            output='screen',
        ),

        # 5. Start Training Node (Fixed executable name)
        Node(
            package='td3',
            executable='train_4wheel_node.py', # Updated from train_velodyne_node.py
            output='screen'
        ),

        # 6. Start RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',  
            arguments=['-d', rviz_file],
            output='screen'
        ),
    ])
