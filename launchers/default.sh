#!/bin/bash
source /environment.sh
dt-launchfile-init
echo "VEHICLE_NAME=$VEHICLE_NAME"

# Start dead reckoning node
#dt-exec roslaunch deadreckoning deadreckoning_node.launch veh:=$VEHICLE_NAME
#dt-exec roslaunch my_package viz_followed_path.launch veh:=ruks007
#dt-exec roslaunch my_map_package map_server.launch veh:=ruks007
# Start your TCP server/logger
dt-exec rosrun my_package vehicle_client.py
#dt-exec rosrun my_package v2v.py

dt-launchfile-join

