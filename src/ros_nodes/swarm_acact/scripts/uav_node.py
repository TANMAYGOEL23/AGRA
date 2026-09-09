#!/usr/bin/env python3
"""
Distributed ROS Noetic Node for Swarm UAV Collision Avoidance with DCACS.
Directly implements the ROS architecture specified in Paper 2 (Adithya & Kundu, Section IV).
"""
import sys
import os
import random
import time
import math

try:
    import rospy
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

DELAY_THRESHOLD = 0.1   # seconds (delta)
CONGESTION_THRESH = 3   # neighbors (gamma)

class UAVROSNode:
    def __init__(self):
        if not ROS_AVAILABLE:
            print("[UAVROSNode] ROS is not installed in the current environment. Standalone simulation engine can be used instead.")
            return

        rospy.init_node('uav_node', anonymous=False)
        
        self.uav_id = rospy.get_param('~uav_id', 1)
        self.total_uavs = rospy.get_param('~total_uavs', 4)
        self.start_x = rospy.get_param('~start_x', 0.0)
        self.start_y = rospy.get_param('~start_y', 0.0)
        self.goal_x = rospy.get_param('~goal_x', 10.0)
        self.goal_y = rospy.get_param('~goal_y', 10.0)
        self.network_delay = rospy.get_param('~network_delay', 0.05)
        
        self.x = self.start_x
        self.y = self.start_y
        self.vx = 0.0
        self.vy = 0.0
        
        # ACACT Gains and Parameters
        self.alpha = 1.0        # Goal attraction
        self.beta = 2.0         # Repulsion gain
        self.ts_base = 0.5      # Base safety time
        self.step_size = 0.1    # Step per tick
        self.v_max = 1.0
        
        # Communication State (DCACS)
        self.comm_mode = "FAST"
        self.loss_prob = 0.2
        self.other_positions = {}  # topic -> [x, y]
        
        # Metrics
        self.t_start = rospy.get_time()
        self.t_travel = 0.0
        self.t_collision = 0.0
        self.reached_goal = False
        
        # Publishers and Subscribers
        self.pub_pos = rospy.Publisher(f'/uav_{self.uav_id}/position', String, queue_size=10)
        self.subs = []
        for i in range(1, self.total_uavs + 1):
            if i != self.uav_id:
                topic = f'/uav_{i}/position'
                sub = rospy.Subscriber(topic, String, self.position_callback, callback_args=topic)
                self.subs.append(sub)
                
        rospy.loginfo(f"[UAV{self.uav_id}] Initialized at ({self.x}, {self.y}) -> Goal ({self.goal_x}, {self.goal_y}) | Delay={self.network_delay}s")

    def update_comm_mode(self):
        """Figure 1 in Paper 2: DCACS mode selection logic."""
        neighbor_count = len(self.other_positions)
        if self.network_delay > DELAY_THRESHOLD or neighbor_count >= CONGESTION_THRESH:
            self.comm_mode = "RELIABLE"
            self.loss_prob = 0.0
        else:
            self.comm_mode = "FAST"
            self.loss_prob = 0.2

    def position_callback(self, msg, topic):
        """Figure 2 in Paper 2: Subscriber callback simulating packet loss & delay."""
        if random.random() < self.loss_prob:
            return  # packet dropped in FAST mode
            
        time.sleep(self.network_delay)
        parts = msg.data.split(",")
        px, py = float(parts[0]), float(parts[1])
        self.other_positions[topic] = [px, py]

    def compute_avoidance_velocity(self):
        """ACACT potential field update (Paper 2 Eq 1-2)."""
        # Attractive force towards goal
        fx = self.alpha * (self.goal_x - self.x)
        fy = self.alpha * (self.goal_y - self.y)
        
        # Repulsive force from neighbors
        ts_adaptive = self.ts_base + self.network_delay
        
        for topic, pos in list(self.other_positions.items()):
            ox, oy = pos[0], pos[1]
            dx = self.x - ox
            dy = self.y - oy
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < 2.0:
                self.t_collision += 0.1  # 10Hz step collision tracking
                
            if 0.001 < dist < 7.0:
                # Repulsive potential
                rep = self.beta / (dist ** 3)
                fx += rep * dx
                fy += rep * dy
                
        # Limit velocity
        mag = math.sqrt(fx*fx + fy*fy)
        if mag > self.v_max:
            fx = (fx / mag) * self.v_max
            fy = (fy / mag) * self.v_max
            
        return fx, fy

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if self.reached_goal:
                rate.sleep()
                continue
                
            # Check goal distance
            dist_to_goal = math.sqrt((self.goal_x - self.x)**2 + (self.goal_y - self.y)**2)
            if dist_to_goal < 0.2:
                self.reached_goal = True
                self.t_travel = rospy.get_time() - self.t_start
                pttr = 1.0 / (1.0 + self.t_travel + self.t_collision)
                rospy.loginfo(f"[UAV{self.uav_id}] Reached goal! Travel: {self.t_travel:.2f}s | Col Time: {self.t_collision:.2f}s | PTTR: {pttr:.5f}")
                rate.sleep()
                continue

            # Update DCACS communication mode
            self.update_comm_mode()
            
            # Broadcast current position
            msg = String()
            msg.data = f"{self.x:.3f},{self.y:.3f}"
            self.pub_pos.publish(msg)
            
            # Compute motion step
            vx, vy = self.compute_avoidance_velocity()
            self.x += vx * 0.1
            self.y += vy * 0.1
            
            rate.sleep()

if __name__ == '__main__':
    node = UAVROSNode()
    if ROS_AVAILABLE:
        node.run()
