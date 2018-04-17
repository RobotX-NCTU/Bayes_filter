#!/usr/bin/env python
import rospy
from sensor_msgs.msg import NavSatFix

import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import normalize
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA


class gps_kalman_node():
	def __init__(self):
		self.gps_sub = rospy.Subscriber("/fix", NavSatFix, self.gps_cb)
		self.marker_pub = rospy.Publisher('GPS_marker_kalman', Marker, queue_size = 10)
		self.marker_cov_pub = rospy.Publisher('GPS_marker_cov', Marker, queue_size = 10)
		self.marker_raw_pub = rospy.Publisher('GPS_marker_raw', Marker, queue_size = 10)
		self.lock = 0
		self.latitude = -1
		self.longitude = -1
		self.altitude = -1
		self.covariance = np.zeros((9,), dtype=float)
		self.started = 0
		self.prior_lat = 0
		self.prior_long = 0
		self.prior_alt = 0
		self.init_latitude = -1
		self.init_longitude = -1
		self.init_altitude = -1
		self.id = 0
	def gps_cb(self, data):
		if self.lock == 0:
			self.latitude = data.latitude*110574
			self.longitude = data.longitude*111320*0.915
			self.altitude = data.altitude
			self.covariance = data.position_covariance
		
	def process(self):
		if self.init_latitude == -1 and self.latitude != -1:
			self.init_latitude = self.latitude
			self.init_longitude = self.longitude
			self.init_altitude = self.altitude
			return
		if self.started == 0 and self.init_latitude != -1 :
			# initialize prior
			self.prior_lat = norm(loc = self.init_latitude, scale = 100)
			self.prior_long = norm(loc = self.init_longitude, scale = 100)
			self.prior_alt = norm(loc = self.init_altitude, scale = 100)
			self.started = 1
			return
		if self.started == 0:
			return
		#print self.latitude-self.init_latitude
		self.lock = 1
		latitude = self.latitude
		longitude = self.longitude
		altitude = self.altitude
		covariance = self.covariance
		self.lock = 0
		#print latitude, longitude, altitude


		#prediction step
		kernel = norm(loc = 0, scale = 0.08)
		predicted_lat = norm(loc = self.prior_lat.mean()+kernel.mean(), scale = np.sqrt(self.prior_lat.var()+kernel.var()))
		predicted_long = norm(loc = self.prior_long.mean()+kernel.mean(), scale = np.sqrt(self.prior_long.var()+kernel.var()))
		predicted_alt = norm(loc = self.prior_alt.mean()+kernel.mean(), scale = np.sqrt(self.prior_alt.var()+kernel.var()))
		#update step
		posterior_lat = self.update_con(predicted_lat, latitude, covariance[4])
		posterior_long = self.update_con(predicted_long, longitude, covariance[0])
		posterior_alt = self.update_con(predicted_alt, altitude, covariance[8])
		self.prior_lat = posterior_lat
		self.prior_long = posterior_long
		self.prior_alt = posterior_alt

		


		print "lat:", (posterior_lat.mean()-self.init_latitude), posterior_lat.var()
		print "long:", (posterior_long.mean()-self.init_longitude), posterior_long.var()
		print "alt:", (posterior_alt.mean()-self.init_altitude), posterior_alt.var()
		#print "lat:",  covariance[4]
		#print "long:", covariance[0]
		#print "alt:", covariance[8]

		## Draw marker
		
		marker = Marker(type=Marker.SPHERE, \
			id=self.id, lifetime=rospy.Duration(), \
			pose=Pose(Point(float((posterior_lat.mean()-self.init_latitude)), float((posterior_long.mean()-self.init_longitude)), float(posterior_alt.mean()-self.init_altitude)), \
			Quaternion(0, 0, 0, 1)),\
			scale=Vector3(0.3, 0.3, 0.3),\
			header=Header(frame_id='gps'),\
			color=ColorRGBA(0.0, 0.0, 0.0, 1))
		self.marker_pub.publish(marker)
		
		marker_cov = Marker(type=Marker.SPHERE, \
			id=self.id+10000, lifetime=rospy.Duration(0.1), \
			pose=Pose(Point(float((posterior_lat.mean()-self.init_latitude)), float((posterior_long.mean()-self.init_longitude)), float(posterior_alt.mean()-self.init_altitude)), \
			Quaternion(0, 0, 0, 1)),\
			#scale=Vector3(np.sqrt(covariance[4]), np.sqrt(covariance[0]), np.sqrt(covariance[8])),\
			scale=Vector3(np.sqrt(posterior_lat.var()), np.sqrt(posterior_long.var()), np.sqrt(posterior_alt.var())), \
			header=Header(frame_id='gps'),\
			color=ColorRGBA(1, 0.6, 0.8, 0.3))
		self.marker_cov_pub.publish(marker_cov)


		marker_raw = Marker(type=Marker.SPHERE, \
			id=self.id+200000, lifetime=rospy.Duration(), \
			pose=Pose(Point(float((latitude-self.init_latitude)), float((longitude-self.init_longitude)), float(altitude-self.init_altitude)), \
			Quaternion(0, 0, 0, 1)),\
			scale=Vector3(0.3, 0.3, 0.3),\
			header=Header(frame_id='gps'),\
			color=ColorRGBA(1.0, 0.0, 0.0, 1))
		self.marker_raw_pub.publish(marker_raw)
		
		self.id += 1


	def measurement(self, measurementx, variance):
		likelihood = norm(loc = measurementx, scale = np.sqrt(variance))
		return likelihood

	def gaussian_multiply(self, g1, g2):
		g1_mean, g1_var = g1.stats(moments='mv')
		g2_mean, g2_var = g2.stats(moments='mv')
		mean = (g1_var * g2_mean + g2_var * g1_mean) / (g1_var + g2_var)
		variance = (g1_var * g2_var) / (g1_var + g2_var)
		#print mean, variance
		return norm(loc = mean, scale = np.sqrt(variance))

	def update_con(self, prior, measurementz, covariance):
		likelihood = self.measurement(measurementz, covariance)
		posterior = self.gaussian_multiply(likelihood, prior)
		return posterior

def main():
	ic = gps_kalman_node()
	rospy.init_node('gps_kalman_node', anonymous = True)
	try:
		while(1):
			ic.process()
			rospy.sleep(0.1)
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main()