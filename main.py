import numpy as np
import os
import glob
import cv2
from imaging_interview import preprocess_image_change_detection, compare_frames_change_detection



if __name__=='__main__':

	files = glob.glob("./test/dataset/*.png")
	print(len(files))

	min_contour_area = 100  # Adjust as needed
	similarity_threshold = 50000  # Adjust as needed
	deleted_files = 0


	rad = [1, 3, 5, 7, 9]

	image1 = cv2.imread(files[0])
	prev_frame = preprocess_image_change_detection(image1, rad)

	for i in range(1, len(files)):

		image2 = cv2.imread(files[i])
		if image2 is None:
			continue
		next_frame = preprocess_image_change_detection(image2, rad)

		if prev_frame.shape[0]!=next_frame.shape[0] or prev_frame.shape[1]!=next_frame.shape[1]:

			if prev_frame.shape[0] < next_frame.shape[0]:
				next_frame = cv2.resize(next_frame, prev_frame.shape[::-1], interpolation = cv2.INTER_LINEAR)
			else:
				prev_frame = cv2.resize(prev_frame, next_frame.shape[::-1], interpolation = cv2.INTER_LINEAR)


		score, _, _ = compare_frames_change_detection(prev_frame, next_frame, min_contour_area)
		print(f"Score at {i} is {score}")

		if score < similarity_threshold:
			os.remove(files[i-1])
			deleted_files += 1


		prev_frame = next_frame
	    
	print(f"Deleted files : {deleted_files}")





