import sys
import os
import numpy as np
import cv2
import math
import argparse
import time


image_folder = './test_images'
min_region_size = 14400

# rect = ((min_x, min_y), (max_x, max_y))
def place_bounding_box(image, rect):
	image = cv2.rectangle(image,(rect[0][1],rect[0][0]),(rect[1][1],rect[1][0]),(0,255,0),3)
	return image

	
def calc_bounding_box(coord_list):
	min_x = min(coord_list, key = lambda t: t[0])[0]

	min_y = min(coord_list, key = lambda t: t[1])[1]

	max_x = max(coord_list, key = lambda t: t[0])[0]

	max_y = max(coord_list, key = lambda t: t[1])[1]

	rect = ((min_x, min_y),(max_x, max_y))

	return rect

def scan_image(image):
	region_list = []

	rows, cols, channels = image.shape
	visited_matrix = [[False for col in range(cols)] for row in range(rows)]
	
	directions = [(-1,-1), (-1, 0), (-1, 1), (0,-1), (0, 1), (1,-1), (1, 0), (1, 1)]

	for row in range(rows):
		for col in range(cols):
			# top level linear scan
			if not visited_matrix[row][col]:
				visited_matrix[row][col] = True
				region = not np.array_equal(image[row,col], [0,0,0])
				# first step of flood fill
				if region:
					coord_list = []
					coord_list.append((row,col))
					to_visit = []
					for direction in directions:
						new_loc = (row + direction[0],col + direction[1])
						if not new_loc[0] < 0 and not new_loc[0] > (rows-1) and not new_loc[1] < 0 and not new_loc[1] > (cols-1):
							if not visited_matrix[new_loc[0]][new_loc[1]]:
								to_visit.append(new_loc)
					#start flood fill
					while to_visit:
						current_loc = to_visit.pop()
						visited_matrix[current_loc[0]][current_loc[1]] = True
						if not np.array_equal(image[current_loc[0],current_loc[1]], [0,0,0]):
							coord_list.append(current_loc)
							for direction in directions:
								new_loc = (current_loc[0] + direction[0], current_loc[1] + direction[1])
								if not new_loc[0] < 0 and not new_loc[0] > (rows-1) and not new_loc[1] < 0 and not new_loc[1] > (cols-1):
									if not visited_matrix[new_loc[0]][new_loc[1]]:
										to_visit.append(new_loc)
					if len(coord_list) > min_region_size:
						region_list.append(coord_list)

	return region_list

start = time.time()

file_names = os.listdir(image_folder)

image_paths = []

for file in file_names:
	image_paths.append(os.path.join(image_folder, file))

output = []

for image_path in image_paths:
	image = cv2.imread(image_path, 1)
	region_list = scan_image(image)
	if region_list:
		rects = []
		for region in region_list:
			rects.append(calc_bounding_box(region))
		image_with_box = image.copy()
		for rect in rects:
			place_bounding_box(image_with_box, rect)
		output.append(image_with_box)
		
end = time.time()

for index, image in enumerate(output):
	cv2.imshow(('Output ' + str(index)), image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


total = end-start
print ('Total: ' + str(total))
print ('Time per image: ' + str(total/len(image_paths)))

