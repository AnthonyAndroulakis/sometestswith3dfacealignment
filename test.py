#video to frames
'''
import cv2
vidcap = cv2.VideoCapture('obama.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
'''

import face_alignment
from skimage import io
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device='cpu')
preds = fa.get_landmarks_from_directory('frames')

import collections
pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

import numpy as np
for i in list(preds.keys()):
	predlist=np.array(preds[i][0]).tolist()
	sequence_containing_x_vals=[i[0] for i in predlist]
	sequence_containing_y_vals=[i[1] for i in predlist]
	sequence_containing_z_vals=[i[2] for i in predlist]

#3D graph
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

for i in list(preds.keys()):
	fig = plt.figure()
	ax = Axes3D(fig)
	predlist=np.array(preds[i][0]).tolist()
	sequence_containing_x_vals=[i[0] for i in predlist]
	sequence_containing_y_vals=[i[1] for i in predlist]
	sequence_containing_z_vals=[i[2] for i in predlist]
	ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
	plt.show()
	time.sleep(10)
	fig.clear()
	print('closed')
'''
