import numpy as np

#
# pose_file_path = '/Users/qanelph/Downloads/data_poses/2013_05_28_drive_0000_sync/poses.txt'
# pose2world_rows = np.loadtxt(pose_file_path)
# pose2world_data = np.reshape(pose2world_rows[:, 1:], (-1, 3, 4))
# print()
#
# frames_numbers = list(np.reshape(pose2world_rows[:, :1], (-1)).astype(int))
# pose2world = {}
#
# current_data = pose2world_data[0]
# for frame_index in range(1, frames_numbers[-1] + 1):
#     if frame_index in frames_numbers:
#         mapped_index = frames_numbers.index(frame_index)
#         current_data = pose2world_data[mapped_index]
#
#     pose2world[frame_index] = current_data
#
#
# R = pose2world[0][:3, :3]
# T = pose2world[0][:3, 3]
# print()