import deepvoNet as deepvo
import dataloader as loader

voloader = loader.voDataLoader(img_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/dataset/sequences',
                               pose_dataset_path='/media/luwis/Linux Workspace/ICSL_Project/Visual SLAM/KITTI_data_odometry_color/data_odometry_poses/dataset/poses',
                               transform=None,
                               test=False,
                               train_sequence=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
                               test_sequence=['01'])
while True:
    voloader.__getitem__(0)

deepvo_model = deepvo.DeepVONet()