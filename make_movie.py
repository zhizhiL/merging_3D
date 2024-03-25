import cv2
import os

def create_movie_from_frames(target, frame_folder, output_path, fps=2):
    frame_files = [f for f in os.listdir(frame_folder) if f.endswith('.png') and f.startswith(target)]
    frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  

    frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs like 'XVID'
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frame_folder, frame_file))
        video_writer.write(frame)

    video_writer.release()

target = 'timestep'
N_realisation = 3
frame_folder = 'noInflux_random_sims/temp'
output_path = frame_folder + '/output_' + target +'_realisation_' + str(N_realisation) + '.mp4'
create_movie_from_frames(target, frame_folder, output_path)
