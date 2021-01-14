from pain_detector import PainDetector
import cv2
import time
import argparse


parser = argparse.ArgumentParser(description='Trains!')
parser.add_argument('-unbc_only', action='store_true', default=False, help='Load the checkpoint that was only trained on UNBC. Otherwise loaded the checkpoint that was train on Both UNBC and UofR datasets')
parser.add_argument('-test_framerate', action='store_true', default=False, help='Runs frame rate test as well')
args = parser.parse_args()

if args.unbc_only:
    pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/59448122/59448122_3/model_epoch13.pt', num_outputs=7)
else:
    pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt', num_outputs=40)

print('Device: ', pain_detector.device)
ref_frame1 = cv2.imread('example_frames/example-reference-frame.png')
ref_frame2 = cv2.imread('example_frames/example-reference-frame.png')
ref_frame3 = cv2.imread('example_frames/example-reference-frame.png')
# In this example the reference frames are identical, but in a real scenario, the idea is to use different
# reference frames from the same person. Ideally, the reference frames should have a neutral expression and should
# exhibit slight lighting and camera angle variations.
pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3])
target_frame = cv2.imread('example_frames/example-target-frame.png')
pain_estimate = pain_detector.predict_pain(target_frame)
print(pain_estimate)

if args.test_framerate:
    num_of_frames = 30
    print('Testing frame rate with {} frames'.format(num_of_frames))
    start_time = time.time()
    for _ in range(num_of_frames):
        pain_detector.predict_pain(target_frame)
    print('FPS: {}'.format(num_of_frames / (time.time() - start_time)))