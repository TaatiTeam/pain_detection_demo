from AFAR.pain_detector import PainDetector
import cv2
import time


pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt')
print(pain_detector.device)
ref_frame1 = cv2.imread('example_frames/example-reference-frame.png')
ref_frame2 = cv2.imread('example_frames/example-reference-frame.png')
ref_frame3 = cv2.imread('example_frames/example-reference-frame.png')
pain_detector.add_references([ref_frame1, ref_frame2, ref_frame3])
target_frame = cv2.imread('example_frames/example-target-frame.png')
pain_estimate = pain_detector.predict_pain(target_frame)
print(pain_estimate)


num_of_frames = 30
print('Testing frame rate with {} frames'.format(num_of_frames))
start_time = time.time()
for _ in range(num_of_frames):
    pain_detector.predict_pain(target_frame)
print('FPS: {}'.format(num_of_frames / (time.time() - start_time)))