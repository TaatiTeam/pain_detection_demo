from AFAR.pain_detector import PainDetector
import cv2
from glob import glob


pain_detector = PainDetector(image_size=160, checkpoint_path='checkpoints/50342566/50343918_3/model_epoch4.pt')
ref_frame1 = cv2.imread('example_frames/example-reference-frame.png')
ref_frame2 = cv2.imread('example_frames/example-reference-frame.png')
pain_detector.add_references([ref_frame1, ref_frame2])
target_frame = cv2.imread('example_frames/example-target-frame.png')
pain_estimate = pain_detector.predict_pain(target_frame)
print(pain_estimate)
