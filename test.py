from AFAR.pain_detector import PainDetector
import cv2
from glob import glob


A=PainDetector(image_size=200, afar_checkpoint='AFAR/model_epoch23.pt')
for fp in glob('*[p][n][g]'):
    B=cv2.imread(fp)
    print(A.predict_pain(B))