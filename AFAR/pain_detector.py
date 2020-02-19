import sys
sys.path.append('..')
sys.path.append('.')
import torch
import face_alignment as FAN
from AFAR.afar import AFAR
import numpy as np
import cv2
from skimage.transform import SimilarityTransform, PiecewiseAffineTransform, warp


class PainDetector:
    def __init__(self, afar_checkpoint='', fan_checkpoint='', channels=1, image_size=160, fc2_size=400):
        """
        :param afar_checkpoint: AFAR checkpoint path, cannot be empty
        :param fan_checkpoint: FAN checkpoint path, if empty will download pretrained model
        :param channels: number or heads for output
        :param image_size: image size after face detection, must correspond to afar_checkpoint
        :param fc2_size: size of fc2, must correspond to afar_checkpoint
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.image_size = image_size
        # load FAN landmark detector including SFD face detector
        self.FAN = FAN.FaceAlignment(FAN.LandmarksType._2D, flip_input=True, device=self.device,
                                     check_point_path=fan_checkpoint)
        self.face_detector = self.FAN.get_landmarks_from_image
        self.mean_lmks = np.load('standard_face_68.npy')
        self.mean_lmks = self.mean_lmks * 155 / self.mean_lmks.max()
        self.mean_lmks[:, 1] += 15
        # load AFAR model
        fc_size = {160: 4608, 200: 10368}
        self.AFAR = AFAR(channels=channels, fc_size=fc_size[image_size], fc2_size=fc2_size).to(self.device)
        self.load_afar_model(afar_checkpoint)
        self.AFAR.eval()

    def load_afar_model(self, checkpoint):
        self.AFAR.load_state_dict(torch.load(checkpoint, map_location=self.device))

    @staticmethod
    def crop_image(frame, bbox):
        fh, fw = frame.shape[:2]
        bl, bt, br, bb = bbox
        fh, fw, bl, bt, br, bb = int(fh), int(fw), int(bl), int(bt), int(br), int(bb)

        a_slice = frame[max(0, min(bt, fh)):min(fh, max(bb, 0)), max(0, min(bl, fw)):min(fw, max(br, 0)), :]
        new_image = np.zeros((bb - bt, br - bl, 3), dtype=np.float32)
        new_image[max(0, min(bt, fh)) - bt:min(fh, max(bb, 0)) - bt, max(0, min(bl, fw)) - bl:min(fw, max(br, 0)) - bl,
                  :] = a_slice

        h, w = new_image.shape[:2]
        m = max(h, w)
        square_image = np.zeros((m, m, 3), dtype=np.float32)
        square_image[(m - h) // 2:h + (m - h) // 2, (m - w) // 2:w + (m - w) // 2, :] = new_image
        return square_image

    @staticmethod
    def similarity_transform(image, landmarks):
        anchor = np.array([[110, 71], [210, 71], [160, 170]], np.float32)
        idx = [36, 45, 57]
        tform = SimilarityTransform()
        tform.estimate(landmarks[idx, :], anchor)
        sim_mat = tform.params[:2, :]
        dst = cv2.warpAffine(image, sim_mat, (image.shape[1], image.shape[0]))
        dst_lmks = np.matmul(np.concatenate((landmarks, np.ones((landmarks.shape[0], 1))), 1), sim_mat.T)[:, :2]
        return dst, dst_lmks

    @staticmethod
    def warp_lmks(tform, coords):
        out = np.empty_like(coords, np.double)
        # determine triangle index for each coordinate
        simplex = tform._inverse_tesselation.find_simplex(coords, tol=None)
        if (simplex == -1).any():  # simplex==-1 when point falls out of convex hull
            pass  # don;t know what to do yet
        for index in range(len(tform._inverse_tesselation.vertices)):
            # affine transform for triangle
            affine = tform.inverse_affines[index]
            # all coordinates within triangle
            index_mask = simplex == index
            out[index_mask, :] = affine(coords[index_mask, :])
        return out

    @staticmethod
    def piecewise_affine_transform(image, source_lmks, target_lmks):
        anchor = list(range(31)) + [36, 39, 42, 45, 48, 51, 54, 57]
        tgt_lmks = target_lmks[anchor, :]
        dst_lmks = source_lmks[anchor, :]
        tform = PiecewiseAffineTransform()
        tform.estimate(tgt_lmks, dst_lmks)
        dst = warp(image, tform, output_shape=image.shape[:2]).astype(np.float32)
        return dst

    def predict_pain(self, image):
        """
        Main predictor function, takes an image as input and returns a float number as pain prediction
        :param image: RGB input image, size (Height x Width x Channel)
        :return: a float32 number
        """
        landmarks = self.face_detector(image)
        pred_pains = []
        with torch.no_grad():
            for landmark in landmarks:
                image_face, lmks = self.similarity_transform(image, landmark)
                image_face = self.piecewise_affine_transform(image_face, lmks, self.mean_lmks)
                landmark = self.mean_lmks.round().astype(np.int)
                b_box = [landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()]
                image_face = self.crop_image(image_face, b_box)
                image_face = cv2.resize(image_face, (self.image_size, self.image_size))
                if len(image_face.shape) > 2 and image_face.shape[2] == 3:
                    image_face = np.matmul(image_face, np.array([[0.114], [0.587], [0.299]]))
                image_face = image_face.transpose((2, 0, 1))[None].astype(np.float32)
                pred_pains.append(self.AFAR(torch.from_numpy(image_face)))
        return pred_pains
