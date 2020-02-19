from __future__ import print_function
import os
import torch
from torch.utils.model_zoo import load_url
from torch.utils.data import DataLoader
from enum import Enum
from skimage import io
from skimage import color
import scipy
import numpy as np
import cv2
import pdb
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

from .models import FAN, ResNetDepth
#from utils.utils import *
from .utils import *
from tqdm import tqdm


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``_2halfD`` - this points represent the projection of the 3D points into 3D
    ``_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    _2D = 1
    _2halfD = 2
    _3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value

models_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE, check_point_path='',
                 device='cuda', flip_input=False, face_detector='sfd', verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                          globals(), locals(), [face_detector], 0)
        self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose)

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)
        if landmarks_type == LandmarksType._2D:
            network_name = '2DFAN-' + str(network_size)
        else:
            network_name = '3DFAN-' + str(network_size)

        model_dir = os.path.join(os.getcwd(), 'pretrained')
        if os.path.isfile(check_point_path):
            fan_weights = torch.load(check_point_path, map_location=device)
        else:
            print(check_point_path, ' is not a file. Loading model from: ', model_dir)
            fan_weights = load_url(models_urls[network_name], model_dir=model_dir, map_location=device)
        self.face_alignment_net.load_state_dict(fan_weights)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType._3D:
            self.depth_prediciton_net = ResNetDepth()

            depth_weights = load_url(models_urls['depth'], model_dir=model_dir, map_location=device)
            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks_from_image(self, image_or_path, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        if isinstance(image_or_path, str):
            try:
                image = io.imread(image_or_path)
            except IOError:
                print("error opening file :: ", image_or_path)
                return None
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        if detected_faces is None:
            detected_faces = self.face_detector.detect_from_image(image[..., ::-1].copy())

        if len(detected_faces) == 0:
            print("Warning: No faces were detected.")
            return None
        # bb = list[(x1,y1,x2,y2),...]
        torch.set_grad_enabled(False)
        landmarks = []
        for i, bb in enumerate(detected_faces):
            center = torch.FloatTensor(
                [bb[2] - (bb[2] - bb[0]) / 2.0, bb[3] - (bb[3] - bb[1]) / 2.0])
            center[1] = center[1] - (bb[3] - bb[1]) * 0.12  # Not sure where 0.12 comes from
            # Not sure where this calculation of `scale` comes from
            scale = (bb[2] - bb[0] + bb[3] - bb[1]) / self.face_detector.reference_scale

            inp = crop(image, center, scale)
            inp = torch.from_numpy(inp.transpose(
                (2, 0, 1))).float()

            inp = inp.to(self.device)
            inp.div_(255.0).unsqueeze_(0)

            out = self.face_alignment_net(inp)[-1].detach()
            if self.flip_input:
                out += flip(self.face_alignment_net(flip(inp))
                            [-1].detach(), is_label=True)
            out = out.cpu()
            pts, pts_img = get_preds_fromhm(out, center.unsqueeze(0), torch.tensor([scale]))
            # pts, pts_img = self.get_preds_fromhm_subpixel(out, center.unsqueeze(0), torch.tensor([scale]))
            pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)

            if self.landmarks_type == LandmarksType._3D:
                heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
                for i in range(68):
                    if pts[i, 0] > 0:
                        heatmaps[i] = draw_gaussian(
                            heatmaps[i], pts[i], 2)
                heatmaps = torch.from_numpy(
                    heatmaps).unsqueeze_(0)

                heatmaps = heatmaps.to(self.device)
                depth_pred = self.depth_prediciton_net(
                    torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
                pts_img = torch.cat(
                    (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

            landmarks.append(pts_img.numpy())

        return landmarks

    @staticmethod
    def create_target_heatmap(target_landmarks, centers, scales):
        """
        Receives a batch of landmarks and returns a heatmap for each image in the batch
        :param target_landmarks: the batch is expected to have the dim (n x 68 x 2). Where n is the batch size
        :return: returns a (n x 68 x 256 x 256) batch of heatmaps
        """
        # todo: see if you can vectorize the for loop
        heatmaps = np.zeros((target_landmarks.shape[0], 68, 64, 64), dtype=np.float32)
        for i in range(heatmaps.shape[0]):
            for p in range(68):
                # Lua code from https://github.com/1adrianb/face-alignment-training/blob/master/dataset-images.lua:
                # drawGaussian(out[i], transform(torch.add(pts[i],1), c, s, 0, 64), 1)
                # Not sure why it adds 1 to each landmark before transform.
                landmark_cropped_coor = transform(target_landmarks[i, p] + 1, centers[i], scales[i], 64, invert=False)
                heatmaps[i, p] = draw_gaussian(heatmaps[i, p], landmark_cropped_coor, 1)
        return torch.tensor(heatmaps)

    @staticmethod
    def create_bounding_box(target_landmarks, expansion_factor=0.0):
        """
        gets a batch of landmarks and calculates a bounding box that includes all the landmarks per set of landmarks in
        the batch
        :param target_landmarks: batch of landmarks of dim (n x 68 x 2). Where n is the batch size
        :param expansion_factor: expands the bounding box by this factor. For example, a `expansion_factor` of 0.2 leads
        to 20% increase in width and height of the boxes
        :return: a batch of bounding boxes of dim (n x 4) where the second dim is (x1,y1,x2,y2)
        """
        # Calc bounding box
        x_y_min, _ = target_landmarks.reshape(-1, 68, 2).min(dim=1)
        x_y_max, _ = target_landmarks.reshape(-1, 68, 2).max(dim=1)
        # expanding the bounding box
        expansion_factor /= 2
        bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
        bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
        x_y_min[:, 0] -= bb_expansion_x
        x_y_max[:, 0] += bb_expansion_x
        x_y_min[:, 1] -= bb_expansion_y
        x_y_max[:, 1] += bb_expansion_y
        return torch.cat([x_y_min, x_y_max], dim=1)

    def crop_batch(self, bb, images):
        """
        Note: the crop operation is not vectorizable
        :param bb:
        :param images:
        :return:
        """
        center = torch.stack([bb[:, 2] - (bb[:, 2] - bb[:, 0]) / 2.0, bb[:, 3] - (bb[:, 3] - bb[:, 1]) / 2.0], dim=1)
        center[:, 1] = center[:, 1] - (bb[:, 3] - bb[:, 1]) * 0.12  # Not sure where 0.12 comes from
        # Not sure where this calculation of `scale` comes from
        scale = (bb[:, 2] - bb[:, 0] + bb[:, 3] - bb[:, 1]) / self.face_detector.reference_scale

        cropped_images = []
        for i in range(images.shape[0]):
            cropped_images.append(torch.tensor(crop(images[i].numpy(), center[i].numpy(), scale[i].numpy())))
        return torch.stack(cropped_images, 0), center, scale

    def calc_landmarks_MSE(self, ground_truth_landmarks, model_landmarks, landmarks_to_use=None):
        """
        Takes in two nx68x2 ndarrays of frame landmarks and returns the MSE of euclidian distance between them
        n is the # of frames.
        `normalized_error` is normalized by intercanthal distance per frame
        `pixel_error` is MSE per landmark
        If `landmarks_to_use` is `None` all landmarks are used. Otherwise an array of landmarks to be used for error
        calculation should be passed.
        """
        intercanthal_distances = torch.sqrt(
            ((ground_truth_landmarks[:, 42:43, :] - ground_truth_landmarks[:, 39:40, :])**2).sum(dim=2)).squeeze()
        if landmarks_to_use:
            ground_truth_landmarks = ground_truth_landmarks[:, landmarks_to_use, :]
            model_landmarks = model_landmarks[:, landmarks_to_use, :]
        pixel_error = torch.sqrt(((ground_truth_landmarks - model_landmarks)**2).sum(dim=2)).mean(dim=1)
        normalized_error = torch.div(pixel_error, intercanthal_distances)
        return normalized_error.mean(), pixel_error.mean()

    def fit(self, dataset, criterion, optim, batch_size, noise_range=0.99, epoch=0, shuffle=False):
        """
        Runs one epoch on the `net` using the `optimizer` and `criterion`
        :param dataset: training dataset
        :param criterion: loss function
        :param optim: the optimizer to use on the model
        :param noise_range: This should be between 0 and 1. The smaller it is, the more drastic the bounding boxes will
            be shifted around
        :return: average loss for epoch
        """
        self.face_alignment_net.train()
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        batches_per_epoch = len(data_loader)
        pbar = tqdm(data_loader, total=batches_per_epoch)
        for batch_num, batch in enumerate(pbar):
            images = batch[0]
            target_landmarks = batch[1]
            # Calc bounding box
            bb = self.create_bounding_box(target_landmarks, expansion_factor=0.05)  # `expansion_factor` is a magic number which can be experimented with
            # Adding noise to the bounding box. Here the bounding box is created from the ground truth landmarks, so it
            # is always good. At test time bb is generated by a face detector which won't always be good. Furthermore,
            # different face detectors may have statistically significant differences in the character of the bb they
            # generate. So in and effort to make the network agnostic to bb, we add noise to it.
            # the bb will be shifted around.
            # bb = torch.mul(bb, torch.zeros_like(bb).uniform_(noise_range, 2 - noise_range))

            # Crop images
            inp, centers, scales = self.crop_batch(bb, images)
            inp = inp.permute((0, 3, 1, 2)).float()

            inp = inp.to(self.device)
            inp.div_(255.0)

            optim.zero_grad()
            output = self.face_alignment_net(inp)[-1]  # [-1] is to get the output of the last hourglass block
            # output = torch.flip(output, dims=(1,))  # todo: this is a trial, remember to remove

            #pts, pts_img = get_preds_fromhm(output, centers, scales)
            pts, pts_img = get_preds_fromhm_old(output, centers, scales)
            # pts, pts_img = self.get_preds_fromhm_subpixel(output, centers, scales)
            # pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            normalized_pixel_mse, pixel_mse = self.calc_landmarks_MSE(target_landmarks, pts_img)
            mse_meter.update(pixel_mse)

            # creating target heatmaps by drawing gaussians on landmark positions. Each landmark has its own heatmap, so
            # the landmarks for each image is represented by a 68x64x64 heatmap.
            target_heatmaps = self.create_target_heatmap(target_landmarks, centers, scales)
            loss = criterion(output, target_heatmaps.to(self.device))
            loss.backward()
            optim.step()
            loss_meter.update(loss.item())
            pbar.set_description("[ batch loss: {:.6f} | batch MSE: {:.4f}] ".format(loss.item(), pixel_mse))
            # list[(x1,y1,x2,y2),...]
            if batch_num == 0:
                save_dir = 'train_monitoring_overlays'
                if not path.exists(save_dir):
                    makedirs(save_dir)
                # Save images of the first batch for monitoring purposes across epochs
                for i, image in enumerate(images):
                    image_overlaid = draw_landmarks(image.detach().numpy(), pts_img[i].numpy(), color=(255, 0, 0))
                    image_overlaid = draw_landmarks(image_overlaid, target_landmarks[i].numpy())
                    scipy.misc.imsave(path.join(save_dir, '{}-{}.jpg'.format(epoch, i)), image_overlaid)

        return loss_meter.avg, mse_meter.avg

    def val(self, dataset, criterion):
        """
        Run validation on the dataset and return the loss
        :param dataset: validation dataset
        :param criterion: loss function
        :return: average loss for the dataset
        """
        self.face_alignment_net.eval()
        loss_meter = AverageMeter()
        mse_meter = AverageMeter()
        data_loader = DataLoader(dataset, batch_size=10)
        for batch in data_loader:
            images = batch[0]
            target_landmarks = batch[1].reshape(-1, 68, 2)
            # Calc bounding box. Here unlike training, I don't add noise to the bounding boxes
            bb = self.create_bounding_box(target_landmarks, 0.05)  # 0.4 is a magic number which can be experimented with
            # Crop images
            inp, centers, scales = self.crop_batch(bb, images)
            inp = inp.permute((0, 3, 1, 2)).float()

            inp = inp.to(self.device)
            inp.div_(255.0)

            output = self.face_alignment_net(inp)[-1] # why [-1]?
            #pts, pts_img = get_preds_fromhm(output, centers, scales)
            pts, pts_img = get_preds_fromhm_old(output, centers, scales)
            # pts, pts_img = self.get_preds_fromhm_subpixel(output, centers, scales)
            # pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
            normalized_pixel_mse, pixel_mse = self.calc_landmarks_MSE(target_landmarks, pts_img)
            mse_meter.update(pixel_mse)
            # creating target heatmaps by drawing gaussians on landmark positions. Each landmark has its own heatmap
            target_heatmaps = self.create_target_heatmap(target_landmarks, centers, scales)
            loss = criterion(output, target_heatmaps.to(self.device))
            loss_meter.update(loss.item())

        return loss_meter.avg, mse_meter.avg


    @staticmethod
    def remove_models(self):
        base_path = os.path.join(appdata_dir('face_alignment'), "data")
        for data_model in os.listdir(base_path):
            file_path = os.path.join(base_path, data_model)
            try:
                if os.path.isfile(file_path):
                    print('Removing ' + data_model + ' ...')
                    os.unlink(file_path)
            except Exception as e:
                print(e)

    @staticmethod
    def get_preds_fromhm_subpixel(hm, center=None, scale=None):
        """Similar to `get_preds_fromhm` Except it tries to estimate the coordinates of the mode
           of the distribution.
        """
        max, idx = torch.max(hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
        preds[..., 0] = (preds[..., 0]) % hm.size(3)
        preds[..., 1].div_(hm.size(2)).floor_()
        eps = torch.tensor(0.0000000001).to(hm.device)
        # This is a magic number as far as understand.
        # 0.545 reduces the quantization error to exactly zero when `scale` is ~1.
        # 0.555 reduces the quantization error to exactly zero when `scale` is ~3.
        # 0.560 reduces the quantization error to exactly zero when `scale` is ~4.
        # 0.565 reduces the quantization error to exactly zero when `scale` is ~5.
        # 0.580 reduces the quantization error to exactly zero when `scale` is ~10.
        # 0.5825 reduces the quantization error to <0.002RMSE  when `scale` is ~100.
        sigma = 0.55
        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
                x0 = pX
                y0 = pY
                p0 = torch.max(hm_[pY, pX], eps)
                if pX < 63:
                    p1 = torch.max(hm_[pY, pX + 1], eps)
                    x1 = x0 + 1
                    y1 = y0
                    x = (3 * sigma)**2 * (torch.log(p1) - torch.log(p0)) - (
                                x0**2 - x1**2 + y0**2 - y1**2) / 2
                    if pY < 63:
                        p2 = torch.max(hm_[pY + 1, pX], eps)
                        x2 = x0
                        y2 = y0 + 1
                        y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                    x0**2 - x2**2 + y0**2 - y2**2) / 2
                    else:
                        p2 = torch.max(hm_[pY - 1, pX], eps)
                        x2 = x0
                        y2 = y1 - 1
                        y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                    x0**2 - x2**2 + y0**2 - y2**2) / 2
                else:
                    p1 = torch.max(hm_[pY, pX - 1], eps)
                    x1 = x0 - 1
                    y1 = y0
                    x = (3 * sigma)**2 * (torch.log(p1) - torch.log(p0)) - (
                                x0**2 - x1**2 + y0**2 - y1**2) / 2
                    if pY < 63:
                        p2 = torch.max(hm_[pY + 1, pX], eps)
                        x2 = x0
                        y2 = y0 + 1
                        y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                    x0**2 - x2**2 + y0**2 - y2**2) / 2
                    else:
                        p2 = torch.max(hm_[pY - 1, pX])
                        x2 = x0
                        y2 = y1 - 1
                        y = (3 * sigma)**2 * (torch.log(p2) - torch.log(p0)) - (
                                    x0**2 - x2**2 + y0**2 - y2**2) / 2
                preds[i, j, 0] = x
                preds[i, j, 1] = y
        preds_orig = torch.zeros(preds.size())
        if center is not None and scale is not None:
            for i in range(hm.size(0)):
                for j in range(hm.size(1)):
                    preds_orig[i, j] = transform(
                        preds[i, j]+0.5, center[i], scale[i], hm.size(2), True)
        return preds, preds_orig
