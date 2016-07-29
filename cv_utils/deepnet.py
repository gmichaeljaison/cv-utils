import numpy as np

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


class DeepNet:
    """
    DeepNetwork model can be loaded and used to extract features from any layer for any input image.
    """

    def __init__(self, prototxt, model_path):
        self.net = caffe.Net(prototxt, model_path, caffe.TEST)

    def extract_feature(self, image, layer):
        """
        Passes image to a deepnet and extracts from the specified layer
        
        :param image: opencv image
        :param layer: network layer name
        :return: extracted features
        """

        img = image.astype(np.float32)
        # Substracting Image mean value
        img -= np.array((104.00698793, 116.66876762, 122.67891434))

        # make dims C x H x W for Caffe
        img = img.transpose((2, 0, 1))
        in_img = img[np.newaxis, :, :, :]

        self.net.blobs['data'].reshape(*in_img.shape)

        data_feat = self.net.forward_all(data=in_img, blobs=[layer])

        feature = data_feat[layer][0]

        # feature = (feature - feature.mean()) / feature.std()
        # feature /= np.linalg.norm(feature)
        return feature

