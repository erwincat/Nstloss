import tensorflow as tf 
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
import argparse
import logging
import os.path



#---------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument

add_arg('--content',        default=None, type=str,         help='Content image path as optimization target.')
add_arg('--content-weight', default=10, type=float,       help='Weight of content relative to style.')
add_arg('--content-layers', default='block4_conv2', type=str,        help='The layer with which to match content.')
add_arg('--style',          default=None, type=str,         help='Style image path to extract patches.')
add_arg('--style-weight',   default=25.0, type=float,       help='Weight of style relative to content.')
add_arg('--style-layers',   default='block3_conv2,block4_conv2', type=str,    help='The layers to match style patches.')
add_arg('--semantic-ext',   default='_sem.png', type=str,   help='File extension for the semantic maps.')
add_arg('--semantic-weight', default=10.0, type=float,      help='Global weight of semantics vs. features.')
add_arg('--output',         default='output.png', type=str, help='Output image path to save once done.')
add_arg('--output-size',    default='512,512', type=str,         help='Size of the output image, e.g. 512x512.')
add_arg('--phases',         default=3, type=int,            help='Number of image scales to process in phases.')
add_arg('--slices',         default=2, type=int,            help='Split patches up into this number of batches.')
add_arg('--cache',          default=0, type=int,            help='Whether to compute matches only once.')
add_arg('--smoothness',     default=1E+0, type=float,       help='Weight of image smoothing scheme.')
add_arg('--variety',        default=0.0, type=float,        help='Bias toward selecting diverse patches, e.g. 0.5.')
add_arg('--seed',           default='noise', type=str,      help='Seed image path, "noise" or "content".')
add_arg('--seed-range',     default='16:240', type=str,     help='Random colors chosen in range, e.g. 0:255.')
add_arg('--iterations',     default=100, type=int,          help='Number of iterations to run each resolution.')
add_arg('--device',         default='cpu', type=str,        help='Index of the GPU number to use, for theano.')
add_arg('--print-every',    default=10, type=int,           help='How often to log statistics to stdout.')
add_arg('--save-every',     default=10, type=int,           help='How frequently to save PNG into `frames`.')
args = parser.parse_args()
#---------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------------------


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

# 内容层将提取出我们的 feature maps （特征图）
# content_layers = ['block4_conv2'] 

# # 我们感兴趣的风格层
# style_layers = ['block2_conv2',
#                 'block3_conv2', 
#                 'block4_conv2']

# style_weight=500000
# content_weight=300000


#------------------------------------------------------------
#辅助函数
#------------------------------------------------------------

#加载图像，并将其最大尺寸限制为512像素
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float64)

  # shape = tf.cast(tf.shape(img)[:-1], tf.float64)
  # long_dim = max(shape)
  # scale = max_dim / long_dim

  # new_shape = tf.cast(shape * scale, tf.int32)

  # img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


#显示图像
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG 
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model



#------------------------------------------------------------
#gram矩阵
#------------------------------------------------------------
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class CSFlow:
    def __init__(self, sigma = float(0.1), b = float(1.0)):
        self.b = b
        self.sigma = sigma #sigma 是变量h

    def __calculate_CS(self, scaled_distances, axis_for_normalization = 3):
        self.scaled_distances = scaled_distances #修正后的余弦距离
        self.cs_weights_before_normalization = tf.exp((self.b - scaled_distances) / self.sigma, name='weights_before_normalization') #权重
        self.cs_NHWC = CSFlow.sum_normalize(self.cs_weights_before_normalization, axis_for_normalization)#CXij 集合

    def reversed_direction_CS(self):
        cs_flow_opposite = CSFlow(self.sigma, self.b)
        cs_flow_opposite.raw_distances = self.raw_distances
        work_axis = [1, 2]
        relative_dist = cs_flow_opposite.calc_relative_distances(axis=work_axis)
        cs_flow_opposite.__calculate_CS(relative_dist, work_axis)
        return cs_flow_opposite


    #-- 计算输入图和目标图特征的相关性，计算了CXij的集合
    @staticmethod
    def create_using_dotP(I_features, T_features, sigma = float(1.0), b = float(1.0)):
        cs_flow = CSFlow(sigma, b)
        # tf.print("create_using_dotP")
        with tf.name_scope('CS'):
            # prepare feature before calculating cosine distance
            T_features, I_features = cs_flow.center_by_T(T_features, I_features)
            with tf.name_scope('TFeatures'):
                patch_size = 3 #将图像切分为3*3的patch
                
                T_features = tf.image.extract_patches(
                    images=T_features, sizes=[1, patch_size, patch_size, 1],
                    strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding='VALID',
                    name='patches_as_depth_vectors')
                # tf.print("tfe:",tf.shape(T_features))
                T_features_norm = CSFlow.l2_normalize_channelwise(T_features)
            with tf.name_scope('IFeatures'):
                
                # work seperatly for each example in dim 1

                # tf.print("Tfeatures:",tf.shape(T_features))
                # tf.print("Ifeatures:",tf.shape(I_features))
                # cosine_dist_l = []
                # N, _, __, ___ = T_features.shape.as_list()
                


                # T_features_i = tf.expand_dims(T_features[i, :, :, :], 0)
                # I_features_i = tf.expand_dims(I_features[i, :, :, :], 0)


                patches_NHWC = tf.reshape(
                    T_features_norm,
                    shape=[-1, 1, 1, T_features_norm.shape[3]],
                    name='patches_PHWC')

                patches_HWCN = tf.transpose(
                    patches_NHWC,
                    perm=[1, 2, 3, 0],
                    name='patches_HWCP') 
                
                I_features = tf.image.extract_patches(
                    images=I_features, sizes=[1, 3, 3, 1],
                    strides=[1, 3, 3, 1], rates=[1, 1, 1, 1], padding='VALID',
                    name='I_patches_as_depth_vectors')
                I_features_norm = CSFlow.l2_normalize_channelwise(I_features)
                cs_flow.cosine_dist = tf.nn.conv2d(I_features_norm, patches_HWCN, strides=[1, 1, 1, 1],
                                                    padding='VALID', name='cosine_dist')

                cs_flow.patch_loss = cs_flow.match_patches(T_features,I_features,cs_flow.cosine_dist)
                # tf.print("cosine_dist:",tf.shape(cosine_dist))
                # cosine_dist_l.append(cosine_dist)

                # cs_flow.cosine_dist = tf.concat(cosine_dist_l, axis = 0)

                cosine_dist_zero_to_one = -(cs_flow.cosine_dist - 1) / 2
                cs_flow.raw_distances = cosine_dist_zero_to_one

                relative_dist = cs_flow.calc_relative_distances()
                cs_flow.__calculate_CS(relative_dist)
                return cs_flow

    def match_patches(self,T_features,I_features,cosine_dist):

        T_features = tf.reshape(T_features,shape = [-1,T_features.shape[3]])
        I_features = tf.reshape(I_features,shape = [-1,I_features.shape[3]])
        cosine_dist = tf.reshape(cosine_dist,shape = [-1,cosine_dist.shape[3]])
       
        match_idx = tf.argmax(cosine_dist,axis=1)


        match_list = tf.gather(T_features,match_idx)

        loss = tf.reduce_mean((I_features - match_list) ** 2)
        
        return loss

    def calc_relative_distances(self, axis=3):
        epsilon = 1e-5
        div = tf.math.reduce_min(self.raw_distances, axis=axis, keepdims=True)
        # div = tf.reduce_mean(self.raw_distances, axis=axis, keep_dims=True)
        relative_dist = self.raw_distances / (div + epsilon)
        return relative_dist

    def weighted_average_dist(self, axis = 3):
        if not hasattr(self, 'raw_distances'):
            raise exception('raw_distances property does not exists. cant calculate weighted average l2')

        multiply = self.raw_distances * self.cs_NHWC
        return tf.reduce_sum(multiply, axis=axis, name='weightedDistPerPatch')



    @staticmethod
    def sum_normalize(cs, axis=3): 
        reduce_sum = tf.math.reduce_sum(cs, axis, keepdims=True, name='sum')
        return tf.divide(cs, reduce_sum, name='sumNormalized') #计算CXij

    def center_by_T(self, T_features, I_features):
        # assuming both input are of the same size

        # calculate stas over [batch, height, width], expecting 1x1xDepth tensor
        axes = [0, 1, 2]
        self.meanT, self.varT = tf.nn.moments(
            T_features, axes, name='TFeatures/moments')

        # we do not divide by std since its causing the histogram
        # for the final cs to be very thin, so the NN weights
        # are not distinctive, giving similar values for all patches.
        # stdT = tf.sqrt(varT, "stdT")
        # correct places with std zero
        # stdT[tf.less(stdT, tf.constant(0.001))] = tf.constant(1)

        # TODO check broadcasting here
        with tf.name_scope('TFeatures/centering'):
            self.T_features_centered = T_features - self.meanT
        with tf.name_scope('IFeatures/centering'):
            self.I_features_centered = I_features - self.meanT

        return self.T_features_centered, self.I_features_centered
    @staticmethod

    def l2_normalize_channelwise(features):
        norms = tf.norm(features, ord='euclidean', axis=3, name='norm')
        # expanding the norms tensor to support broadcast division
        # tf.print("norm:",tf.shape(norms))
        norms_expanded = tf.expand_dims(norms, 3)
        # tf.print("norms_expanded:",tf.shape(norms_expanded))
        features = tf.divide(features, norms_expanded, name='normalized')
        return features

    def patch_decomposition(self, T_features):
        # patch decomposition
        # see https://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
        patch_size = 3 #将图像切分为3*3的patch
        tf.print("tfe:",tf.shape(T_features))
        patches_as_depth_vectors = tf.image.extract_patches(
            images=T_features, sizes=[1, patch_size, patch_size, 1],
            strides=[1, patch_size, patch_size, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')
        tf.print("patches_as_depth_vectors:",tf.shape(patches_as_depth_vectors))
        self.patches_NHWC = tf.reshape(
            patches_as_depth_vectors,
            shape=[-1, 1, 1, patches_as_depth_vectors.shape[3]],
            name='patches_PHWC')
        tf.print("patches_as_depth_vectors:",tf.shape(self.patches_NHWC))
        self.patches_HWCN = tf.transpose(
            self.patches_NHWC,
            perm=[1, 2, 3, 0],
            name='patches_HWCP')  # tf.conv2 ready format

        return self.patches_HWCN


    def do_extract_patches(self,features):
        patch_size = 3 #将图像切分为3*3的patch
        patches_as_depth_vectors = tf.image.extract_patches(
            images=features, sizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID',
            name='patches_as_depth_vectors')
        patches_list = tf.reshape(
            patches_as_depth_vectors,
            shape=[-1, patches_as_depth_vectors.shape[3]],
            name='patches_PHWC')
        norms = tf.norm(patches_list, ord='euclidean', axis=1, name='norm')
        return patches_list

#--------------------------------------------------
#     MRF help
#--------------------------------------------------
def mrf_loss(style_layer, generated_layer, patch_size=3, name=''):
    # type: (tf.Tensor, tf.Tensor, int, str) -> tf.Tensor
    """

    :param style_layer: The vgg feature layer by feeding it the style image.
    :param generated_layer: The vgg feature layer by feeding it the generated image.
    :param patch_size: The patch size of the mrf.
    :param name: Name scope of this loss.
    :return: the mrf loss between the two inputted layers represented as a scalar tensor.
    """

    # TODO: Maybe I should make style_layer static to improve speed.
    with tf.name_scope('mrf_loss' + name):
        generated_layer_patches = create_local_patches(generated_layer, patch_size)
        style_layer_patches = create_local_patches(style_layer, patch_size)
        generated_layer_nn_matched_patches = patch_matching(generated_layer_patches, style_layer_patches, patch_size)
        # _, height, width, number = map(lambda i: i.value, generated_layer.get_shape())
        _, height, width, number = generated_layer.shape.as_list()
        size = height * width * number
        # Normalize by the size of the image as well as the patch area.
        loss = tf.divide(tf.reduce_sum(tf.math.square(tf.math.subtract(generated_layer_patches, generated_layer_nn_matched_patches))),
                      size * (patch_size ** 2))
        return loss


def create_local_patches(layer, patch_size, padding='VALID'):
    # type: (tf.Tensor, int, str) -> tf.Tensor
    """

    :param layer: Feature layer tensor with dimension (1, height, width, feature)
    :param patch_size: The width and height of the patch. It is set to 3 in the paper https://arxiv.org/abs/1601.04589
    :param padding: a string representing the padding style.
    :return: Patches with dimension (cardinality, patch_size, patch_size, feature)
    """
    return tf.image.extract_patches(layer, sizes=[1, patch_size, patch_size, 1],
                                    strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding=padding)


def patch_matching(generated_layer_patches, style_layer_patches, patch_size):
    # type: (tf.Tensor, tf.Tensor, int) -> tf.Tensor
    """
    The patch matching is implemented as an additional convolutional layer for fast computation.
    In this case patches sampled from the style image are treated as the filters.
    :param generated_layer_patches: Size (batch, height, width, patch_size * patch_size * feature)
    :param style_layer_patches:Size (1, height, width, patch_size * patch_size * feature)
    :param patch_size: the patch size for mrf.
    :return: Best matching patch with size (batch, height, width, patch_size * patch_size * feature)
    """
    # Every patch and every feature layer are treated as equally important after normalization.
    normalized_generated_layer_patches = tf.math.l2_normalize(generated_layer_patches, axis=[3])
    normalized_style_layer_patches = tf.math.l2_normalize(style_layer_patches, axis=[3])
    # A better way to do this is to treat them as convolutions.
    # They have to be in dimension
    # (height * width, patch_size, patch_size, feature) <=> (batch, in_height, in_width, in_channels)
    # (patch_size, patch_size, feature, height * width) <= > (filter_height, filter_width, in_channels, out_channels)
    # Initially they are in [batch, out_rows, out_cols, patch_size * patch_size * depth]
    original_shape = normalized_style_layer_patches.get_shape().as_list()
    height = original_shape[1]
    width = original_shape[2]
    depth = tf.cast(original_shape[3] / patch_size / patch_size,tf.int32)
    # tf.print("depth",depth)
    normalized_style_layer_patches = tf.squeeze(normalized_style_layer_patches)

    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches,
                                                [height, width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.reshape(normalized_style_layer_patches,
                                                [height * width, patch_size, patch_size, depth])
    normalized_style_layer_patches = tf.transpose(normalized_style_layer_patches, perm=[1, 2, 3, 0])
    style_layer_patches_reshaped = tf.reshape(style_layer_patches, [height, width, patch_size, patch_size, depth])
    style_layer_patches_reshaped = tf.reshape(style_layer_patches_reshaped,
                                              [height * width, patch_size, patch_size, depth])

    normalized_generated_layer_patches_per_batch = tf.unstack(normalized_generated_layer_patches, axis=0)
    ret = []
    for batch in normalized_generated_layer_patches_per_batch:
        original_shape = batch.get_shape().as_list()
        height = original_shape[0]
        width = original_shape[1]
        depth = tf.cast(original_shape[2] / patch_size / patch_size,tf.int32)
        batch = tf.squeeze(batch)

        batch = tf.reshape(batch, [height, width, patch_size, patch_size, depth])
        batch = tf.reshape(batch, [height * width, patch_size, patch_size, depth])
        # According to images-analogies github, for cross-correlation, we should flip the kernels
        # That is normalized_style_layer_patches should be [:, ::-1, ::-1, :]
        # I didn't see that in any other source, nor do I see why I should do so.
        convs = tf.nn.conv2d(batch, normalized_style_layer_patches, strides=[1, 1, 1, 1], padding='VALID')
        argmax = tf.squeeze(tf.argmax(convs, axis=3))
        # best_match has shape [height * width, patch_size, patch_size, depth]
        best_match = tf.gather(style_layer_patches_reshaped, indices=argmax)
        best_match = tf.reshape(best_match, [height, width, patch_size, patch_size, depth])
        best_match = tf.reshape(best_match, [height, width, patch_size * patch_size * depth])
        ret.append(best_match)
    ret = tf.stack(ret)
    return ret

#--------------------------------------------------
#           CX loss
#--------------------------------------------------


def CX_loss(T_features, I_features, nnsigma=float(1.0)):
    T_features = tf.convert_to_tensor(T_features, dtype=tf.float32)
    I_features = tf.convert_to_tensor(I_features, dtype=tf.float32)

    with tf.name_scope('CX'):
        cs_flow = CSFlow.create_using_dotP(I_features, T_features, nnsigma, float(1.0))
        # sum_normalize:
        height_width_axis = [1, 2]
        # To:
        cs = cs_flow.cs_NHWC
        k_max_NC = tf.math.reduce_max(cs, axis=height_width_axis) #找到CXij集合中最大的CXij
        CS = tf.math.reduce_mean(k_max_NC, axis=[1]) #计算均值，CX(X,Y)，返回的是一个数组
        CX_as_loss = 1 - CS
        CX_loss = -tf.math.log(1 - CX_as_loss)
        CX_loss = tf.math.reduce_mean(CX_loss) #返回损失值，一个float32
        # print("CXLOSS：{}".format(float(CX_loss)))
        return cs_flow.patch_loss * (CX_loss + 1 )

#--------------------------------------------------
#           CX loss helper
#--------------------------------------------------
def random_sampling(tensor_NHWC, n, indices=None):
    N, H, W, C = tf.convert_to_tensor(tensor_NHWC).shape.as_list()
    S = H * W
    tensor_NSC = tf.reshape(tensor_NHWC, [N, S, C])
    all_indices = list(range(S))
    shuffled_indices = tf.random.shuffle(all_indices)
    indices = tf.gather(shuffled_indices, list(range(n)), axis=0) if indices is None else indices
    indices_old = tf.random_uniform([n], 0, S, tf.int32) if indices is None else indices
    res = tf.gather(tensor_NSC, indices, axis=1)
    return res, indices


def random_pooling(T_features, I_features, output_1d_size=100):
    # is_input_tensor = type(feats) is tf.Tensor

    # if is_input_tensor:
    #     feats = [feats]

    # # convert all inputs to tensors
    # feats = [tf.convert_to_tensor(feats_i) for feats_i in feats]

    N, T_H, T_W, C = T_features.shape.as_list()
    _, I_H, I_W, _ = I_features.shape.as_list()
    if T_H * T_W < I_H * I_W:
        T_feats_sampled, indices = random_sampling(T_features, output_1d_size ** 2)
        I_feats_sampled, _ = random_sampling(I_features, -1, indices)
    else:
        I_feats_sampled, indices = random_sampling(I_features, output_1d_size ** 2)
        T_feats_sampled, _ = random_sampling(T_features, -1, indices)
    # tf.print("rs end")
    res = [T_feats_sampled,I_feats_sampled]
    # for i in range(1, len(feats)):
    #     feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
    #     tf.print("rsp end")
    #     res.append(feats_sampled_i)

    res = [tf.reshape(feats_sampled_i, [N, output_1d_size, output_1d_size, C]) for feats_sampled_i in res]
    # if is_input_tensor:
    #     return res[0]
    return res


def crop_quarters(feature_tensor):
    N, fH, fW, fC = feature_tensor.shape.as_list()
    quarters_list = []
    quarter_size = [N, round(fH / 2), round(fW / 2), fC]
    quarters_list.append(tf.slice(feature_tensor, [0, 0, 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), 0, 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, 0, round(fW / 2), 0], quarter_size))
    quarters_list.append(tf.slice(feature_tensor, [0, round(fH / 2), round(fW / 2), 0], quarter_size))
    feature_tensor = tf.concat(quarters_list, axis=0)
    return feature_tensor


def CX_loss_helper(T_features, I_features,nnsigma=float(0.5)):
    # if CX_config.crop_quarters is True:
    #     T_features = crop_quarters(T_features)
    #     I_features = crop_quarters(I_features)
    # _, fH, fW, fC = T_features.shape.as_list()
    # fS = fH * fW
    
    # if fS <= 65 ** 2:
    #     print(' #### Skipping pooling for CX....')
    # else:
    #     T_features, I_features = random_pooling(T_features, I_features, output_1d_size=65)
        # fSzie = int(max(fH,fW) / 65)
        # max_features = tf.nn.max_pool(T_features,[1,fSzie,fSzie,1],[1,fSzie,fSzie,1],padding = 'VALID')
        # avg_features = tf.nn.avg_pool(T_features,[1,fSzie,fSzie,1],[1,fSzie,fSzie,1],padding = 'VALID')
        # T_features = 0.5 * (max_features + avg_features)


    # N, iH, iW, iC = I_features.shape.as_list()
    # iS = iH * iW
    
    # if iS <= 65 ** 2:
    #     print(' #### Skipping pooling for CX....')
    # else:
    #     iSzie = int(max(iH,iW) / 65)
    #     I_max_features = tf.nn.max_pool(I_features,[1,iSzie,iSzie,1],[1,iSzie,iSzie,1],padding = 'VALID')
    #     I_avg_features = tf.nn.avg_pool(I_features,[1,iSzie,iSzie,1],[1,iSzie,iSzie,1],padding = 'VALID')
    #     I_features = 0.5 * (I_max_features + I_avg_features)

    loss = CX_loss(T_features, I_features,nnsigma)
    # loss = mrf_loss(T_features,I_features)
    return loss

#------------------------------------------------------------
#Model
#------------------------------------------------------------
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    # style_outputs = [gram_matrix(style_output)
    #                  for style_output in style_outputs]  #使用了gram矩阵

    style_dict = {style_name:value 
                     for style_name, value 
                     in zip(self.style_layers, style_outputs)}

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    # style_dict = {style_name:value
    #               for style_name, value
    #               in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}

#------------------------------------------------------------
#run
#------------------------------------------------------------

def clip_0_1(image):
  	return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs,style_targets,content_targets):
    num_content_layers = len(args.content_layers.split(','))
    num_style_layers = len(args.style_layers.split(','))
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    # style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
    #                        for name in style_outputs.keys()])

    # tf.print("num_style_layers:",num_style_layers)
    #有待改进，重复
    style_loss = tf.add_n([CX_loss_helper(style_targets[name],style_outputs[name],float(0.2)) for name in style_outputs.keys()])
    # style_loss *= args.style_weight / num_style_layers
    style_loss *= args.style_weight / num_style_layers

    # tf.print("style_loss:",style_loss)
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])

    # print("content_outputs:{}".format(content_outputs.keys()))
    # print("content_targets:{}".format(content_targets.keys()))
    # content_loss = tf.add_n([CX_loss_helper(content_targets[name],content_outputs[name],float(0.1)) for name in content_outputs.keys()])
    content_loss *= args.content_weight / num_content_layers
    # tf.print("content_loss:",content_loss)
    loss = style_loss + content_loss
    return loss, style_loss, content_loss






def run(content_path,style_path):
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    extractor = StyleContentModel(args.style_layers.split(","), args.content_layers.split(","))
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # results = extractor(tf.constant(content_image))
    # style_results = results['style']

    outputSize=args.output_size.split(",")
    outputImage_height= outputSize[0]
    outputImage_wight = outputSize[1]

    image = tf.compat.v1.get_variable("outputImage",shape=([1,content_image.shape[1],content_image.shape[2],3]),dtype=tf.float64,initializer=tf.random_normal_initializer(mean=0,stddev=120)) #

    # image = tf.image.resize(image,[content_image.shape[1],content_image.shape[2]])
    # image = tf.Variable(content_image)
    opt = tf.optimizers.Adam(learning_rate=1, beta_1=0.9, epsilon=1e-1)

    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            loss = train_step(image,extractor,style_targets,content_targets,opt)
            print(".", end='')
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("Train step: {}".format(step))
        print("time: {:.1f}".format(time.time()-start))
        tensor_to_image(image).save("frame/test{}.png".format(n+1))
    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    tensor_to_image(image).save(args.output)

@tf.function()
def train_step(image,extractor,style_targets,content_targets,opt):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss,style_loss, content_loss = style_content_loss(outputs,style_targets,content_targets)

    # tf.compat.v1.logging.info("loss:%f",float(loss))
    tf.print("this function ",["loss:",loss / 1000.0,"style_loss:",style_loss / 1000.0,"content_loss:",content_loss / 1000.0])
    grad = tape.gradient(loss, image)
    # print("grad:{}".format(grad))
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss


if __name__ == "__main__":
	run(args.content,args.style)

  	




























