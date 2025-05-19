"""
Implements a style transfer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torch.nn as nn
from a6_helper import *

def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from style_transfer.py!')

def content_loss(content_weight, content_current, content_original):
    """
    Compute the content loss for style transfer.
    
    Inputs:
    - content_weight: Scalar giving the weighting for the content loss.
    - content_current: features of the current image; this is a PyTorch Tensor of shape
      (1, C_l, H_l, W_l).
    - content_original: features of the content image, Tensor with shape (1, C_l, H_l, W_l).

    Returns:
    - scalar content loss
    """
    ############################################################################
    # TODO: Compute the content loss for style transfer.                       #
    ############################################################################
    # Replace "pass" statement with your code
    loss = content_weight * torch.sum((content_current - content_original) ** 2)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    
    Inputs:
    - features: PyTorch Tensor of shape (N, C, H, W) giving features for
      a batch of N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, C, C) giving the
      (optionally normalized) Gram matrices for the N input images.
    """
    gram = None
    ############################################################################
    # TODO: Compute the Gram matrix from features.                             #
    # Don't forget to implement for both normalized and non-normalized version #
    ############################################################################
    # Replace "pass" statement with your code
    N, C, H, W = features.shape
    reshaped_features = features.view(N, C, H * W)
    gram = torch.bmm(reshaped_features, reshaped_features.transpose(1, 2))
    
    if normalize:
        denominator = C * H * W
        if denominator > 0:
            gram = gram / denominator
        else:
            gram = torch.zeros_like(gram) 
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return gram


def style_loss(feats, style_layers, style_targets, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the style loss at a set of layers.                        #
    # Hint: you can do this with one for loop over the style layers, and       #
    # should not be very much code (~5 lines).                                 #
    # You will need to use your gram_matrix function.                          #
    ############################################################################
    # Replace "pass" statement with your code
    s_loss = torch.tensor(0.0, device=feats[0].device, dtype=feats[0].dtype)

    for i in range(len(style_layers)):
        layer_idx = style_layers[i]
        current_feat = feats[layer_idx]
        target_gram = style_targets[i]
        weight = style_weights[i]
        
        gram_current = gram_matrix(current_feat, normalize=True)
        
        layer_loss = weight * torch.sum((gram_current - target_gram) ** 2)
        s_loss = s_loss + layer_loss
    
    return s_loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    ############################################################################
    # TODO: Compute total variation loss.                                      #
    # Your implementation should be vectorized and not require any loops!      #
    ############################################################################
    # Replace "pass" statement with your code
    # 水平方向像素差的平方和
    h_variance = torch.sum(torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2))
    
    # 垂直方向像素差的平方和
    w_variance = torch.sum(torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2))
    
    loss = tv_weight * (h_variance + w_variance)
    return loss
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################


def guided_gram_matrix(features, masks, normalize=True):
  """
  Inputs:
    - features: PyTorch Tensor of shape (N, R, C, H, W) giving features for
      a batch of N images.
    - masks: PyTorch Tensor of shape (N, R, H, W)
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
    
    Returns:
    - gram: PyTorch Tensor of shape (N, R, C, C) giving the
      (optionally normalized) guided Gram matrices for the N input images.
  """
  guided_gram = None
  ##############################################################################
  # TODO: Compute the guided Gram matrix from features.                        #
  # Apply the regional guidance mask to its corresponding feature and          #
  # calculate the Gram Matrix. You are allowed to use one for-loop in          #
  # this problem.                                                              #
  ##############################################################################
  # Replace "pass" statement with your code
  N_batch, R_regions, C_channels, H_dim, W_dim = features.shape

  if R_regions == 0:
      guided_gram = torch.empty((N_batch, 0, C_channels, C_channels), 
                                device=features.device, dtype=features.dtype)
  else:
      all_regional_grams = []
      for r_idx in range(R_regions):
          features_for_region = features[:, r_idx, :, :, :]
          mask_for_region = masks[:, r_idx, :, :]
          
          mask_for_region_expanded = mask_for_region.unsqueeze(1)
          
          masked_features_r = features_for_region * mask_for_region_expanded
          
          reshaped_masked_features_r = masked_features_r.view(N_batch, C_channels, H_dim * W_dim)
          gram_r = torch.bmm(reshaped_masked_features_r, reshaped_masked_features_r.transpose(1, 2))
          
          if normalize:
              denominator = C_channels * H_dim * W_dim
              if denominator > 0:
                  gram_r = gram_r / denominator
              else:
                  gram_r = torch.zeros_like(gram_r)
          
          all_regional_grams.append(gram_r)
      
      guided_gram = torch.stack(all_regional_grams, dim=1)
  
  return guided_gram
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################


def guided_style_loss(feats, style_layers, style_targets, style_weights, content_masks):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image, as produced by
      the extract_features function.
    - style_layers: List of layer indices into feats giving the layers to include in the
      style loss.
    - style_targets: List of the same length as style_layers, where style_targets[i] is
      a PyTorch Tensor giving the guided Gram matrix of the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_layers, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
    - content_masks: List of the same length as feats, giving a binary mask to the
      features of each layer.
      
    Returns:
    - style_loss: A PyTorch Tensor holding a scalar giving the style loss.
    """
    ############################################################################
    # TODO: Computes the guided style loss at a set of layers.                 #
    ############################################################################
    # Replace "pass" statement with your code
    gsl = torch.tensor(0.0, device=feats[0].device, dtype=feats[0].dtype)

    for k_idx in range(len(style_layers)):
        layer_index = style_layers[k_idx]
        
        current_feat_map = feats[layer_index]
        current_region_masks = content_masks[layer_index]
        target_guided_gram_k = style_targets[k_idx]
        style_weight_k = style_weights[k_idx]
        
        calculated_guided_gram = guided_gram_matrix(current_feat_map, 
                                                    current_region_masks, 
                                                    normalize=True)
        
        layer_gsl = style_weight_k * torch.sum((calculated_guided_gram - target_guided_gram_k) ** 2)
        
        gsl = gsl + layer_gsl
        
    return gsl
    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
