import numpy as np

def gazeto3d(gaze):#把二维的视线方向转换成三维 [yaw,pitch]
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[1]) * np.sin(gaze[0])
  gaze_gt[1] = -np.sin(gaze[1])
  gaze_gt[2] = -np.cos(gaze[1]) * np.cos(gaze[0])
  return gaze_gt

def gazeto3d_eth(gaze):#把二维的视线方向转换成三维 eth中[pitch,yaw]
  assert gaze.size == 2, "The size of gaze must be 2"
  gaze_gt = np.zeros([3])
  gaze_gt[0] = -np.cos(gaze[0]) * np.sin(gaze[1])
  gaze_gt[1] = -np.sin(gaze[0])
  gaze_gt[2] = -np.cos(gaze[0]) * np.cos(gaze[1])
  return gaze_gt

# 计算两个3D向量之间夹角的函数
def angular(gaze, label):
  assert gaze.size == 3, "The size of gaze must be 3"
  assert label.size == 3, "The size of label must be 3"

  total = np.sum(gaze * label)
  return np.arccos(min(total/(np.linalg.norm(gaze)* np.linalg.norm(label)), 0.9999999))*180/np.pi

def CropImg(img, X, Y, W, H):
    """
    X, Y is the corrdinate of the left-top corner of images. 
    W, H is weight and high.
    """

    Y_lim, X_lim  = img.shape[0], img.shape[1]
    H =  min(H, Y_lim)
    W = min(W, X_lim)

    X, Y, W, H = list(map(int, [X, Y, W, H]))
    X = max(X, 0)
    Y = max(Y, 0)

    if X + W > X_lim:
        X = X_lim - W

    if Y + H > Y_lim:
        Y = Y_lim - H

    return img[Y:(Y+H),X:(X+W)]
