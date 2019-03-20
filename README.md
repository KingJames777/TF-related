# TF-related

CNN with structures like:
[ conv - bn - relu - pool ] * 2 - conv - bn - relu - affine - relu - affine - relu - affine - softmax
Previous version seemingly has the ceiling of 0.5 at val data.
