import torch   #查看torch版本
import torch.nn.functional as F
print(torch.__version__)  #注意是双下划线

#import torchvision  #查看torchvision版本
#print(torchvision.__version__)  #注意是双下划线

print(torch.version.cuda)  #注意是双下划线

print(torch.cuda.is_available())#如果cuda可用，则返回True