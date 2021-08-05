import torch
import torch.nn as nn

def doubleConv(in_channels, out_channels):
    
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
        
        nn.BatchNorm2d(out_channels),
        
        nn.ReLU(inplace = True),
        
        nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
        
        nn.BatchNorm2d(out_channels),
        
        nn.ReLU(inplace = True)
        )

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    
    delta = tensor_size - target_size
    delta = delta // 2
    
    if((tensor_size - target_size) % 2 == 0):
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]
    else:
        return tensor[:, :, delta:tensor_size - delta - 1, delta:tensor_size - delta - 1]
    



class UNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(UNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.MaxPool = nn.MaxPool2d(2, stride = 2)
        
        self.DownConvs1 = doubleConv(3, 64)
        self.DownConvs2 = doubleConv(64, 128)
        self.DownConvs3 = doubleConv(128, 256)
        self.DownConvs4 = doubleConv(256, 512)
        self.DownConvs5 = doubleConv(512, 1024)
        
        self.UpConvs1 = doubleConv(1024, 512)
        self.UpConvs2 = doubleConv(512, 256)
        self.UpConvs3 = doubleConv(256, 128)
        self.UpConvs4 = doubleConv(128, 64)
        
        self.UpTrans1 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
        self.UpTrans2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 2, stride = 2)
        self.UpTrans3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 2, stride = 2)
        self.UpTrans4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 2, stride = 2)
        
        self.OutConv = nn.Conv2d(64, n_classes, kernel_size = 1, stride = 1)
        
        
    def forward(self, x):
        x_1_1_conv = self.DownConvs1(x)
        x_1_1 = self.MaxPool(x_1_1_conv)
        
        x_1_2_conv = self.DownConvs2(x_1_1)
        x_1_2 = self.MaxPool(x_1_2_conv)
        
        x_1_3_conv = self.DownConvs3(x_1_2)
        x_1_3 = self.MaxPool(x_1_3_conv)
        
        x_1_4_conv = self.DownConvs4(x_1_3)
        x_1_4 = self.MaxPool(x_1_4_conv)
        
        x_1_5_conv = self.DownConvs5(x_1_4)
        
        # THE END OF CONTRACTING SIDE
        
        #START 1
        x_2_1 = self.UpTrans1(x_1_5_conv)
        y_1_4 = crop_img(x_1_4_conv, x_2_1)
        x_2_1 = torch.cat([x_2_1, y_1_4], dim = 1)
        
        x_2_1_conv = self.UpConvs1(x_2_1)
        #END 1
        #START 2
        x_2_2 = self.UpTrans2(x_2_1_conv)
        y_1_3 = crop_img(x_1_3_conv, x_2_2)
        x_2_2 = torch.cat([x_2_2, y_1_3], dim = 1)
        
        x_2_2_conv = self.UpConvs2(x_2_2)
        #END 2
        #START 3
        x_2_3 = self.UpTrans3(x_2_2_conv)
        y_1_2 = crop_img(x_1_2_conv, x_2_3)
        x_2_3 = torch.cat([x_2_3, y_1_2], dim = 1)
        
        x_2_3_conv = self.UpConvs3(x_2_3)
        #END 3
        #START 4
        x_2_4 = self.UpTrans4(x_2_3_conv)
        y_1_1 = crop_img(x_1_1_conv, x_2_4)
        x_2_4 = torch.cat([x_2_4, y_1_1], dim = 1)
       
        x_2_4_conv = self.UpConvs4(x_2_4)
        #END 4
       
        x = self.OutConv(x_2_4_conv)
        
        result = nn.Softmax(dim = 1)(x)
       
        return result
        
        
if __name__ == '__main__':
    dummy = torch.rand((1, 3, 224, 224))
    
    model = UNet((224,224), 2)
    
    output = model(dummy)
    print(output.size())
    
    