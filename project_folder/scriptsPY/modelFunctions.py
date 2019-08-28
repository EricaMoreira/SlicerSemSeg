###############################################################################
###############################################################################
####### The following code was written by:
#######
####### Erica Moreira
####### Maja Garbulinska
####### 
####### Define functions that will be used by the model and in the rest
####### of the process 
#######
###############################################################################
###############################################################################



##############################################################################
##############################################################################
############################ NETWORK #########################################
##############################################################################
##############################################################################


# Set default weights as suggested in the Unet paper
def SetWeights(convlayer):
    fchannels = convlayer.in_channels
    std = np.sqrt(2.0/(9*fchannels))
    init.normal_(convlayer.weight.data, mean=0, std=std)
    return convlayer

class UNet(nn.Module):
    # number of channels input (CI) to the Unet and channels output (CO) by it.
    def __init__(self,CI=1,CO=3):
        super().__init__()

        class Downconv(nn.Module):
            def __init__(self,cinp,cout):
                super().__init__()

                self.ops = nn.Sequential()
                
                self.ops.add_module("mpool",nn.MaxPool2d(kernel_size=2, 
                                                         stride=2, 
                                                         padding=0, 
                                                         dilation=1, 
                                                         ceil_mode=False))
                
                self.ops.add_module("conv1",SetWeights(nn.Conv2d(cinp, 
                                                                 cout, 
                                                                 kernel_size=3, 
                                                                 stride=1, 
                                                                 padding=1)))
                
                self.ops.add_module("relu1",nn.ReLU(inplace=True))
                
                self.ops.add_module("conv2",SetWeights(nn.Conv2d(cout, 
                                                                 cout, 
                                                                 kernel_size=3, 
                                                                 stride=1, 
                                                                 padding=1)))
                
                self.ops.add_module("relu2",nn.ReLU(inplace=True))
                
            def forward(self,x):
                return self.ops(x)

        class Upconv(nn.Module):
            def __init__(self,cinp,cout):
                super().__init__()

                self.op1 = nn.Sequential()
                self.op2 = nn.Sequential()
                self.op3 = nn.Sequential()
                self.op1.add_module("dconv",
                                    nn.ConvTranspose2d(cinp,
                                                       cout, 
                                                       kernel_size=2, 
                                                       stride=2))
                # concat layer from before
                self.op2.add_module("conv1",
                                    SetWeights(nn.Conv2d(cinp, 
                                                         cout, 
                                                         kernel_size=3, 
                                                         stride=1, 
                                                         padding=1))) 
                
                self.op2.add_module("relu1",
                                    nn.ReLU(inplace=True))
                
                self.op3.add_module("conv2",
                                    SetWeights(nn.Conv2d(cout, 
                                                         cout, 
                                                         kernel_size=3, 
                                                         stride=1, 
                                                         padding=1)))
                
                self.op3.add_module("relu2",
                                    nn.ReLU(inplace=True))

            def forward(self,x,y):
                x = self.op1(x)
                x = self.op2(torch.cat((x,y),1))
                x = self.op3(x)
                return x

        self.conv1 = SetWeights(nn.Conv2d(CI, 
                                          32, 
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1))
        
        self.conv2 = SetWeights(nn.Conv2d(32, 
                                          32, 
                                          kernel_size=3, 
                                          stride=1, 
                                          padding=1))
        self.dn1 = Downconv(32,64)
        self.dn2 = Downconv(64,128)
        self.dn3 = Downconv(128,256)
        self.up1 = Upconv(256,128)
        self.up2 = Upconv(128,64)
        self.up3 = Upconv(64,32)
        self.lastconv = SetWeights(nn.Conv2d(32, 
                                             CO, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1))

        # self.lastconv4 = SetWeights(nn.Conv2d(128, CO, kernel_size=3, stride=1, padding=1))
        # self.lastconv2 = SetWeights(nn.Conv2d( 64, CO, kernel_size=3, stride=1, padding=1))
   
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x0 = F.relu(self.conv2(x))
        x1 = self.dn1(x0)
        x2 = self.dn2(x1)
        x3 = self.dn3(x2)
        x4 = self.up1(x3,x2)
        x5 = self.up2(x4,x1)
        x = self.up3(x5,x0)
        # no activation needed after lastconv if using F.CrossEntropyLoss since it
        # combines LogSoftmax() with NLLLoss()
        # print(x4.shape, x5.shape)
        return self.lastconv(x) #, self.lastconv2(F.relu(x5)), self.lastconv4(F.relu(x4))


class UnetFinal(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoderUnet = UNet(1,3)


    def forward(self,x):
        yy = self.encoderUnet(x) # label predictions are output
    
        return yy


##############################################################################
##############################################################################
############################ OTHER FUNCTIONS #################################
##############################################################################
##############################################################################



###############################################################################
#######    numpy2d_to_tensor
###############################################################################

def numpy2d_to_tensor(array, C=2):
    """ Converts a numpy array to Pytorch tensor.
        An image in a numpy array is usually stored as (H,W,C), 
        however, it is not always true especially if the array was created by the user.
        If C=0 you are telling this function that the Channels are in first position (C,H,W), 
        this allows to not transpose the array when creating the tensor!
    """
    if np.ndim(array) == 2:
        array = array[np.newaxis, :]
    elif np.ndim(array) == 3:
        if C==2:
            array = array.transpose(2,0,1)
    else:
        raise ValueError("Numpy array dimensionality not understood! (ndim={})".format(np.ndim(array)))
    return torch.from_numpy(array)

    
###############################################################################
#######    lossfunc
###############################################################################

# This function computes the loss as a sum of the cross-entropy loss in prediction
# of the labels and the MSE loss in reconstructing the original image.
# The two losses are relatively weighted using alpha (between 0 and 1).
#
# NOTE: F.cross_entropy function takes as input:
# - predictions of shape (batchsize,channels,height,widht)
# - groundtruth or target of shape (batchsize,height,widht), where each value is the class label from 0 to numclasses-1
# - weightperclass of shape (numclasses), i.e one weight per class

def lossfunc(pred, labels, img, wtperclass, ignore_index=-100):
    output = F.cross_entropy(pred,labels,ignore_index = ignore_index)


    return output
    
###############################################################################
#######    save_example_results
###############################################################################

def save_example_results(model, X_valid, y_valid, cuda_available):
    # resultpath = '/home/achanta/pytorching/unet/unet_microia/results/'
    resultpath = '/users/achanta/rktemp/'


    ind = np.random.randint(0,X_valid.shape[0],(1,1)).squeeze()

    count = 0
    for fimage, fmask in zip(X_valid, y_valid):
        if True:#count == ind: #0 == count%10:
     
            img = imread(fimage).astype(np.float32)/255.0
            mask =imread( fmask).astype(np.int64) - 1
            img = img[2:458, 2:458] # change in dimension done to adjust to maxpool operations of unet
            mask = mask[2:458, 2:458] # change in dimension done to adjust to maxpool operations of unet
            h = img.shape[0]
            w = img.shape[1]
            inpimg =  torch.from_numpy(img.reshape((1,1,h,w)))

            if cuda_available:
                inpimg = inpimg.cuda()
            labels,decimg = model(inpimg)
            if cuda_available:
                labels = np.uint8(np.argmax(labels.cpu().detach().numpy(), 1))
                decimg = np.uint8(decimg.cpu().detach().numpy())
            else:
                labels = np.uint8(np.argmax(labels.detach(), 1))
                decimg = np.uint8(decimg.detach())

            img = np.uint8(img*255.0)
            mask = np.uint8(mask)
            
            comb_img = np.dstack([img]*3)
            comb_mask1 = np.dstack([mask.squeeze()]*3)*80
            comb_mask2 = np.dstack([labels.squeeze()]*3)*80
            #comb_mask3 = np.dstack([out, mask, mask])
            dec_image = np.dstack([decimg.squeeze()]*3)
            
            # big = np.hstack([comb_img.copy(), comb_mask1.copy(), comb_mask2.copy(), dec_image.copy()])
            big = np.hstack([comb_img, comb_mask1, comb_mask2, dec_image])
            savepath = resultpath+fimage.split('/')[-1].split('.')[0] + ".png"
            imwrite(savepath,big)
            

        count += 1
        if(count >= 3):
            break
        
    print("results saved") 
    
    
###############################################################################
#######    FromImageFilenames
###############################################################################


class FromImageFilenames(data.Dataset):
    def __init__(self, images, labels, n_classes=3, transformations=None):
        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.transf = transformations
        self.len = len(images)
        
    def __getitem__(self, index):
        
        # get images
        image = imread(self.images[index]).astype(np.float32)/255.0
        ## labels are saved as integers starting from 1 to n. 
        ## the model accepts labels that start from 0 to n 
        ## unlabeled pixels should be -100 for the loss function 
        ## to ignore them. 
        label = (imread(self.labels[index])).astype(np.int64) - 1 
        label = np.where(label==-1, -100, label)
        
        if np.ndim(label) != 2:
            raise ValueError("Labels array must have two dimensions only! (ndim={})".format(np.ndim(labels)))

        # apply transformations
        if self.transf is not None:
            image, label = self.transf(image, label)

        # create the target volume for the cross-entropy loss
        binning = np.arange(self.n_classes)         
        # labels = np.dstack([(labels==b)*1 for b in binning]).astype(np.int64)
  
        # convert numpy arrays to tensors
        image =  numpy2d_to_tensor(image)
        label = numpy2d_to_tensor(label).squeeze()
        # labels = np.expand_dims(labels,axis=0)

        return image,label


    def __len__(self):
        return self.len
        
        
###############################################################################
#######    DataLoader
###############################################################################

class DataLoader(data.DataLoader):
    def __init__(self,  *args, **kwargs):
        super(DataLoader, self).__init__(*args, **kwargs)
        self.iterator = self.__iter__()
        
    def __next__(self):
        try:
            return next(self.iterator)
        except:
            self.iterator = self.__iter__()
            return next(self.iterator)
            
            
###############################################################################
#######    train
###############################################################################

# call model.train() prior to calling the train() function
def train(epoch, model, optimizer, train_loader, cuda_available, wtperclass):

    lwt0,lwt2,lwt4 = 1.0,1.0,1.0

    for batch_idx, (image, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        pred0 = model(image) # model outputs label predictions and decoded/reconstucted image
        
        # saving predictions as png while training is in place to see what is going on
        #length=len(pred0.max(1)[1])
        #for i in range(0, length):
            #pic=pred0.max(1)[1][i].numpy()
            #im = Image.fromarray(pic.astype('uint8'))   
            #plt.clf()
            #plt.imshow(im)
            #string  = "example" + str(i) + ".png"
            #plt.savefig(os.path.join(results_path, string))
        

        if cuda_available:
            image, labels = image.cuda(), labels.cuda()
        
        loss0 = lossfunc(pred0, labels, image, wtperclass = wtperclass)

        loss0.backward()
        optimizer.step()
        
        # maja changed what is displayed because it did not work 
        #len(train_loader.dataset) as oppposed to len(data)
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(train_loader.dataset), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss0.item()))

    return

###############################################################################
#######    test
###############################################################################

## the computation below should be authomatic plus what is really the size of the dataloader?!
  
def test(model, test_loader, cuda_available, wtperclass):
    lwt0, lwt2,lwt4 = 1.0,1.0,1.0
    test_loss = 0
    correct = 0
    
    for image, labels in test_loader:
        if cuda_available:
            image, labels = image.cuda(), labels.cuda()
        pred0 = model(image)
        loss0 = lossfunc(pred0, labels, image, wtperclass)
        
        test_loss = test_loss + loss0
        pred = pred0.max(0)[1] # get the index of the max log-probability

        correct = correct + pred.eq(labels).sum()
        
        dim1=np.shape(pred)[1]
        dim2=np.shape(pred)[2]
        print("dimentions", dim1, dim2)


    test_loss = test_loss.item()
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct.item(), len(test_loader.dataset)*dim1*dim2,
        100. * correct / (len(test_loader.dataset)* dim2 * dim1)))
    
###############################################################################
#######    predict
###############################################################################

## the computation below should be authomatic plus what is really the size of the dataloader?!
  
def predict(model, X_train, ind, labelssave):
    j=0
    for i in ind: 
         img = imread(X_train[j]).astype(np.float32)/255.0
         img = img[2:458,2:458] # adjust to max pooling
         h = img.shape[0]
         w = img.shape[1]
         inpimg =  torch.from_numpy(img.reshape((1,1,h,w)))
         out = model(inpimg)[0] # take only the first element(prediction)
         out = out.max(0)[1] # get the index of the max log-probability
         ## adding one because slicer sees labels as 1 to n 
         out = out.numpy()+1 
         # pad to reverse img = img[2:458,2:458]
         out = np.pad(out, 2, mode='constant')
         # save labels with the correct shape
         form = '{}_{}'.format("label",str(i).zfill(4))
         pictureid = form+".tiff"
         savepath = os.path.join(labelssave,pictureid)
         imsave(savepath, out)
         
         form = '{}_{}'.format("label",str(i).zfill(4))
         pictureid = form+".nrrd"
         savepath = os.path.join(labelssave,pictureid)
         nrrd.write(savepath, out, index_order='C')
         j=j+1
    
        
###############################################################################
#######    Transformations
###############################################################################    
     
class Transformations(object):
    def __init__(self):
               
        self.transforms = []
            
    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask
        
    def add(self, transform):
        self.transforms.append(transform)

###############################################################################
#######    CenterCrop
###############################################################################    
          
        
#class CenterCrop(object):
#    def __init__(self, size_img, size_mask):
#        self.size_img = size_img
#        self.size_mask = size_mask
#      
#    @staticmethod
#    def crop_center(img, size):
#        if isinstance(img, PIL.Image.Image):
#            # image and mask must have the same size
#            w, h = img.size
#            th, tw = size
#            x1 = int(round((w - tw) / 2.))
#            y1 = int(round((h - th) / 2.))
#            return img.crop((x1, y1, x1 + tw, y1 + th))
#        elif isinstance(img, np.ndarray):
#            w, h = img.shape[1:]
#            th, tw = size
#            x1 = int(round((w - tw) / 2.))
#            y1 = int(round((h - th) / 2.))
#            return img[:,x1:x1 + tw, y1:y1 + th]
#
#    def __call__(self, img, mask):
#        return self.crop_center(img, self.size_img), self.crop_center(mask, self.size_mask)


###############################################################################
#######    RandomCrop
###############################################################################    
   
class RandomCropTrain(object):

    def __init__(self, size_img):
        self.size_img = size_img
        self.th, self.tw = size_img

    def __call__(self, img, mask):
        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        elif isinstance(img, np.ndarray):
            # numpy array must be (H,W,C)
            h, w = img.shape[:2]

        self.x1 = random.randint(0, w - self.tw)
        self.y1 = random.randint(0, h - self.th)

        if isinstance(img, PIL.Image.Image):
            img_crop = img.crop((self.x1, self.y1, self.x1 + self.tw, self.y1 + self.th))
            mask_crop = mask.crop((self.x1, self.y1, self.x1 + self.tw, self.y1 + self.th))   
        elif isinstance(img, np.ndarray):
            img_crop = img[self.y1:self.y1 + self.th, self.x1:self.x1 + self.tw].copy()
            mask_crop = mask[self.y1:self.y1 + self.th, self.x1:self.x1 + self.tw].copy()  
        return img_crop, mask_crop
    

###############################################################################
#######    RandomCrop
###############################################################################    
   
class CropValid(object):

    def __init__(self, size_img):
        self.size_img = size_img
        self.th, self.tw = size_img

    def __call__(self, img, mask):
        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        elif isinstance(img, np.ndarray):
            # numpy array must be (H,W,C)
            h, w = img.shape[:2]

        self.x1 = int(np.round((w - self.tw)/2))
        self.y1 = int(np.round((h - self.th)/2))
    

        if isinstance(img, PIL.Image.Image):
            img_crop = img.crop((self.x1, self.y1, self.x1 + self.tw, self.y1 + self.th))
            mask_crop = mask.crop((self.x1, self.y1, self.x1 + self.tw, self.y1 + self.th))   
        elif isinstance(img, np.ndarray):
            img_crop = img[self.y1:self.y1 + self.th, self.x1:self.x1 + self.tw].copy()
            mask_crop = mask[self.y1:self.y1 + self.th, self.x1:self.x1 + self.tw].copy()  
        return img_crop, mask_crop
        
        
        
###############################################################################
#######    RandomCropAroundPosition
###############################################################################    

# The assumption is that patch size is much smaller than image size
# This function needs to first find label blob centers. Then it needs
# to pick one of the blob centers randomly and then pick one of the
# positions randomly.
class RandomCropAroundPosition(object): # RK added this

    def __init__(self, size_img, cx, cy, blobwd=112, patchwd=456):
        self.size_img = size_img
        self.th, self.tw = size_img
        self.cx = cx
        self.cy = cy
        self.p  = patchwd
        self.b  = blobwd

    def __call__(self, img, mask):
        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        elif isinstance(img, np.ndarray):
            # numpy array must be (H,W,C)
            h, w = img.shape[:2]

        shift = int(self.p/2 - self.b/2)
        xlo, ylo = 0, 0
        xhi, yhi = self.tw, self.th

        if cx < self.tw - cx: # blob is closer to left border
            xlo = max(cx - self.p/2 - shift, 0)
        else:
            xhi = min(cx - self.p/2 + shift, self.tw-self.p)

        if cy < self.th - cy: # blob is closer to top border
            ylo = max(cy - self.p/2 - shift, 0)
        else:
            yhi = min(cy - self.p/2 + shift, self.th-self.p)

        self.x1 = random.randint(xlo, xhi)
        self.y1 = random.randint(ylo, yhi)


        if isinstance(img, PIL.Image.Image):
            img_crop = img.crop((self.x1, self.y1, self.x1 + self.tw, self.y1 + self.th))
            mask_crop = mask.crop((self.x1, self.y1, self.x1 + self.tw, self.y1 + self.th))   
        elif isinstance(img, np.ndarray):
            img_crop = img[self.y1:self.y1 + self.th, self.x1:self.x1 + self.tw].copy()
            mask_crop = mask[self.y1:self.y1 + self.th, self.x1:self.x1 + self.tw].copy()  
        return img_crop, mask_crop

###############################################################################
#######    RandomFlipLR
###############################################################################    

class RandomFlipLR(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask):
        
        if random.random() > self.prob:
          
            if isinstance(img, PIL.Image.Image):
                pass   
            elif isinstance(img, np.ndarray):
                img_flip = np.fliplr(img).copy()
                mask_flip = np.fliplr(mask).copy()
                    
            return img_flip, mask_flip 
            
        else:
            return img, mask
            
###############################################################################
#######    RandomBrightness
###############################################################################    
 
class RandomBrightness(object):

    def __init__(self, prob=0.5, limit=(-0.15, 0.15)):
        self.prob = prob
        self.limit = limit
        
    def random_brightness(self, img):
        alpha = 1.0 - np.random.uniform(self.limit[0], self.limit[1])
        img = alpha * img
        return np.clip(img, 0., 1.)

    def __call__(self, img, mask):
        
        if random.random() > self.prob:
          
            if isinstance(img, PIL.Image.Image):
                pass   
            elif isinstance(img, np.ndarray):
                img_bright = self.random_brightness(img)
                    
            return img_bright, mask            
        else:
            return img, mask
                    
        
###############################################################################
#######    TrainTestSplit
###############################################################################    
 
def TrainTestSplit(labelpath, seed=None, numbertrainingimg=3, 
                   numbervalidimg=None, X_filenames=X_filenames, 
                   z_filenames=z_filenames, training_path=training_path):
    
    print("")
    print("")
    print("Temporary labels path:", labelpath)
    print("")
    print("")
    
    # setting seed if provided
    if seed is not None:
        random.seed(seed)
        
    indeces = sorted(random.sample(range(0, np.shape(X_filenames)[0]), numbertrainingimg))
    
    # when starting the labeling labels are not available.
    # creating "unlabeled" label files for the easy of setting correct filepaths
    
    # delete if the folder exists 
    if os.path.exists(labelpath):
        shutil.rmtree(labelpath)
        
    # create it again 
    os.makedirs(labelpath)
    
    # creating a matrix of zeros that is the size of the original images
    shape = np.shape(imread(X_filenames[0]))
    h = shape[0]
    w = shape[1]
    matrix = np.zeros((h, w))
    
    # making fake labels to have something in the folder
    for i in indeces:
        form = '{}_{}'.format("label",str(i).zfill(4))
        string=form+".tiff"
        labelpathtiff=os.path.join(labelpath, string)
        imsave(labelpathtiff, matrix) 
    
    pattern=os.path.join(labelpath, "*.tif*")
    y_filenames = np.array(sorted(glob(pattern)))
    
    ######## define the training and the testing set
    X_train = X_filenames[indeces]
    y_train = y_filenames

    ######## if there are files for validation
    # cases: 
    # 1. there are files in z_filenames and numbervalidimg is none
    # will take the validation files to be all the files not used in training 
    # 2. numbervalidimg is not none 
    # select at random numbervalidimg images with indeces from the ones that are different.
    # 3. we do not have images available for validation 
    if len(z_filenames)!=0:
        print("Detected files for validation")
        range2 = range(0, np.shape(X_filenames)[0])
        if numbervalidimg is None:
            # will get the files from index startvalid to the end unless you insert stopvalid here
            X_valid = np.delete(X_filenames, indeces) 
            y_valid = np.delete(z_filenames, indeces) 
            indecesvalid = [x for x in range2 if x not in indeces]

        if numbervalidimg is not None:
            indecesvalid = [x for x in range2 if x not in indeces]
            # choose indeces for validation but those that are not the same as the training onces
            indecesvalid = sorted(random.sample(indecesvalid, numbervalidimg))
            print("")
            print("The following indeces will be used for testing")
            print("")
            print(indecesvalid)
            X_valid = X_filenames[indecesvalid]
            y_valid = z_filenames[indecesvalid] 
    else:
        print("There are no files to be used in validation")
        X_valid = "empty"
        y_valid = "empty"
        indecesvalid = "empty"
        
    
    ## creating a folder for training data original images
    ## if it already exist delete it first with all its content
    if os.path.exists(training_path):
        shutil.rmtree(training_path)
        
    os.makedirs(training_path)
    
    for files in X_train:
        shutil.copy2(files, training_path) 
    
    ## change X_train to the files from the new folder. 
    X_train = np.array(sorted(glob(os.path.join(training_path, "*.*"))))
    
    
    print("")
    print("The following indeces will be used for training")
    print("")
    print(indeces)
    
    return(X_train, y_train, X_valid, y_valid, indeces, indecesvalid)
    
        
        
    
        
    