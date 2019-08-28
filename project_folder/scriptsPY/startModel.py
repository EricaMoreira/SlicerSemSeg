###############################################################################
###############################################################################
####### The following code was written by:
#######
####### Erica Moreira
####### Maja Garbulinska
#######
###############################################################################
###############################################################################


# crop the image into a square with the dimention of the max possible number that is a multiple of 8
image = imread(X_train[0]).astype(np.float32)
mod = int(min(np.shape(image)[0], np.shape(image)[1])/8) * 8 

## define transformations of the data for training and validation
## do not random flip the validation data 
transformationsTrain = Transformations()
transformationsTrain.add(RandomCropTrain((mod,mod)))
transformationsTrain.add(RandomFlipLR(0.5))
transformationsValid = Transformations()
transformationsValid.add(CropValid((mod,mod)))
# transformations.add(my_transforms.RandomBrightness(prob=0.5, limit=(-0.15, 0.15)))

## define the number of classes 
n_classes = 3

## transform the training set and put it in the data loader.  
training_set = FromImageFilenames(X_train, y_train, n_classes, transformationsTrain)
train_loader = DataLoader(training_set, batch_size=1, shuffle=True)

## define test set only if there are test images available 

if "empty" in X_valid:
	print("No validation files to pass through the test data loader")
else: 
	## transform the test set and put it in the data loader.  
	test_set = FromImageFilenames(X_valid, y_valid, n_classes, transformationsValid)
	test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = UnetFinal()
model.zero_grad()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


if os.path.isfile(os.path.join(model_path, "model.pth")):
## load in saved weights if the model was run before. 
    model.load_state_dict(torch.load(os.path.join(model_path, "model.pth")))

if os.path.isfile(os.path.join(os.path.join(model_path, "optim.pth"))): 
    optimizer.load_state_dict(torch.load(os.path.join(model_path, "optim.pth")))

is_cuda_available = torch.cuda.is_available()
if is_cuda_available: model.cuda()

## weights 
wtperclass = torch.from_numpy(np.asarray([25.0,1.0,20.0],dtype=np.float32))
