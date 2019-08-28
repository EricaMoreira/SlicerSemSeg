###############################################################################
###############################################################################
####### The following code was written by:
####### 
####### Erica Moreira
####### Maja Garbulinska
####### 
####### Test the model on an unseen set of images
###############################################################################
###############################################################################

if "empty" in X_valid:
		print("There are no images available for testing")
else:
		test(model, test_loader, is_cuda_available, wtperclass)
		print("Testing finished")
