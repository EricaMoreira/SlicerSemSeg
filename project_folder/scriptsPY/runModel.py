###############################################################################
###############################################################################
#######
####### The following code was written by:
####### 
####### Erica Moreira
####### Maja Garbulinska
####### 
####### Train the model. You can go to the modelFunctions.py to see how the 
####### train function is defined. 
#######
###############################################################################
###############################################################################

for epoch in range(epochs):
    train(epoch, model, optimizer, train_loader, is_cuda_available, wtperclass)
    
torch.save(model.state_dict(), os.path.join(model_path, "model.pth"))
torch.save(optimizer.state_dict(), os.path.join(model_path, "optim.pth"))

print("Training done and model weights saved")
