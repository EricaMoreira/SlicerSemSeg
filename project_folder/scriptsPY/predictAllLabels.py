###############################################################################
###############################################################################
####### The following code was written by:
####### 
####### Erica Moreira
####### Maja Garbulinska
####### 
####### Predict labels for all the images in the original images folder.
####### 
###############################################################################
###############################################################################

indecespred = list(range(0, np.shape(X_filenames)[0]))

# predict and save to the predictions folder in results. 
predict(model, 
        X_filenames, 
        ind=indecespred, 
        labelssave=os.path.join(results_path, "predictions"))

