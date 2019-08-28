##############################################################################
##############################################################################
####### The following code was written by:
####### 
####### Erica Moreira
####### Maja Garbulinska
#######
#######
####### Saving the label
#######
##############################################################################
##############################################################################

#########################
# Saving the label
#########################
# Solution from : https://discourse.slicer.org/t/segment-binarylabelmap-to-numpy-array/778
    
# Create a binary label volume from segmentation
labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    
# To export specific segments instead, edit the following line with another one listed in above solution link
slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, 
labelmapVolumeNode, referenceVolumeNode)
    
referenceVolumeNode=slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')

# Export data as numpy arrays
referenceImg = arrayFromVolume(referenceVolumeNode)
label = arrayFromVolume(labelmapVolumeNode) 


# Checking what the mask looks like
print('Unique values in the label: {0}'.format(np.unique(label)))
print('\n Labels shape: {0}'.format(np.shape(label)))
print('\n Original image shape: {0}'.format(np.shape(referenceImg)))
    
# this will be the output of the initial annotation, and the input to the model
## save as tiff and nrrd

j=0
for i in indeces:
    form = '{}_{}'.format("label",str(i).zfill(4))
    string = form + ".tiff"
    labelpathtiff = os.path.join(labels_path, string)
    pictureid = form + ".nrrd"
    labelpathnrrd = os.path.join(labels_path, pictureid)
    currentlabel = label[j]
    imsave(labelpathtiff, currentlabel.astype(np.uint8))
    nrrd.write(labelpathnrrd, currentlabel, index_order='C')
    j=j+1
    
    

    
    
