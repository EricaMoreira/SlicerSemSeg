###############################################################################
###############################################################################
####### The following code was written by:
####### 
####### Erica Moreira
####### Maja Garbulinska
####### 
#######
###############################################################################
###############################################################################


# clear the scene
slicer.mrmlScene.Clear(False) 
   
# this will load all images 
[success, volume] = slicer.util.loadVolume(filename = X_train[0], returnNode=True) 

# do not interporate between images! This is not a 3d case! 
def NoInterpolate(caller,event):
  for node in slicer.util.getNodes('*').values():
    if node.IsA('vtkMRMLScalarVolumeDisplayNode'):
      node.SetInterpolate(0)

slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeAddedEvent, NoInterpolate)
    
#Other properties for loadVolume function:
# loadVolume(filename, properties={}, returnNode=False)
#  - name: this name will be used as node name for the loaded volume
#  - labelmap: interpret volume as labelmap
#  - singleFile: ignore all other files in the directory
#  - center: ignore image position
#  - discardOrientation: ignore image axis directions
#  - autoWindowLevel: compute window/level automatically
#  - show: display volume in slice viewers after loading is completed
#  - fileNames: list of filenames to load the volume from
    
# set the layout view in the slicer GUI (in this case to the red/one image only view)
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
# set the initial module to Segment Editor (allowing us to paint and annotate the image)
slicer.util.mainWindow().moduleSelector().selectModule('SegmentEditor')
    
# Create segmentation
segmentationNode = slicer.vtkMRMLSegmentationNode()
slicer.mrmlScene.AddNode(segmentationNode)
segmentationNode.CreateDefaultDisplayNodes() # "only needed for display" (not sure what this means, but it is in example code)
segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(volume)
    
# Get the volume node
#referenceVolumeNode = slicer.util.getNode('Heart_Image_Sample')  # another way to do it
referenceVolumeNode=slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')
    
# Get the segmentation node
# segmentationNode = slicer.util.getNode('Segmentation') # another way to do it
segmentationNode=slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLSegmentationNode') 
    
# Create segment editor to get access to effects
#segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()  # This creates a separate window, so instead use the following
segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
# To show segment editor widget (supposedly useful for debugging): segmentEditorWidget.show()
segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
slicer.mrmlScene.AddNode(segmentEditorNode)
segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)
segmentEditorWidget.setMasterVolumeNode(referenceVolumeNode)
segmentEditorWidget.setActiveEffectByName("Paint")
    
# Add segmentation
segmentationNode.GetSegmentation().AddEmptySegment("Collagen")
s=getNode('vtkMRMLSegmentationNode1')
se=s.GetSegmentation()
seg=se.GetSegment('Collagen')
seg.SetColor(1,0,0) # To change the primary color of the segment (that is also saved to disk etc.)
    
# Add segmentation
segmentationNode.GetSegmentation().AddEmptySegment("Background")
s=getNode('vtkMRMLSegmentationNode1')
se=s.GetSegmentation()
seg=se.GetSegment("Background")
    
# Add segmentation
segmentationNode.GetSegmentation().AddEmptySegment("Cells")
s=getNode('vtkMRMLSegmentationNode1')
se=s.GetSegmentation()
seg=se.GetSegment("Cells")
    
if success == True:
    print("Image of size {0} loaded successfully".format(volume.GetImageData().GetDimensions()))
else:
    print("Image not loaded successfully, please quit Jupyter notebook and try again.")

## repeating this in saveLabels.py coz otherwise it doesnt work properly. Check what is wrong!             
# Create a binary label volume from segmentation
labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
    
# To export specific segments instead, edit the following line with another one listed in above solution link
slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(segmentationNode, 
labelmapVolumeNode, referenceVolumeNode)
referenceVolumeNode=slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLScalarVolumeNode')

# Export data as numpy arrays
referenceImg = arrayFromVolume(referenceVolumeNode)
label = arrayFromVolume(labelmapVolumeNode)
