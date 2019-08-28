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

# load the original images
[success, volume] = slicer.util.loadVolume(filename = X_train[0], returnNode=True) 

# do not interporate between images! This is not a 3d dataset example
def NoInterpolate(caller,event):
  for node in slicer.util.getNodes('*').values():
    if node.IsA('vtkMRMLScalarVolumeDisplayNode'):
      node.SetInterpolate(0)
	
slicer.mrmlScene.AddObserver(slicer.mrmlScene.NodeAddedEvent, NoInterpolate)

# set the layout view in the slicer GUI (in this case to the red/one image only view)
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)

# set the initial module to Segment Editor (allowing us to paint and annotate the image)
slicer.util.mainWindow().moduleSelector().selectModule('SegmentEditor')

# load the labels (as an nrrd file)
# this will be the labels that have been updated by the predictions of the model
# and converted from numpy array back to nrrd


string = y_train[0].split('.')[0] + ".nrrd"
[success, labelmapVolumeNode] = slicer.util.loadLabelVolume(filename = string, returnNode=True)

# Import labelmap to segmentation:
segmentationNode = slicer.vtkMRMLSegmentationNode()
slicer.mrmlScene.AddNode(segmentationNode)
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)

# Create segment editor to get access to effects
#segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()  # This creates a separate window, so instead use the following
segmentEditorWidget = slicer.modules.segmenteditor.widgetRepresentation().self().editor
# To show segment editor widget (supposedly useful for debugging): segmentEditorWidget.show()
segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
segmentEditorNode = slicer.vtkMRMLSegmentEditorNode()
slicer.mrmlScene.AddNode(segmentEditorNode)
segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
segmentEditorWidget.setSegmentationNode(segmentationNode)
segmentEditorWidget.setMasterVolumeNode(volume)

segmentEditorWidget.setActiveEffectByName("Paint")
# This is how you would automatically set the brush size, but documentation is limited
# see source code here: https://github.com/Slicer/Slicer/blob/7cf1daed5654dd57dccae3f8b0c7f6f758aeb000/Modules/Loadable/Segmentations/EditorEffects/qSlicerSegmentEditorPaintEffect.cxx
#paintEffect = segmentEditorWidget.activeEffect()
#paintEffect.setParameter("BrushAbsoluteDiameter", 1)

# Change the names of the segmentations
segmentation = segmentationNode.GetSegmentation()
segment0 = segmentation.GetSegment(segmentation.GetNthSegmentID(0))
#segment0.SetColor(1,0,0)
segment0.SetName('Collagen')

segment1 = segmentation.GetSegment(segmentation.GetNthSegmentID(1))
#segment0.SetColor(0,1,0)
segment1.SetName('Background')

segment2 = segmentation.GetSegment(segmentation.GetNthSegmentID(2))
#segment0.SetColor(0,1,0)
segment2.SetName('Cells')
