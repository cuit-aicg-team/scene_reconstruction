from  AICGRender.AICGRenderInterface import OutdoorsceneReconstruction,IndoorsceneReconstruction,ObjectReconstruction
def test():
  outdoorsceneReconstruction = OutdoorsceneReconstruction()
  # outdoorsceneReconstruction.call_class_aicg_point_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/points")
  # outdoorsceneReconstruction.call_class_aicg_depth_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/depth")
  # outdoorsceneReconstruction.aicg_outdoor_mesh_reconstruct("output/points",iteration=100)

  indoorsceneReconstruction = IndoorsceneReconstruction()
  indoorsceneReconstruction.aicg_indoor_mesh_reconstruct(image_path_in = "input/001/out",iteration=40)

  objectReconstruction = ObjectReconstruction()
  objectReconstruction.aicg_object_mesh_reconstruct(image_path_in = "input/001/out",iteration=100)

test()
