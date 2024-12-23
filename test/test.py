from  AICGRender.AICGRenderInterface import OutdoorsceneReconstruction,IndoorsceneReconstruction,ObjectReconstruction
def test():
  outdoorsceneReconstruction = OutdoorsceneReconstruction()
  # outdoorsceneReconstruction.call_class_aicg_point_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/points")
  # outdoorsceneReconstruction.call_class_aicg_depth_create("/home/guowenwu/workspace/packaging_tutorial/input/rgb","output/depth")
  # outdoorsceneReconstruction.aicg_outdoor_mesh_reconstruct("output/points",iteration=10000,save_output_path="output/ourmesh.ply")
  # indoorsceneReconstruction = IndoorsceneReconstruction()
  # indoorsceneReconstruction.aicg_indoor_mesh_reconstruct(point_path_in = "input/001/out")
  objectReconstruction = ObjectReconstruction()
  objectReconstruction.aicg_object_mesh_reconstruct(point_path_in = "input/001/out")

test()
