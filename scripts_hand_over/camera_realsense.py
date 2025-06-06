import pyrealsense2 as rs
import numpy as np

def initialize_camera():
    """
    Initializes the RealSense pipeline and returns (pipeline, align).
    """
    pipeline = rs.pipeline()
    config = rs.config()
    # Adjust resolutions & FPS as desired
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None, None

    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())  # 16-bit depth
    return color_image, depth_image, depth_frame

def stop_camera(pipeline):

    pipeline.stop()
