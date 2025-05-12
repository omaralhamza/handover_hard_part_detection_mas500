import pyrealsense2 as rs

# Start RealSense pipeline with specific resolution,1920x1080,1280x720,640x480
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)  # Explicitly set 1920x1080

# Start the pipeline with the config
profile = pipeline.start(config)

# Get the intrinsics for the color stream
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Print factory intrinsics for 1920x1080,1280x720,640x480
print(f"Factory Intrinsics for 640x480:")
print(f"  fx = {intrinsics.fx}, fy = {intrinsics.fy}")
print(f"  cx = {intrinsics.ppx}, cy = {intrinsics.ppy}")

# Stop the pipeline
pipeline.stop()
