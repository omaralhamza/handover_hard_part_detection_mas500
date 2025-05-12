import cv2
import numpy as np
import pyrealsense2 as rs
import apriltag

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    # Align depth frame to color frame
    align = rs.align(rs.stream.color)
    
    # Initialize AprilTag detector
    detector = apriltag.Detector()
    
    try:
        while True:
            # Wait for a coherent pair of frames and align them
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # Detect AprilTags
            detections = detector.detect(gray)
            for detection in detections:
                # Draw the tag's corners
                corners = detection.corners.astype(int)
                for i in range(4):
                    pt1 = tuple(corners[i])
                    pt2 = tuple(corners[(i + 1) % 4])
                    cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)
                
                # Compute the center and query depth
                center = detection.center
                center_int = (int(center[0]), int(center[1]))
                depth = depth_frame.get_distance(center_int[0], center_int[1])
                cv2.circle(color_image, center_int, 5, (0, 0, 255), -1)
                cv2.putText(color_image, f"ID: {detection.tag_id}",
                            (center_int[0] - 10, center_int[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Print detection info to terminal
                print(f"Detection ID: {detection.tag_id}, Center: {center_int}, Depth: {depth:.3f} m, Corners: {corners.tolist()}")
            
            # Display the stream
            cv2.imshow("AprilTag Detection", color_image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
