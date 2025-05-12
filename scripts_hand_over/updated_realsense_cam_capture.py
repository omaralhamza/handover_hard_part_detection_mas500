import numpy as np
import cv2
import pyrealsense2 as rs
import os

def main():
    # Configure Intel RealSense pipeline for both depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 1920x1080 resolution

    # Start the pipeline
    pipeline.start(config)

    # Set the target directory to save images
    save_dir = "/home/omar/Cameras/other/Calibration_images/Calibration_images_640×480"

    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    img_counter = 1

    try:
        while True:
            # Wait for frames from the camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                print("No frame received.")
                continue

            # Convert the frames to a NumPy array
            color_image = np.asanyarray(color_frame.get_data())

            # Display the image
            cv2.imshow("Camera Feed", color_image)

            # Check for key presses
            key = cv2.waitKey(1)
            if key == ord('s'):  # If 's' is pressed, save the image
                img_name = f"color_img{img_counter}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, color_image)
                print(f"{img_name} saved at {save_dir}!")
                img_counter += 1

            elif key == ord('q'):  # If 'q' is pressed, quit the program
                print("Exiting...")
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
