# video.py


def process_video(file_path, table_name):
    embeddings = []
    texts = []
    try:
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                break

            # Process every nth frame (adjust n as needed for performance)
            n = 10  # Process every 10th frame
            if i % n == 0:
                # Image Embeddings
                _, buffer = cv2.imencode(".jpg", frame)  # Encode frame as JPG
                base64_image = base64.b64encode(buffer).decode("utf-8")
                image_info = {
                    "filename": f"frame_{i}.jpg",
                    "file_path": f"data:image/jpeg;base64,{base64_image}",
                }  # Use data URL for OpenAI
                image_embedding_response = get_llm_response(
                    "Describe this image.",
                    image=image_info,
                    model="gpt-4",
                    provider="openai",
                )  # Replace with your image embedding model
                if (
                    isinstance(image_embedding_response, dict)
                    and "error" in image_embedding_response
                ):
                    print(
                        f"Error generating image embedding: {image_embedding_response['error']}"
                    )
                else:
                    # Assuming your image embedding model returns a textual description
                    embeddings.append(image_embedding_response)
                    texts.append(f"Frame {i}: {image_embedding_response}")

        video.release()
        return embeddings, texts

    except Exception as e:
        print(f"Error processing video: {e}")
        return [], []  # Return empty lists in case of error

    import torch


from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np
import cv2

# Configuration
device = "cpu"
prompt = "A panda rolling down a hill"
num_inference_steps = 10
num_frames = 125
height = width = 256

# Load pipeline
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32
).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

print(f"Generating {num_frames} frames...")
output = pipe(
    prompt,
    num_inference_steps=num_inference_steps,
    num_frames=num_frames,
    height=height,
    width=width,
)


def save_frames_to_video(frames, output_path, fps=8):
    """Handle the specific 5D array format (1, num_frames, H, W, 3) with proper type conversion"""
    # Verify input format
    if not (
        isinstance(frames, np.ndarray) and frames.ndim == 5 and frames.shape[-1] == 3
    ):
        raise ValueError(
            f"Unexpected frame format. Expected 5D RGB array, got {frames.shape}"
        )

    # Remove batch dimension and convert to 0-255 uint8
    frames = (frames[0] * 255).astype(np.uint8)  # Shape: (num_frames, H, W, 3)

    # Get video dimensions
    height, width = frames.shape[1:3]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise IOError(f"Could not open video writer for {output_path}")

    # Write frames (convert RGB to BGR for OpenCV)
    for frame in frames:
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Successfully saved {frames.shape[0]} frames to {output_path}")


# Save video
output_path = "./spaceship_fixed.mp4"
save_frames_to_video(output.frames, output_path)
