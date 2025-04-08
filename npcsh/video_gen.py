def generate_video_diffusers(
    prompt,
    model,
    npc=None,
    device="cpu",
    output_path="",
    num_inference_steps=10,
    num_frames=125,
    height=256,
    width=256,
):

    import torch
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    import numpy as np
    import cv2

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float32
    ).to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

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
            isinstance(frames, np.ndarray)
            and frames.ndim == 5
            and frames.shape[-1] == 3
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

    os.makedirs("~/.npcsh/videos/")
    if output_path == "":

        output_path = "~/.npcsh/videos/" + prompt[0:8] + ".mp4"
    save_frames_to_video(output.frames, output_path)
    return output_path
