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